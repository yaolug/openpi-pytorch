#!/usr/bin/env python3
"""
PyTorch training entrypoint for PI0 with multi-GPU and multi-node (DDP) support.

This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Key features
- Uses the same TrainConfig/tyro CLI as the JAX script (see available configs in
  `src/openpi/training/config.py`).
- Supports multi-GPU and multi-node training via DistributedDataParallel (DDP).
- Cosine LR with warmup (parameters read from the selected config).
- AdamW optimizer and gradient clipping.
- Comprehensive checkpoint saving and resume mechanism with configurable intervals.
- Checkpoints saved on rank 0 to `config.checkpoint_dir/<step>/` containing model, optimizer, and metadata.

Requirements
- PyTorch >= 2.0, torch.distributed (NCCL for CUDA, Gloo for CPU).
- Multiple GPUs for DDP (optional).
- Network connectivity between nodes for multi-node training.

Usage

Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --ckpt_save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test

Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test

Multi-Node Training:
  # On master node (node 0):
  torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --rdzv_id=<unique_id> --rdzv_backend=c10d --rdzv_endpoint=<master_ip>:<port> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  
  # On worker nodes (node 1, 2, ...):
  torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --rdzv_id=<unique_id> --rdzv_backend=c10d --rdzv_endpoint=<master_ip>:<port> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  
  Example (2 nodes, 4 GPUs each):
  # Master node (192.168.1.100):
  torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=192.168.1.100:29400 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_multi_node
  
  # Worker node (192.168.1.101):
  torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=192.168.1.100:29400 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_multi_node

Multi-Node Setup Requirements:
1. Network connectivity: All nodes must be able to communicate on the specified port
2. Shared filesystem: All nodes must have access to the same dataset and checkpoint directories
3. Environment consistency: Same Python environment and dependencies on all nodes
4. Firewall configuration: Ensure the rendezvous port (e.g., 29400) is open between nodes
5. SSH access: Nodes should be able to SSH to each other (for torchrun coordination)

Environment Variables for Multi-Node:
- MASTER_ADDR: IP address of the master node (auto-set by torchrun)
- MASTER_PORT: Port for rendezvous (auto-set by torchrun)
- WORLD_SIZE: Total number of processes across all nodes
- RANK: Global rank of the process (0 to WORLD_SIZE-1)
- LOCAL_RANK: Local rank within the node (0 to nproc_per_node-1)
- NODE_RANK: Rank of the node (0 to nnodes-1)

Checkpoint Parameters:
- --ckpt_save_interval: Override the checkpoint save interval from config (e.g., --save_interval 500)
- --resume: Resume training from the latest checkpoint in the checkpoint directory
- --overwrite: Overwrite existing checkpoint directory (cannot be used with --resume)

Notes
- The global batch size must be divisible by world size (number of processes).
- The data pipeline and transforms are identical to the JAX version and are controlled
  by the selected TrainConfig (e.g., `LeRobot*` configs for real datasets or `FakeDataConfig`).
- Supports Weights & Biases (wandb) logging for experiment tracking and visualization.
- Checkpoints include model state, optimizer state, and training metadata for complete resume capability.
- For optimal multi-node performance, ensure high-bandwidth network connectivity (e.g., InfiniBand).
- Monitor GPU utilization and network bandwidth during multi-node training.
"""
import argparse
import dataclasses
import logging
import os
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm

import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.models.model as _model
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models.pi0_config import Pi0Config


def init_logging():
	level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

	class CustomFormatter(logging.Formatter):
		def format(self, record):
			record.levelname = level_mapping.get(record.levelname, record.levelname)
			return super().format(record)

	formatter = CustomFormatter(
		fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
		datefmt="%H:%M:%S",
	)
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	if not logger.handlers:
		ch = logging.StreamHandler()
		ch.setFormatter(formatter)
		logger.addHandler(ch)
	else:
		logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
	"""Initialize wandb logging."""
	if not enabled:
		wandb.init(mode="disabled")
		return

	ckpt_dir = config.checkpoint_dir
	if not ckpt_dir.exists():
		raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
	
	if resuming:
		run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
		wandb.init(id=run_id, resume="must", project=config.project_name)
	else:
		wandb.init(
			name=config.exp_name,
			config=dataclasses.asdict(config),
			project=config.project_name,
		)
		(ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
	world_size = int(os.environ.get("WORLD_SIZE", "1"))
	use_ddp = world_size > 1
	if use_ddp and not dist.is_initialized():
		backend = "nccl" if torch.cuda.is_available() else "gloo"
		dist.init_process_group(backend=backend, init_method="env://")
	local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
	device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		torch.cuda.set_device(device)
	return use_ddp, local_rank, device


def cleanup_ddp():
	if dist.is_initialized():
		dist.barrier()
		dist.destroy_process_group()


def set_seed(seed: int, local_rank: int):
	torch.manual_seed(seed + local_rank)
	np.random.seed(seed + local_rank)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
	# Reuse existing dataset + transforms pipeline
	data_conf = config.data.create(config.assets_dirs, config.model)
	dataset = _data.create_torch_dataset(data_conf, config.model.action_horizon, config.model)
	dataset = _data.transform_dataset(dataset, data_conf)
	return dataset, data_conf


def collate_to_numpy(batch_list: list[Dict[str, Any]]) -> Dict[str, Any]:
	# Recursively stack leaves with numpy
	def stack_leaf(*xs):
		return np.stack([np.asarray(x) for x in xs], axis=0)
	return torch.utils.data.default_collate(batch_list) if not isinstance(batch_list[0], dict) else _tree_map_multi(stack_leaf, batch_list)


def _tree_map_multi(func, batch_list):
	# batch_list is a list of dicts with same structure; reduce by zipping leaves
	def recurse(keys, items):
		if isinstance(items[0], dict):
			return {k: recurse(keys + [k], [it[k] for it in items]) for k in items[0].keys()}
		return func(*items)
	return recurse([], batch_list)


def batch_to_torch(batch: Dict[str, Any], device: torch.device) -> Tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
	# Maintain canonical image key order
	image_keys = _model.IMAGE_KEYS
	import jax
	batch = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device), batch)
	batch['state'] = batch['state'].to(dtype=torch.float32)
	batch['actions'] = batch['actions'].to(dtype=torch.float32)

	# batch["image"] = [torch.as_tensor(np.asarray(batch["image"][k]), device=device) for k in image_keys]
	# batch["image_mask"] = [torch.as_tensor(np.asarray(batch["image_mask"][k]), device=device, dtype=torch.bool) for k in image_keys]

	# lang_tokens = None
	# lang_masks = None
	# if batch.get("tokenized_prompt") is not None:
	# 	batch["tokenized_prompt"] = torch.as_tensor(np.asarray(batch["tokenized_prompt"]), device=device, dtype=torch.long)
	# 	batch["tokenized_prompt_mask"] = torch.as_tensor(np.asarray(batch["tokenized_prompt_mask"]), device=device, dtype=torch.bool)
	# else:
	# 	# Create zero-length placeholders if language is absent
	# 	bsize = batch["image"][0].shape[0]
	# 	lang_tokens = torch.zeros((bsize, 0), dtype=torch.long, device=device)
	# 	lang_masks = torch.zeros((bsize, 0), dtype=torch.bool, device=device)

	# batch["state"] = torch.as_tensor(np.asarray(batch["state"]), device=device, dtype=torch.float32)
	return batch


def save_checkpoint(model, optimizer, global_step, config, is_main, ckpt_save_interval=None, ema_model=None):
	"""Save a checkpoint with model state, optimizer state, EMA state, and metadata."""
	if not is_main:
		return
	
	# Use ckpt_save_interval if provided, otherwise use config.save_interval
	save_interval = ckpt_save_interval if ckpt_save_interval is not None else config.save_interval
	
	# Only save if it's time to save or if it's the final step
	if (global_step % save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
		ckpt_dir = os.path.join(config.checkpoint_dir, f"{global_step}")
		os.makedirs(ckpt_dir, exist_ok=True)
		
		# Save model state
		state_dict = (model.module if isinstance(model, DDP) else model).state_dict()
		torch.save(state_dict, os.path.join(ckpt_dir, "pytorch_model.pt"))
		
		# Save optimizer state
		torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
		
		# Save EMA state if available
		if ema_model is not None:
			torch.save(ema_model.state_dict(), os.path.join(ckpt_dir, "ema_model.pt"))
		
		# Save training metadata
		metadata = {
			"global_step": global_step,
			"config": dataclasses.asdict(config),
			"timestamp": time.time(),
		}
		torch.save(metadata, os.path.join(ckpt_dir, "metadata.pt"))
		
		logging.info(f"Saved checkpoint at step {global_step} -> {ckpt_dir}")
		
		# Log checkpoint to wandb
		if config.wandb_enabled:
			wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, config, device, ema_model=None):
	"""Load the latest checkpoint and return the global step."""
	checkpoint_steps = []
	for d in config.checkpoint_dir.iterdir():
		if d.is_dir() and d.name.isdigit():
			checkpoint_steps.append(int(d.name))
	
	if not checkpoint_steps:
		raise FileNotFoundError(f"No checkpoints found in {config.checkpoint_dir}")
	
	latest_step = max(checkpoint_steps)
	ckpt_dir = os.path.join(config.checkpoint_dir, f"{latest_step}")
	
	# Load model state
	model_state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.pt"), map_location=device)
	(model.module if isinstance(model, DDP) else model).load_state_dict(model_state_dict)
	
	# Load optimizer state
	optimizer_state_dict = torch.load(os.path.join(ckpt_dir, "optimizer.pt"), map_location=device)
	optimizer.load_state_dict(optimizer_state_dict)
	
	# Load EMA state if available
	if ema_model is not None and os.path.exists(os.path.join(ckpt_dir, "ema_model.pt")):
		ema_state_dict = torch.load(os.path.join(ckpt_dir, "ema_model.pt"), map_location=device)
		ema_model.load_state_dict(ema_state_dict)
		logging.info(f"Loaded EMA state from checkpoint")
	
	# Load metadata
	metadata = torch.load(os.path.join(ckpt_dir, "metadata.pt"), map_location=device)
	
	logging.info(f"Loaded checkpoint from step {latest_step} -> {ckpt_dir}")
	return metadata["global_step"]


def get_latest_checkpoint_step(config):
	"""Get the latest checkpoint step number."""
	checkpoint_steps = []
	for d in config.checkpoint_dir.iterdir():
		if d.is_dir() and d.name.isdigit():
			checkpoint_steps.append(int(d.name))
	
	return max(checkpoint_steps) if checkpoint_steps else None


def train_loop(config: _config.TrainConfig, ckpt_save_interval: int = None):
	use_ddp, local_rank, device = setup_ddp()
	is_main = (not use_ddp) or (dist.get_rank() == 0)
	set_seed(config.seed, local_rank)

	# Initialize checkpoint directory and wandb
	resuming = False
	if config.resume:
		# Check if checkpoint directory exists and has checkpoints
		if config.checkpoint_dir.exists():
			latest_step = get_latest_checkpoint_step(config)
			if latest_step is not None:
				resuming = True
				logging.info(f"Resuming from checkpoint directory: {config.checkpoint_dir} at step {latest_step}")
			else:
				raise FileNotFoundError(f"No checkpoints found in {config.checkpoint_dir} for resume")
		else:
			raise FileNotFoundError(f"Checkpoint directory {config.checkpoint_dir} does not exist for resume")
	elif config.overwrite and config.checkpoint_dir.exists():
		import shutil
		shutil.rmtree(config.checkpoint_dir)
		logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")
	
	# Create checkpoint directory
	config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
	
	# Initialize wandb (only on main process)
	if is_main:
		init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

	# Build dataset + sampler + loader
	dataset, data_conf = build_datasets(config)
	sampler = None
	if use_ddp:
		sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=True)
	loader = TorchDataLoader(dataset, batch_size=config.batch_size // (dist.get_world_size() if use_ddp else 1), shuffle=(sampler is None), sampler=sampler, num_workers=config.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_to_numpy)

	# Log sample images to wandb on first batch
	if is_main and config.wandb_enabled and not resuming:
		# sample_batch = next(iter(loader))
		# sample_batch = batch_to_torch(sample_batch, device)

		# # Create sample images for wandb
		# images_to_log = []
		# for i in range(min(5, sample_batch['image'][0].shape[0])):
		# 	# Concatenate all camera views horizontally
		# 	img_concatenated = torch.cat([img[i].cpu().numpy() for img in sample_batch['image']], axis=1)
		# 	# Convert from [-1, 1] to [0, 1] for wandb
		# 	img_concatenated = (img_concatenated + 1) / 2
		# 	# Convert to uint8 for wandb
		# 	img_concatenated = (img_concatenated * 255).astype(np.uint8)
		# 	images_to_log.append(wandb.Image(img_concatenated))
		
		# wandb.log({"camera_views": images_to_log}, step=0)
		
		# Reset the loader iterator
		loader = TorchDataLoader(dataset, batch_size=config.batch_size // (dist.get_world_size() if use_ddp else 1), shuffle=(sampler is None), sampler=sampler, num_workers=config.num_workers, pin_memory=True, drop_last=True, collate_fn=collate_to_numpy)

	# Build model
	if not isinstance(config.model, Pi0Config):
		# Convert dataclass to Pi0Config if needed
		model_cfg = Pi0Config(
			action_dim=config.model.action_dim,
			action_horizon=config.model.action_horizon,
			max_token_len=config.model.max_token_len,
			paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
			action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
			pi05=getattr(config.model, "pi05", False),
		)
	else:
		model_cfg = config.model

	model = PI0Pytorch(model_cfg).to(device)
	if use_ddp:
		model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)

	# Load weights from weight_loader if specified (for fine-tuning)
	if isinstance(config.weight_loader, str):
		weight_path = config.weight_loader
		logging.info(f"Loading weights from: {weight_path}")

		model_path = os.path.join(weight_path, "model.safetensors")
		from safetensors.torch import load_model
		load_model((model.module if isinstance(model, DDP) else model), model_path)
		logging.info(f"Loaded PyTorch weights from {weight_path}")

	# Optimizer + learning rate schedule from config
	warmup_steps = config.lr_schedule.warmup_steps
	peak_lr = config.lr_schedule.peak_lr
	decay_steps = config.lr_schedule.decay_steps
	end_lr = config.lr_schedule.decay_lr

	# Create optimizer with config parameters
	optim = torch.optim.AdamW(
		model.parameters(), 
		lr=peak_lr, 
		betas=(config.optimizer.b1, config.optimizer.b2), 
		eps=config.optimizer.eps,
		weight_decay=config.optimizer.weight_decay
	)

	# Initialize EMA if specified in config
	ema_model = None
	if config.ema_decay is not None:
		ema_model = PI0Pytorch(model_cfg).to(device)
		ema_model.load_state_dict(model.state_dict())
		ema_model.eval()
		logging.info(f"Initialized EMA with decay {config.ema_decay}")

	# Load checkpoint if resuming
	global_step = 0
	if resuming:
		global_step = load_checkpoint(model, optim, config, device, ema_model)
		logging.info(f"Resumed training from step {global_step}")

	def lr_schedule(step: int):
		if step < warmup_steps:
			return peak_lr * (step + 1) / warmup_steps
		# cosine decay
		progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
		cos = 0.5 * (1 + np.cos(np.pi * progress))
		return end_lr + (peak_lr - end_lr) * cos

	scaler = torch.amp.GradScaler('cuda', enabled=False)  # switch to True if using fp16

	model.train()
	start_time = time.time()
	infos = []  # Collect stats over log interval
	if is_main:
		logging.info(f"Running on: {platform.node()} | world_size={dist.get_world_size() if use_ddp else 1}")
		logging.info(f"Training config: batch_size={config.batch_size}, num_train_steps={config.num_train_steps}")
		logging.info(f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}")
		logging.info(f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}")
		if config.ema_decay is not None:
			logging.info(f"EMA decay: {config.ema_decay}")

	# Training loop - iterate until we reach num_train_steps
	pbar = tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main) if is_main else None
	
	while global_step < config.num_train_steps:
		if use_ddp:
			sampler.set_epoch(global_step // len(loader))
			
		for batch in loader:
			# Check if we've reached the target number of steps
			if global_step >= config.num_train_steps:
				break
				
			# Convert dict batch directly to torch tensors (bypass Observation.from_dict for PyTorch)
			batch = batch_to_torch(batch, device)
			actions = batch["actions"]
			# actions = torch.from_numpy(np.array(batch["actions"])).to(device=device, dtype=torch.float32)

			# Update LR
			for pg in optim.param_groups:
				pg["lr"] = lr_schedule(global_step)

			optim.zero_grad(set_to_none=True)

			observation = _model.Observation.from_dict(batch)
			with torch.amp.autocast('cuda', enabled=False):
				losses = model(observation, actions)
				loss = losses.mean()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
			optim.step()

			# Update EMA if enabled
			if ema_model is not None:
				with torch.no_grad():
					for param, ema_param in zip(model.parameters(), ema_model.parameters()):
						ema_param.data.mul_(config.ema_decay).add_(param.data, alpha=1 - config.ema_decay)

			# Collect stats
			if is_main:
				infos.append({
					"loss": loss.item(),
					"learning_rate": optim.param_groups[0]['lr'],
				})

			if is_main and (global_step % config.log_interval == 0):
				elapsed = time.time() - start_time
				
				# Average stats over log interval
				avg_loss = sum(info["loss"] for info in infos) / len(infos)
				avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
				
				logging.info(f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s")
				
				# Log to wandb
				if config.wandb_enabled:
					wandb.log({
						"loss": avg_loss,
						"learning_rate": avg_lr,
						"step": global_step,
						"time_per_step": elapsed / config.log_interval,
					}, step=global_step)
				
				start_time = time.time()
				infos = []  # Reset stats collection

			# Save checkpoint using the new mechanism
			save_checkpoint(model, optim, global_step, config, is_main, ckpt_save_interval, ema_model)

			global_step += 1
			
			# Update progress bar
			if pbar is not None:
				pbar.update(1)
				pbar.set_postfix({
					'loss': f'{loss.item():.4f}',
					'lr': f'{optim.param_groups[0]["lr"]:.2e}',
					'step': global_step
				})

	# Close progress bar
	if pbar is not None:
		pbar.close()

	# Finish wandb run
	if is_main and config.wandb_enabled:
		wandb.finish()
	
	cleanup_ddp()


def main():
	init_logging()
	config = _config.cli()
	
	# Parse additional command line arguments for checkpoint interval
	import argparse
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--ckpt_save_interval", type=int, default=None, 
						help="Interval for saving checkpoints (overrides config.save_interval)")
	args, _ = parser.parse_known_args()
	
	train_loop(config, ckpt_save_interval=args.ckpt_save_interval)


if __name__ == "__main__":
	main()
