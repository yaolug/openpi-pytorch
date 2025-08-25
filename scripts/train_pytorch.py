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
- Memory optimizations: mixed precision training, gradient accumulation, and efficient data handling.
Requirements
- PyTorch >= 2.0, torch.distributed (NCCL for CUDA, Gloo for CPU).
- Multiple GPUs for DDP (optional).
- Network connectivity between nodes for multi-node training.
Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --ckpt_save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
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
- --cleanup_checkpoints: Clean up corrupted checkpoints during resume (keeps last 3 valid ones)
- --overwrite: Overwrite existing checkpoint directory (cannot be used with --resume)
Memory Optimization Parameters:
- --gradient_accumulation_steps: Number of steps to accumulate gradients (default: 1)
- --mixed_precision: Enable mixed precision training (default: True)
- --max_memory_usage: Maximum GPU memory usage in GB (default: None, auto-detect)
- --gradckpt: Enable gradient checkpointing for memory optimization
Notes
- The global batch size must be divisible by world size (number of processes).
- The data pipeline and transforms are identical to the JAX version and are controlled
  by the selected TrainConfig (e.g., `LeRobot*` configs for real datasets or `FakeDataConfig`).
- Supports Weights & Biases (wandb) logging for experiment tracking and visualization.
- Checkpoints include model state, optimizer state, and training metadata for complete resume capability.
- Checkpoints are saved in experiment-specific directories: <checkpoint_dir>/<step>/
- Resume functionality automatically finds the latest checkpoint for the specified experiment name.
- Checkpoint loading handles both PyTorch and JAX/Flax checkpoints for compatibility.
- For optimal multi-node performance, ensure high-bandwidth network connectivity (e.g., InfiniBand).
- Monitor GPU utilization and network bandwidth during multi-node training.
- Memory optimizations can significantly reduce GPU memory usage while maintaining training quality.
"""
import argparse
import dataclasses
import logging
import os
import platform
import time
import gc
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
		
		# Set up debugging environment variables for DDP issues
		if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
			os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
		
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
	print(f"data_conf: {data_conf}")
	dataset = _data.transform_dataset(dataset, data_conf)
	return dataset, data_conf


def collate_to_numpy(batch_list: list[Dict[str, Any]]) -> Dict[str, Any]:
	# Recursively stack leaves with numpy
	def stack_leaf(*xs):
		return np.stack([np.asarray(x) for x in xs], axis=0)

	# Memory-efficient collation
	result = torch.utils.data.default_collate(batch_list) if not isinstance(batch_list[0], dict) else _tree_map_multi(stack_leaf, batch_list)

	# Clear batch list from memory
	del batch_list

	return result


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

	# Memory-efficient conversion: convert to torch tensors and move to device in one step
	batch = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device), batch)

	# Convert to float32 for memory efficiency (avoid float64)
	batch['state'] = batch['state'].to(dtype=torch.float32)
	batch['actions'] = batch['actions'].to(dtype=torch.float32)

	# Clear numpy arrays from memory if they exist
	del jax

	return batch


def get_model_state_dict(model):
	"""Get state dict from model, handling DDP wrapper."""
	return model.module.state_dict() if isinstance(model, DDP) else model.state_dict()


def get_model_parameters(model):
	"""Get parameters from model, handling DDP wrapper."""
	return model.module.parameters() if isinstance(model, DDP) else model.parameters()


def save_checkpoint(model, optimizer, global_step, config, is_main, ckpt_save_interval=None, ema_model=None):
	"""Save a checkpoint with model state, optimizer state, EMA state, and metadata."""
	if not is_main:
		return

	# Use ckpt_save_interval if provided, otherwise use config.save_interval
	save_interval = ckpt_save_interval if ckpt_save_interval is not None else config.save_interval

	# Only save if it's time to save or if it's the final step
	if (global_step % save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
		# Ensure checkpoint_dir is a Path object and create the step-specific directory
		ckpt_dir = config.checkpoint_dir / f"{global_step}"
		ckpt_dir.mkdir(parents=True, exist_ok=True)

		# Save model state
		state_dict = get_model_state_dict(model)
		torch.save(state_dict, ckpt_dir / "pytorch_model.pt")

		# Save optimizer state
		torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

		# Save EMA state if available
		if ema_model is not None:
			torch.save(ema_model.state_dict(), ckpt_dir / "ema_model.pt")

		# Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
		metadata = {
			"global_step": global_step,
			"config": dataclasses.asdict(config),
			"timestamp": time.time(),
		}
		torch.save(metadata, ckpt_dir / "metadata.pt")

		logging.info(f"Saved checkpoint at step {global_step} -> {ckpt_dir}")

		# Log checkpoint to wandb
		if config.wandb_enabled:
			wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, ema_model=None):
	"""Load the latest checkpoint and return the global step."""
	checkpoint_steps = []
	for d in checkpoint_dir.iterdir():
		if d.is_dir() and d.name.isdigit():
			checkpoint_steps.append(int(d.name))
	
	if not checkpoint_steps:
		raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
	
	latest_step = max(checkpoint_steps)
	ckpt_dir = checkpoint_dir / f"{latest_step}"
	
	# Load model state with error handling
	try:
		model_state_dict = torch.load(ckpt_dir / "pytorch_model.pt", map_location=device, weights_only=False)
		(model.module if isinstance(model, DDP) else model).load_state_dict(model_state_dict)
		logging.info(f"Successfully loaded model state from step {latest_step}")
	except Exception as e:
		logging.error(f"Failed to load model state from step {latest_step}: {e}")
		raise RuntimeError(f"Model checkpoint corrupted at step {latest_step}. Cannot resume training.")
	
	# Load optimizer state with error handling and fallback
	optimizer_loaded = False
	try:
		optimizer_state_dict = torch.load(ckpt_dir / "optimizer.pt", map_location=device, weights_only=False)
		optimizer.load_state_dict(optimizer_state_dict)
		optimizer_loaded = True
		logging.info(f"Successfully loaded optimizer state from step {latest_step}")
	except Exception as e:
		logging.warning(f"Failed to load optimizer state from step {latest_step}: {e}")
		logging.warning("Optimizer state corrupted. Will continue with fresh optimizer state.")
		# Reset optimizer to fresh state
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group.get('lr', 1e-4)  # Use default LR or current LR
		optimizer.zero_grad()
		optimizer_loaded = False
	
	# Load EMA state if available
	ema_loaded = False
	if ema_model is not None and (ckpt_dir / "ema_model.pt").exists():
		try:
			ema_state_dict = torch.load(ckpt_dir / "ema_model.pt", map_location=device, weights_only=False)
			ema_model.load_state_dict(ema_state_dict)
			ema_loaded = True
			logging.info(f"Successfully loaded EMA state from step {latest_step}")
		except Exception as e:
			logging.warning(f"Failed to load EMA state from step {latest_step}: {e}")
			logging.warning("EMA state corrupted. Will continue without EMA.")
			ema_loaded = False
	
	# Load metadata (weights_only=False needed for older checkpoints that might contain JAX/Flax objects)
	try:
		metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
		global_step = metadata.get("global_step", latest_step)
		logging.info(f"Successfully loaded metadata from step {latest_step}")
		return global_step
	except Exception as e:
		logging.warning(f"Failed to load metadata from checkpoint: {e}")
		logging.warning("Using checkpoint step number as global step")
		return latest_step


def get_latest_checkpoint_step(checkpoint_dir):
	"""Get the latest checkpoint step number from a checkpoint directory."""
	checkpoint_steps = []
	for d in checkpoint_dir.iterdir():
		if d.is_dir() and d.name.isdigit():
			checkpoint_steps.append(int(d.name))

	return max(checkpoint_steps) if checkpoint_steps else None


def validate_checkpoint_integrity(checkpoint_dir, step):
	"""Validate that a checkpoint at the given step is complete and uncorrupted."""
	ckpt_dir = checkpoint_dir / f"{step}"
	
	required_files = ["pytorch_model.pt", "optimizer.pt", "metadata.pt"]
	optional_files = ["ema_model.pt"]
	
	# Check if all required files exist
	for file_name in required_files:
		file_path = ckpt_dir / file_name
		if not file_path.exists():
			logging.warning(f"Required checkpoint file missing: {file_path}")
			return False
	
	# Try to validate file integrity by attempting to load them
	try:
		# Test model file
		device = torch.device("cpu")  # Use CPU for validation to avoid GPU memory issues
		model_state = torch.load(ckpt_dir / "pytorch_model.pt", map_location=device, weights_only=False)
		if not isinstance(model_state, dict):
			logging.warning(f"Model checkpoint file corrupted at step {step}")
			return False
		
		# Test optimizer file
		optimizer_state = torch.load(ckpt_dir / "optimizer.pt", map_location=device, weights_only=False)
		if not isinstance(optimizer_state, dict):
			logging.warning(f"Optimizer checkpoint file corrupted at step {step}")
			return False
		
		# Test metadata file
		metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
		if not isinstance(metadata, dict) or "global_step" not in metadata:
			logging.warning(f"Metadata checkpoint file corrupted at step {step}")
			return False
		
		logging.info(f"Checkpoint at step {step} validated successfully")
		return True
		
	except Exception as e:
		logging.warning(f"Checkpoint validation failed at step {step}: {e}")
		return False


def find_latest_valid_checkpoint(checkpoint_dir):
	"""Find the latest checkpoint that passes integrity validation."""
	checkpoint_steps = []
	for d in checkpoint_dir.iterdir():
		if d.is_dir() and d.name.isdigit():
			checkpoint_steps.append(int(d.name))
	
	if not checkpoint_steps:
		return None
	
	# Sort steps in descending order to check latest first
	checkpoint_steps.sort(reverse=True)
	
	for step in checkpoint_steps:
		if validate_checkpoint_integrity(checkpoint_dir, step):
			return step
	
	logging.error("No valid checkpoints found in directory")
	return None


def cleanup_corrupted_checkpoints(checkpoint_dir, keep_last_n=3):
	"""Clean up corrupted checkpoints, keeping only the last N valid ones."""
	checkpoint_steps = []
	for d in checkpoint_dir.iterdir():
		if d.is_dir() and d.name.isdigit():
			checkpoint_steps.append(int(d.name))
	
	if not checkpoint_steps:
		return
	
	# Sort steps in descending order
	checkpoint_steps.sort(reverse=True)
	
	valid_checkpoints = []
	corrupted_checkpoints = []
	
	# Validate all checkpoints
	for step in checkpoint_steps:
		if validate_checkpoint_integrity(checkpoint_dir, step):
			valid_checkpoints.append(step)
		else:
			corrupted_checkpoints.append(step)
	
	# Keep only the last N valid checkpoints
	checkpoints_to_keep = valid_checkpoints[:keep_last_n]
	checkpoints_to_remove = valid_checkpoints[keep_last_n:] + corrupted_checkpoints
	
	# Remove old valid checkpoints and all corrupted ones
	for step in checkpoints_to_remove:
		checkpoint_path = checkpoint_dir / f"{step}"
		try:
			import shutil
			shutil.rmtree(checkpoint_path)
			logging.info(f"Removed checkpoint at step {step}")
		except Exception as e:
			logging.warning(f"Failed to remove checkpoint at step {step}: {e}")
	
	logging.info(f"Checkpoint cleanup complete. Kept {len(checkpoints_to_keep)} valid checkpoints: {checkpoints_to_keep}")


def debug_unused_parameters(model, device):
	"""Debug function to identify unused parameters in the model."""
	if isinstance(model, DDP):
		model = model.module
	
	logging.info("Checking for potentially unused parameters...")
	
	# Get all parameter names and their indices
	param_info = {}
	idx = 0
	for name, param in model.named_parameters():
		if param.requires_grad:
			param_info[idx] = name
			idx += 1
	
	logging.info(f"Total trainable parameters: {len(param_info)}")
	
	# Check which parameters have gradients after a forward pass
	# This is a diagnostic function that can be called if needed
	return param_info


def check_model_parameters(model, device):
	"""Check for unused parameters and provide debugging information."""
	if isinstance(model, DDP):
		model = model.module
	
	total_params = 0
	used_params = 0
	
	for name, param in model.named_parameters():
		total_params += param.numel()
		if param.requires_grad:
			used_params += param.numel()
	
	logging.info(f"Model parameters: {total_params:,} total, {used_params:,} trainable")
	
	# Check for parameters that might be unused
	unused_params = []
	for name, param in model.named_parameters():
		if param.requires_grad and param.grad is None:
			unused_params.append(name)
	
	if unused_params:
		logging.warning(f"Found {len(unused_params)} parameters that might be unused:")
		for name in unused_params:  # Show first 10
			logging.warning(f"  - {name}")
		# if len(unused_params) > 10:
		# 	logging.warning(f"  ... and {len(unused_params) - 10} more")


def log_memory_usage(device, step, phase="unknown"):
	"""Log detailed memory usage information."""
	if not torch.cuda.is_available():
		return
	
	memory_allocated = torch.cuda.memory_allocated(device) / 1e9
	memory_reserved = torch.cuda.memory_reserved(device) / 1e9
	memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
	memory_free = memory_free / 1e9
	
	# Get more detailed memory info
	memory_stats = torch.cuda.memory_stats(device)
	max_memory_allocated = memory_stats.get('allocated_bytes.all.peak', 0) / 1e9
	max_memory_reserved = memory_stats.get('reserved_bytes.all.peak', 0) / 1e9
	
	# Get DDP info if available
	ddp_info = ""
	if dist.is_initialized():
		ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"
	
	logging.info(f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}")


def setup_memory_optimizations(model, device, enable_gradient_checkpointing=False):
	"""Setup memory optimization techniques for the model."""
	# Set memory optimization environment variables
	os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
	os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
	
	if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
		model.gradient_checkpointing_enable()
		logging.info("Enabled gradient checkpointing for memory optimization")

	# Enable memory efficient attention if available
	if hasattr(model, 'config') and hasattr(model.config, 'attention_mode'):
		model.config.attention_mode = 'flash_attention_2'
		logging.info("Enabled Flash Attention 2 for memory efficiency")

	# Set memory efficient settings
	if torch.cuda.is_available():
		# Enable memory efficient algorithms
		torch.backends.cudnn.benchmark = False  # Disable for memory efficiency
		torch.backends.cudnn.deterministic = True  # Enable for memory efficiency

		# Set memory fraction if needed
		if device.index is not None:
			torch.cuda.empty_cache()
			logging.info(f"Cleared CUDA cache for device {device.index}")
			
		logging.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce memory fragmentation")


def train_loop(config: _config.TrainConfig, resume: bool = False, ckpt_save_interval: int = None, gradient_accumulation_steps: int = 1, mixed_precision: bool = True, max_memory_usage: float = None, enable_gradient_checkpointing: bool = False, cleanup_checkpoints: bool = False):
	use_ddp, local_rank, device = setup_ddp()
	is_main = (not use_ddp) or (dist.get_rank() == 0)
	set_seed(config.seed, local_rank)

	# Memory optimization: Set memory fraction if specified
	if max_memory_usage is not None and torch.cuda.is_available():
		torch.cuda.set_per_process_memory_fraction(max_memory_usage / torch.cuda.get_device_properties(device).total_memory * 1e-9)

	# Initialize checkpoint directory and wandb
	resuming = False
	if resume:
		# Find checkpoint directory based on experiment name
		exp_checkpoint_dir = config.checkpoint_dir
		if exp_checkpoint_dir.exists():
			# Use validation to find the latest working checkpoint
			latest_step = find_latest_valid_checkpoint(exp_checkpoint_dir)
			if latest_step is not None:
				resuming = True
				logging.info(f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}")
				
				# Clean up corrupted checkpoints if requested
				if cleanup_checkpoints and is_main:
					logging.info("Cleaning up corrupted checkpoints...")
					cleanup_corrupted_checkpoints(exp_checkpoint_dir, keep_last_n=3)
			else:
				raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
		else:
			raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
	elif config.overwrite and config.checkpoint_dir.exists():
		import shutil
		shutil.rmtree(config.checkpoint_dir)
		logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

	# Create checkpoint directory with experiment name
	if not resuming:
		# For new runs, create experiment-specific checkpoint directory
		exp_checkpoint_dir = config.checkpoint_dir
		exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
		logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
	else:
		# For resume, checkpoint_dir is already set to the experiment directory
		logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

	# Initialize wandb (only on main process)
	if is_main:
		init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

	# Build dataset + sampler + loader
	dataset, data_conf = build_datasets(config)
	sampler = None
	if use_ddp:
		sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=True)

	# Reduce batch size for gradient accumulation
	effective_batch_size = config.batch_size // (dist.get_world_size() if use_ddp else 1)

	# Memory-efficient data loading with reduced pin_memory for large datasets
	pin_memory = False  # Disable pin_memory to reduce memory usage
	logging.info("Disabled pin_memory to reduce memory usage")

	loader = TorchDataLoader(dataset, batch_size=effective_batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=config.num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=collate_to_numpy)

	# Log sample images to wandb on first batch
	if is_main and config.wandb_enabled and not resuming:
		sample_batch = next(iter(loader))
		sample_batch = batch_to_torch(sample_batch, device)

		# Create sample images for wandb
		images_to_log = []
		# Get batch size from the first image tensor
		batch_size = next(iter(sample_batch['image'].values())).shape[0]
		for i in range(min(5, batch_size)):
			# Concatenate all camera views horizontally for this batch item
			img_concatenated = torch.cat([img[i] for img in sample_batch['image'].values()], axis=1)
			img_concatenated = img_concatenated.cpu().numpy()
			images_to_log.append(wandb.Image(img_concatenated))

		wandb.log({"camera_views": images_to_log}, step=0)

		# Clear sample batch from memory
		del sample_batch, images_to_log
		torch.cuda.empty_cache() if torch.cuda.is_available() else None

		# Reset the loader iterator
		loader = TorchDataLoader(dataset, batch_size=effective_batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=config.num_workers, pin_memory=pin_memory, drop_last=True, collate_fn=collate_to_numpy)

	# Test gradient checkpointing with a small forward pass (moved to after model creation)

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
	
	# Apply memory optimizations
	setup_memory_optimizations(model, device, enable_gradient_checkpointing)
	
	# Log initial memory usage after model creation
	if is_main and torch.cuda.is_available():
		log_memory_usage(device, 0, "after_model_creation")
	
	# Log gradient checkpointing status if enabled
	if enable_gradient_checkpointing and is_main:
		if hasattr(model, 'get_gradient_checkpointing_status'):
			status = model.get_gradient_checkpointing_status()
			logging.info(f"Gradient checkpointing status: {status}")
			
			# Verify that gradient checkpointing is actually enabled
			if hasattr(model, 'is_gradient_checkpointing_enabled'):
				is_enabled = model.is_gradient_checkpointing_enabled()
				logging.info(f"Gradient checkpointing is enabled: {is_enabled}")
				
				# Check if we're in training mode
				logging.info(f"Model training mode: {model.training}")
				
				# Verify the underlying models have gradient checkpointing enabled
				if hasattr(model, 'paligemma_with_expert'):
					if hasattr(model.paligemma_with_expert, 'paligemma'):
						if hasattr(model.paligemma_with_expert.paligemma, 'language_model'):
							paligemma_gc = getattr(model.paligemma_with_expert.paligemma.language_model, 'gradient_checkpointing', False)
							logging.info(f"PaliGemma language model gradient checkpointing: {paligemma_gc}")
						
						if hasattr(model.paligemma_with_expert.paligemma, 'vision_tower'):
							vision_gc = getattr(model.paligemma_with_expert.paligemma.vision_tower, 'gradient_checkpointing', False)
							logging.info(f"PaliGemma vision tower gradient checkpointing: {vision_gc}")
					
					if hasattr(model.paligemma_with_expert, 'gemma_expert'):
						if hasattr(model.paligemma_with_expert.gemma_expert, 'model'):
							gemma_gc = getattr(model.paligemma_with_expert.gemma_expert.model, 'gradient_checkpointing', False)
							logging.info(f"Gemma expert model gradient checkpointing: {gemma_gc}")
		else:
			logging.info("Gradient checkpointing enabled but status check not available")
	
	# Test gradient checkpointing with a small forward pass
	if is_main and enable_gradient_checkpointing:
		logging.info("Testing gradient checkpointing with a small forward pass...")
		try:
			# Create a small test batch
			test_batch = next(iter(loader))
			test_batch = batch_to_torch(test_batch, device)
			test_actions = test_batch["actions"]
			
			# Record memory before forward pass
			if torch.cuda.is_available():
				memory_before = torch.cuda.memory_allocated(device) / 1e9
				logging.info(f"Memory before test forward pass: {memory_before:.2f}GB")
			
			# Do a test forward pass
			with torch.no_grad():
				test_observation = _model.Observation.from_dict(test_batch)
				test_losses = model(test_observation, test_actions)
			
			# Record memory after forward pass
			if torch.cuda.is_available():
				memory_after = torch.cuda.memory_allocated(device) / 1e9
				logging.info(f"Memory after test forward pass: {memory_after:.2f}GB")
				logging.info(f"Memory difference: {memory_after - memory_before:.2f}GB")
			
			# Clear test data
			del test_batch, test_actions, test_observation, test_losses
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				gc.collect()
			
			logging.info("Gradient checkpointing test completed successfully")
		except Exception as e:
			logging.warning(f"Gradient checkpointing test failed: {e}")
			logging.warning("Continuing with training...")
	
	if use_ddp:
		# Enable unused parameter detection to handle cases where some parameters don't participate in loss
		model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)

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
		try:
			ema_model = PI0Pytorch(model_cfg).to(device)
			
			# Get the correct state dict from the main model
			main_model_state_dict = get_model_state_dict(model)
			
			# Load the state dict into EMA model
			ema_model.load_state_dict(main_model_state_dict)
			ema_model.eval()
			logging.info(f"Initialized EMA with decay {config.ema_decay}")
		except Exception as e:
			logging.error(f"Failed to initialize EMA model: {e}")
			logging.error("Continuing without EMA...")
			ema_model = None

	# Load checkpoint if resuming
	global_step = 0
	if resuming:
		global_step = load_checkpoint(model, optim, config.checkpoint_dir, device, ema_model)
		logging.info(f"Resumed training from step {global_step}")

	def lr_schedule(step: int):
		if step < warmup_steps:
			return peak_lr * (step + 1) / warmup_steps
		# cosine decay
		progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
		cos = 0.5 * (1 + np.cos(np.pi * progress))
		return end_lr + (peak_lr - end_lr) * cos

	# Enable mixed precision training for memory optimization
	scaler = torch.amp.GradScaler(enabled=mixed_precision and torch.cuda.is_available())
	
	# Set memory efficient settings
	if torch.cuda.is_available():
		# Enable memory efficient algorithms
		torch.backends.cudnn.benchmark = False  # Disable for memory efficiency
		torch.backends.cudnn.deterministic = True  # Enable for memory efficiency
		
		# Set memory fraction if needed
		if device.index is not None:
			torch.cuda.empty_cache()
			logging.info(f"Cleared CUDA cache for device {device.index}")

	model.train()
	start_time = time.time()
	infos = []  # Collect stats over log interval
	if is_main:
		logging.info(f"Running on: {platform.node()} | world_size={dist.get_world_size() if use_ddp else 1}")
		logging.info(f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}")
		logging.info(f"Memory optimizations: gradient_accumulation_steps={gradient_accumulation_steps}, mixed_precision={mixed_precision}, gradient_checkpointing={enable_gradient_checkpointing}")
		logging.info(f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}")
		logging.info(f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}")
		if config.ema_decay is not None:
			logging.info(f"EMA decay: {config.ema_decay}")

	# Training loop - iterate until we reach num_train_steps
	pbar = tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main) if is_main else None

	# Check model parameters after first few steps when gradients are available
	parameters_checked = False

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

			# Update LR
			for pg in optim.param_groups:
				pg["lr"] = lr_schedule(global_step)

			# Forward pass with mixed precision
			observation = _model.Observation.from_dict(batch)
			try:
				with torch.amp.autocast('cuda', enabled=mixed_precision and torch.cuda.is_available()):
					losses = model(observation, actions)
					# Ensure losses is a tensor and handle different return types
					if isinstance(losses, (list, tuple)):
						losses = torch.stack(losses)
					elif not isinstance(losses, torch.Tensor):
						losses = torch.tensor(losses, device=device, dtype=torch.float32)
					
					loss = losses.mean() / gradient_accumulation_steps  # Scale loss for gradient accumulation
					
					# Debug gradient checkpointing on first few steps
					if global_step < 5 and is_main:
						if hasattr(model, 'is_gradient_checkpointing_enabled'):
							gc_enabled = model.is_gradient_checkpointing_enabled()
							logging.info(f"Step {global_step}: Gradient checkpointing enabled: {gc_enabled}")
							if torch.cuda.is_available():
								log_memory_usage(device, global_step, "after_forward")
			except RuntimeError as e:
				if "Expected to have finished reduction" in str(e) or "did not receive grad" in str(e):
					logging.error(f"DDP error on rank {dist.get_rank() if use_ddp else 0}: {e}")
					logging.error("This usually indicates unused parameters in the model.")
					logging.error("Try setting TORCH_DISTRIBUTED_DEBUG=DETAIL for more information.")
					raise
				else:
					raise

			# Backward pass with gradient scaling
			scaler.scale(loss).backward()
			
			# Aggressive memory cleanup after backward pass
			if torch.cuda.is_available():
				# Clear intermediate activations that might still be in memory
				torch.cuda.empty_cache()
				gc.collect()
				
				# Log memory usage after backward pass for debugging
				if global_step < 5 and is_main:
					log_memory_usage(device, global_step, "after_backward")

			# Gradient accumulation logic
			if (global_step + 1) % gradient_accumulation_steps == 0:
				# Unscale gradients for clipping
				scaler.unscale_(optim)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

				# Optimizer step
				scaler.step(optim)
				scaler.update()
				optim.zero_grad(set_to_none=True)
				
				# Clear gradients more aggressively
				for param in model.parameters():
					if param.grad is not None:
						param.grad.detach_()
						param.grad = None

				# Update EMA if enabled
				if ema_model is not None:
					try:
						with torch.no_grad():
							# Get parameters from the correct model structure
							main_model_params = get_model_parameters(model)
							for param, ema_param in zip(main_model_params, ema_model.parameters()):
								ema_param.data.mul_(config.ema_decay).add_(param.data, alpha=1 - config.ema_decay)
					except Exception as e:
						logging.warning(f"Failed to update EMA model: {e}")
						# Continue training without EMA update

			# # Check model parameters after first few steps when gradients are available
			# if not parameters_checked and global_step >= 16510 and is_main:
			# 	check_model_parameters(model, device)
			# 	parameters_checked = True

			# Collect stats (only on accumulation steps)
			if (global_step + 1) % gradient_accumulation_steps == 0 and is_main:
				infos.append({
					"loss": loss.item() * gradient_accumulation_steps,  # Unscale for logging
					"learning_rate": optim.param_groups[0]['lr'],
				})

			if is_main and (global_step % config.log_interval == 0) and (global_step + 1) % gradient_accumulation_steps == 0:
				elapsed = time.time() - start_time

				# Average stats over log interval
				avg_loss = sum(info["loss"] for info in infos) / len(infos)
				avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

				logging.info(f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s")

				# Log to wandb
				if config.wandb_enabled and len(infos) > 1:
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
					'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
					'lr': f'{optim.param_groups[0]["lr"]:.2e}',
					'step': global_step
				})
            
			# Memory cleanup after each batch
			del batch, actions, observation, losses, loss
			
			# More aggressive memory cleanup
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				# Force garbage collection
				gc.collect()
				
				# Log memory usage for debugging gradient checkpointing
				if is_main and global_step % 100 == 0:
					memory_allocated = torch.cuda.memory_allocated(device) / 1e9
					memory_reserved = torch.cuda.memory_reserved(device) / 1e9
					logging.info(f"Step {global_step}: GPU memory allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB")

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

	# Parse additional command line arguments for memory optimization
	import argparse
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--resume", action="store_true", default=False,
						help="Resume training from the latest checkpoint for the experiment (handles both PyTorch and JAX checkpoints)")
	parser.add_argument("--cleanup_checkpoints", action="store_true", default=False,
						help="Clean up corrupted checkpoints during resume (keeps last 3 valid checkpoints)")
	parser.add_argument("--ckpt_save_interval", type=int, default=None, 
						help="Interval for saving checkpoints (overrides config.save_interval)")
	parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
						help="Number of steps to accumulate gradients (default: 1)")
	parser.add_argument("--mixed_precision", action="store_true", default=False,
						help="Enable mixed precision training (default: True)")
	parser.add_argument("--no_mixed_precision", action="store_true", default=True,
						help="Disable mixed precision training")
	parser.add_argument("--max_memory_usage", type=float, default=None,
						help="Maximum GPU memory usage in GB (default: None, auto-detect)")
	parser.add_argument("--gradckpt", action="store_true", default=False,
						help="Enable gradient checkpointing for memory optimization")
	parser.add_argument("--ddp_debug_level", type=str, default="INFO", choices=["INFO", "DETAIL", "OFF"],
						help="DDP debugging level (default: INFO)")
	args, _ = parser.parse_known_args()
	
	# Handle mixed precision flag
	mixed_precision = args.mixed_precision and not args.no_mixed_precision
	
	# Set DDP debug level
	if args.ddp_debug_level != "OFF":
		os.environ["TORCH_DISTRIBUTED_DEBUG"] = args.ddp_debug_level

	train_loop(config, 
			   resume=args.resume,
			   ckpt_save_interval=args.ckpt_save_interval,
			   gradient_accumulation_steps=args.gradient_accumulation_steps,
			   mixed_precision=mixed_precision,
			   max_memory_usage=args.max_memory_usage,
			   enable_gradient_checkpointing=True,
			   cleanup_checkpoints=args.cleanup_checkpoints)


if __name__ == "__main__":
	main()
