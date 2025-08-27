#!/usr/bin/env python3
"""
Train on a single example for debugging JAX vs PyTorch comparison.

This script creates a deterministic dataset with one example and trains on it
to help debug differences between JAX and PyTorch implementations.

Usage examples:

# Test pi05_droid model
python scripts/train_single_example.py \
    --model_name pi05_droid \
    --jax_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi05_droid \
    --pytorch_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi05_droid_pytorch

# Test pi0_droid model  
python scripts/train_single_example.py \
    --model_name pi0_droid \
    --jax_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi0_droid \
    --pytorch_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi0_droid_pytorch

# Test pi0_aloha_sim model
python scripts/train_single_example.py \
    --model_name pi0_aloha_sim \
    --jax_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi0_aloha_sim \
    --pytorch_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi0_aloha_sim_pytorch

# Test pi05_libero model with pickle file (FASTEST for debugging)
python scripts/train_single_example.py \
    --model_name pi05_libero \
    --jax_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi05_libero \
    --pytorch_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi05_libero_pytorch \
    --load_pickle ./libero_sample.pkl

# Test pi05_libero model with small dataset (RECOMMENDED for first-time setup)
python scripts/train_single_example.py \
    --model_name pi05_libero \
    --jax_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi05_libero \
    --pytorch_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi05_libero_pytorch \

Data Loading Options (in order of speed):
1. --load_pickle: Load from pickle file (instant, fastest for repeated debugging)


Setup Workflow:
1. First time: Run save_libero_sample.py to create pickle file (takes 10-30 minutes)
2. Subsequent runs: Use --load_pickle for instant loading (takes ~1 second)

This script:
- Loads a real example from the pi05_libero dataset using JAX data loader
- Uses the same noise and time values for both JAX and PyTorch
- Disables preprocessing for fair comparison
- Compares losses between implementations
- Tests forward pass, backward pass, and another forward pass
- Provides detailed analysis of differences
"""

import argparse
import logging
import numpy as np
import torch
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax
import safetensors
from unittest.mock import patch
import time
import signal
import sys
import optax

import openpi.models.model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
import openpi.training.config
from openpi.training.data_loader import create_data_loader
import openpi.training.optimizer
from openpi.shared import nnx_utils


def setup_logging():
    """Setup logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_sample_from_pickle(pickle_path: str):
    """Load a sample from a pickle file saved by save_libero_sample.py."""
    print(f"🔄 Loading sample from pickle file: {pickle_path}")
    start_time = time.time()
    
    try:
        import pickle
        with open(pickle_path, 'rb') as f:
            sample_data = pickle.load(f)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Sample loaded successfully in {elapsed_time:.2f} seconds")
        
        # Extract the data in the same format as create_fixed_example
        observation_data = sample_data['observation']
        
        # Return in the same format as create_fixed_example
        example = {
            "image": observation_data.get("image", {}),
            "image_mask": observation_data.get("image_mask", {}),
            "state": observation_data.get("state", np.array([])),
            "actions": observation_data.get("actions", np.array([])),
            "tokenized_prompt": observation_data.get("tokenized_prompt", np.array([])),
            "tokenized_prompt_mask": observation_data.get("tokenized_prompt_mask", np.array([])),
        }
        
        print(f"  - Image keys: {list(example['image'].keys()) if example['image'] else 'None'}")
        print(f"  - State shape: {example['state'].shape}")
        print(f"  - Actions shape: {example['actions'].shape}")
        
        return example
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Failed to load pickle file after {elapsed_time:.2f} seconds: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_fixed_example(model_config):
    """Create a fixed example for debugging."""
    np.random.seed(42)

    batch_size = 1
    action_dim = model_config.action_dim
    action_horizon = model_config.action_horizon
    image_size = 224
    max_token_len = model_config.max_token_len

    # Create fixed images
    images = {}
    for key in _model.IMAGE_KEYS:
        img = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float32)

        # Simple gradient pattern
        for i in range(image_size):
            for j in range(image_size):
                val = (i + j) / (2 * image_size) * 2 - 1
                img[0, i, j, :] = [val, val * 0.5, val * 0.25]

        images[key] = img

    # Create fixed state and actions
    state = np.random.randn(batch_size, action_dim).astype(np.float32) * 0.1
    actions = np.random.randn(batch_size, action_horizon, action_dim).astype(np.float32) * 0.1

    # Create fixed language tokens
    tokenized_prompt = np.random.randint(0, 1000, (batch_size, max_token_len), dtype=np.int32)
    tokenized_prompt_mask = np.ones((batch_size, max_token_len), dtype=bool)

    # Create image masks
    image_masks = {key: np.ones(batch_size, dtype=bool) for key in _model.IMAGE_KEYS}

    return {
        "image": images,
        "image_mask": image_masks,
        "state": state,
        "actions": actions,
        "tokenized_prompt": tokenized_prompt,
        "tokenized_prompt_mask": tokenized_prompt_mask,
    }


def create_fixed_noise_and_time(batch_size, action_horizon, action_dim):
    """Create fixed noise and time values for deterministic comparison."""
    np.random.seed(42)  # Use same seed for consistency

    # Create fixed noise
    noise = np.random.randn(batch_size, action_horizon, action_dim).astype(np.float32) * 0.1

    # Create fixed time values (beta distribution like in the models)
    time_beta = np.random.beta(1.5, 1.0, batch_size).astype(np.float32)
    time = time_beta * 0.999 + 0.001

    return noise, time


def mock_preprocess_observation(rng, observation, **kwargs):
    """Mock function that returns observation unchanged to disable preprocessing."""
    return observation


def mock_preprocess_observation_pytorch(observation, **kwargs):
    """Mock function that returns observation unchanged to disable preprocessing."""
    return observation


def test_pytorch_single_example(noise, time, model_name, pytorch_checkpoint_dir, load_pickle=None):
    """Test PyTorch training on single example."""
    print("\n=== Testing PyTorch on Single Example ===")

    # Create model using the training config
    train_config = openpi.training.config.get_config(model_name)
    model = PI0Pytorch(train_config.model)  # Use train_config.model instead of train_config

    # Load pre-trained weights
    pytorch_checkpoint_dir = pytorch_checkpoint_dir + "/model.safetensors"
    print(f"Loading PyTorch weights from: {pytorch_checkpoint_dir}")

    safetensors.torch.load_model(model, pytorch_checkpoint_dir)

    # Load data based on arguments
    if load_pickle:
        print(f"🎯 Loading sample from pickle file: {load_pickle}")
        example = load_sample_from_pickle(load_pickle)
    else:
        print("🎯 Using fixed example for testing")
        example = create_fixed_example(train_config.model)

    # Convert to PyTorch tensors
    pytorch_example = {}
    for key, value in example.items():
        if key == "image":
            # Convert channels-last [B, H, W, C] to channels-first [B, C, H, W] for PyTorch
            pytorch_example[key] = {}
            for k, v in value.items():
                # v is [B, H, W, C], convert to [B, C, H, W]
                v_tensor = torch.from_numpy(v)
                v_tensor = v_tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                pytorch_example[key][k] = v_tensor
        elif key == "image_mask":
            pytorch_example[key] = {k: torch.from_numpy(v) for k, v in value.items()}
        else:
            pytorch_example[key] = torch.from_numpy(value)

    # Convert noise and time to PyTorch tensors
    noise_tensor = torch.from_numpy(noise)
    time_tensor = torch.from_numpy(time)

    # Create observation
    observation_torch = _model.Observation.from_dict(pytorch_example)
    actions_torch = pytorch_example["actions"]

    print(f"Observation state shape: {observation_torch.state.shape}")
    print(f"Observation state dtype: {observation_torch.state.dtype}")
    print(f"Actions shape: {actions_torch.shape}")
    print(f"Actions dtype: {actions_torch.dtype}")
    print(f"Noise shape: {noise_tensor.shape}, dtype: {noise_tensor.dtype}")
    print(f"Time shape: {time_tensor.shape}, dtype: {time_tensor.dtype}")

    # Setup optimizer from config
    print(f"Setting up optimizer from config: {type(train_config.optimizer).__name__}")
    # Use the exact same optimizer creation code as in train_pytorch.py
    warmup_steps = train_config.lr_schedule.warmup_steps
    peak_lr = train_config.lr_schedule.peak_lr
    decay_steps = train_config.lr_schedule.decay_steps
    end_lr = train_config.lr_schedule.decay_lr
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=peak_lr, 
        betas=(train_config.optimizer.b1, train_config.optimizer.b2), 
        eps=train_config.optimizer.eps,
        weight_decay=train_config.optimizer.weight_decay
    )
    print(f"✅ Optimizer created: {type(optimizer).__name__}")
    print(f"  - Learning rate: {peak_lr}")
    print(f"  - Betas: ({train_config.optimizer.b1}, {train_config.optimizer.b2})")
    print(f"  - Epsilon: {train_config.optimizer.eps}")
    print(f"  - Weight decay: {train_config.optimizer.weight_decay}")
    
    # Define learning rate schedule function (same as in train_pytorch.py)
    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # Test forward pass, backward pass, and another forward pass
    model.train()  # Set to training mode for backward pass
    
    try:
        # Use mock to disable preprocessing
        with patch('openpi.models.model.preprocess_observation_pytorch', side_effect=mock_preprocess_observation_pytorch):
            # First forward pass
            print("🔄 First forward pass...")
            losses_1 = model(observation_torch, actions_torch, noise=noise_tensor, time=time_tensor)
            loss_1 = losses_1.mean()
            print(f"First forward pass successful! Loss: {loss_1.item():.6f}")
            
            # Backward pass
            print("🔄 Backward pass...")
            loss_1.backward()
            print("Backward pass successful!")
            
            # Check gradients
            total_grad_norm = 0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
                    param_count += 1
            total_grad_norm = total_grad_norm ** 0.5
            print(f"Gradient norm: {total_grad_norm:.6f} (from {param_count} parameters)")
            
            # Gradient clipping (same as in train_pytorch.py)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.optimizer.clip_gradient_norm)
            print(f"  - Gradient clipping applied, clipped norm: {grad_norm:.6f}")
            
            # Optimizer step to update parameters
            print("🔄 Optimizer step...")
            # Update learning rate using the schedule (same as in train_pytorch.py)
            current_lr = lr_schedule(0)  # Use step 0 for this test
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            print(f"  - Updated learning rate to: {current_lr:.2e}")
            
            optimizer.step()
            print("Optimizer step successful!")
            
            # Clear gradients for next iteration
            optimizer.zero_grad()
            
            # Second forward pass
            print("🔄 Second forward pass...")
            losses_2 = model(observation_torch, actions_torch, noise=noise_tensor, time=time_tensor)
            loss_2 = losses_2.mean()
            print(f"Second forward pass successful! Loss: {loss_2.item():.6f}")
            
            # Compare losses
            loss_diff = abs(loss_1.item() - loss_2.item())
            print(f"Loss difference between passes: {loss_diff:.8f}")
            
            return True, losses_1, losses_2
            
    except Exception as e:
        print(f"PyTorch forward/backward pass failed: {e}")
        return False, None, None


def test_jax_single_example(noise, time, model_name, jax_checkpoint_dir, load_pickle=None):
    """Test JAX training on single example."""
    print("\n=== Testing JAX on Single Example ===")

    # Create model using the training config
    train_config = openpi.training.config.get_config(model_name)
    rng = jax.random.key(42)
    model = train_config.model.create(rng)  # Use train_config.model instead of config.create

    jax_checkpoint_dir = jax_checkpoint_dir + '/params/'
    # Load pre-trained weights
    print(f"Loading JAX weights from: {jax_checkpoint_dir}")
    params = _model.restore_params(jax_checkpoint_dir, dtype=jnp.bfloat16)
    params_to_use = params
    print("✅ JAX weights loaded successfully!")

    # Apply the params to the model using NNX state management
    import flax.nnx as nnx
    graphdef, model_state = nnx.split(model)

    # Now try to load parameters
    try:
        model_state.replace_by_pure_dict(params_to_use)
        model = nnx.merge(graphdef, model_state)
        print("✅ Parameters loaded successfully!")
    except Exception as e:
        print(f"❌ Parameter loading failed: {e}")
        print("🔄 Continuing with random initialization...")
        model = nnx.merge(graphdef, model_state)

    # Load data based on arguments
    if load_pickle:
        print(f"🎯 Loading sample from pickle file: {load_pickle}")
        example = load_sample_from_pickle(load_pickle)
    else:
        print("🎯 Using fixed example for testing")
        example = create_fixed_example(train_config.model)

    # Convert to JAX arrays
    jax_example = {}
    for key, value in example.items():
        if key == "image":
            jax_example[key] = {k: jnp.array(v) for k, v in value.items()}
        elif key == "image_mask":
            jax_example[key] = {k: jnp.array(v) for k, v in value.items()}
        else:
            jax_example[key] = jnp.array(value)

    # Convert noise and time to JAX arrays
    noise_jax = jnp.array(noise)
    time_jax = jnp.array(time)

    # Create observation
    observation = _model.Observation.from_dict(jax_example)
    actions = jax_example["actions"]

    print(f"Observation state shape: {observation.state.shape}")
    print(f"Observation state dtype: {observation.state.dtype}")
    print(f"Actions shape: {actions.shape}")
    print(f"Actions dtype: {actions.dtype}")
    print(f"Noise shape: {noise_jax.shape}, dtype: {noise_jax.dtype}")
    print(f"Time shape: {time_jax.shape}, dtype: {time_jax.dtype}")

    # Get learning rate from config
    lr = train_config.lr_schedule.peak_lr if hasattr(train_config.lr_schedule, 'peak_lr') else 1e-4
    print(f"Using learning rate from config: {lr}")
    
    # Create optimizer using the exact same code as in train.py
    print("Setting up JAX optimizer from config...")
    tx = openpi.training.optimizer.create_optimizer(train_config.optimizer, train_config.lr_schedule, weight_decay_mask=None)
    print(f"✅ JAX optimizer created: {type(tx).__name__}")
    
    # Initialize optimizer state
    # Get trainable parameters from the model using NNX
    params = nnx.state(model)
    trainable_params = params.filter(train_config.trainable_filter)
    opt_state = tx.init(trainable_params)
    print(f"✅ Optimizer state initialized")

    # Test forward pass, backward pass, and another forward pass
    try:
        # Use mock to disable preprocessing
        with patch('openpi.models.model.preprocess_observation', side_effect=mock_preprocess_observation):
            # JIT compile the compute_loss method for memory efficiency
            print("🔄 JIT compiling compute_loss method...")
            jitted_compute_loss = nnx_utils.module_jit(model.compute_loss)
            
            # First forward pass
            print("🔄 First forward pass...")
            losses_1 = jitted_compute_loss(rng, observation, actions, train=True, noise=noise_jax, time=time_jax)
            loss_1 = losses_1.mean()
            print(f"First forward pass successful! Loss: {loss_1.item():.6f}")
            
            # Use the same approach as in train.py for gradient computation and parameter updates
            print("🔄 Computing gradients and updating parameters...")
            
            # Define loss function for gradient computation using JIT compiled method
            def loss_fn(model, rng, observation, actions):
                chunked_loss = jitted_compute_loss(rng, observation, actions, train=True, noise=noise_jax, time=time_jax)
                return jnp.mean(chunked_loss)
            
            # Filter out frozen params and compute gradients (same as train.py)
            diff_state = nnx.DiffState(0, train_config.trainable_filter)
            loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, rng, observation, actions)
            
            print("Gradients computed successfully!")
            
            # Check gradient norms using JAX's global_norm
            try:
                grad_norm = optax.global_norm(grads)
                print(f"Gradient norm: {grad_norm.item():.6f}")
            except Exception as e:
                print(f"Could not compute gradient norm: {e}")
                print("Continuing without gradient norm...")
            
            # Update parameters using optimizer (same as train.py)
            print("🔄 Updating parameters with optimizer...")
            params = nnx.state(model).filter(train_config.trainable_filter)
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            
            # Update the model in place (same as train.py)
            nnx.update(model, new_params)
            opt_state = new_opt_state
            print("Parameter update successful!")
            
            # Second forward pass
            print("🔄 Second forward pass...")
            losses_2 = jitted_compute_loss(rng, observation, actions, train=True, noise=noise_jax, time=time_jax)
            loss_2 = losses_2.mean()
            print(f"Second forward pass successful! Loss: {loss_2.item():.6f}")
            
            # Compare losses
            loss_diff = abs(loss_1.item() - loss_2.item())
            print(f"Loss difference between passes: {loss_diff:.8f}")
            
            return True, losses_1, losses_2
            
    except Exception as e:
        print(f"JAX forward/backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def compare_losses(pytorch_loss_1, pytorch_loss_2, jax_loss_1, jax_loss_2):
    """Compare losses and compute relative differences."""
    if pytorch_loss_1 is None or jax_loss_1 is None:
        return

    print("\n" + "=" * 70)
    print("📊 LOSS COMPARISON")
    print("=" * 70)

    print(f"PyTorch first pass loss: {pytorch_loss_1}")
    print(f"PyTorch second pass loss: {pytorch_loss_2}")
    print(f"JAX first pass loss: {jax_loss_1}")
    print(f"JAX second pass loss: {jax_loss_2}")

    # Additional tensor analysis if both are tensors
    pytorch_loss_1 = pytorch_loss_1.to(torch.float32)
    pytorch_loss_2 = pytorch_loss_2.to(torch.float32)
    jax_loss_1 = jax_loss_1.astype(jnp.float32)
    jax_loss_2 = jax_loss_2.astype(jnp.float32)
    
    if hasattr(pytorch_loss_1, 'shape') and hasattr(jax_loss_1, 'shape'):
        print(f"\n📐 Tensor Analysis:")

        # Check if shapes match
        if pytorch_loss_1.shape == jax_loss_1.shape:
            print(f"✅ Tensor shapes match: {pytorch_loss_1.shape}")

            # Element-wise comparison
            if hasattr(pytorch_loss_1, 'flatten') and hasattr(jax_loss_1, 'flatten'):
                # Convert to numpy for element-wise analysis
                try:
                    pytorch_flat_1 = pytorch_loss_1.detach().cpu().numpy().flatten()
                    pytorch_flat_2 = pytorch_loss_2.detach().cpu().numpy().flatten()
                    jax_flat_1 = jax_loss_1.flatten()
                    jax_flat_2 = jax_loss_2.flatten()

                    # Element-wise differences between implementations
                    element_diff_1 = np.abs(pytorch_flat_1 - jax_flat_1)
                    element_diff_2 = np.abs(pytorch_flat_2 - jax_flat_2)
                    
                    max_element_diff_1 = np.max(element_diff_1)
                    mean_element_diff_1 = np.mean(element_diff_1)
                    max_element_diff_2 = np.max(element_diff_2)
                    mean_element_diff_2 = np.mean(element_diff_2)

                    print(f"  First pass - Max element-wise difference: {max_element_diff_1:.8f}")
                    print(f"  First pass - Mean element-wise difference: {mean_element_diff_1:.8f}")
                    print(f"  Second pass - Max element-wise difference: {max_element_diff_2:.8f}")
                    print(f"  Second pass - Mean element-wise difference: {mean_element_diff_2:.8f}")

                    # Element-wise relative differences
                    # Avoid division by zero by adding small epsilon
                    epsilon = 1e-12
                    pytorch_flat_1_safe = pytorch_flat_1 + epsilon
                    jax_flat_1_safe = jax_flat_1 + epsilon
                    pytorch_flat_2_safe = pytorch_flat_2 + epsilon
                    jax_flat_2_safe = jax_flat_2 + epsilon

                    # Compute relative differences for each element
                    rel_diff_pytorch_1 = (element_diff_1 / np.abs(pytorch_flat_1_safe)) * 100
                    rel_diff_jax_1 = (element_diff_1 / np.abs(jax_flat_1_safe)) * 100
                    rel_diff_pytorch_2 = (element_diff_2 / np.abs(pytorch_flat_2_safe)) * 100
                    rel_diff_jax_2 = (element_diff_2 / np.abs(jax_flat_2_safe)) * 100

                    # Compute mean of relative differences
                    mean_rel_diff_pytorch_1 = np.mean(rel_diff_pytorch_1)
                    mean_rel_diff_jax_1 = np.mean(rel_diff_jax_1)
                    mean_rel_diff_pytorch_2 = np.mean(rel_diff_pytorch_2)
                    mean_rel_diff_jax_2 = np.mean(rel_diff_jax_2)

                    print(f"  First pass - Mean relative difference (w.r.t. PyTorch): {mean_rel_diff_pytorch_1:.4f}%")
                    print(f"  First pass - Mean relative difference (w.r.t. JAX): {mean_rel_diff_jax_1:.4f}%")
                    print(f"  Second pass - Mean relative difference (w.r.t. PyTorch): {mean_rel_diff_pytorch_2:.4f}%")
                    print(f"  Second pass - Mean relative difference (w.r.t. JAX): {mean_rel_diff_jax_2:.4f}%")

                    # Count elements with significant differences
                    significant_threshold = 1e-4
                    significant_count_1 = np.sum(element_diff_1 > significant_threshold)
                    significant_count_2 = np.sum(element_diff_2 > significant_threshold)
                    total_elements = len(element_diff_1)
                    significant_percentage_1 = (significant_count_1 / total_elements) * 100
                    significant_percentage_2 = (significant_count_2 / total_elements) * 100

                    print(f"  First pass - Elements with diff > {significant_threshold}: {significant_count_1}/{total_elements} ({significant_percentage_1:.2f}%)")
                    print(f"  Second pass - Elements with diff > {significant_threshold}: {significant_count_2}/{total_elements} ({significant_percentage_2:.2f}%)")

                except Exception as e:
                    print(f"  ⚠️  Could not perform element-wise analysis: {e}")
        else:
            print(f"❌ Tensor shapes don't match: PyTorch {pytorch_loss_1.shape} vs JAX {jax_loss_1.shape}")


def main():
    """Main function to test both implementations."""
    parser = argparse.ArgumentParser(description="Train on a single example for JAX vs PyTorch comparison")
    parser.add_argument("--model_name", type=str, default="pi05_libero", 
                       choices=["pi0_aloha_sim", "pi0_aloha_towel", "pi0_base", "pi05_droid", "pi0_droid", "pi0_libero", "pi05_libero"],
                       help="Model name to use")
    parser.add_argument("--jax_checkpoint_dir", type=str, required=True,
                       help="Directory containing JAX model checkpoints")
    parser.add_argument("--pytorch_checkpoint_dir", type=str, required=True,
                       help="Directory containing PyTorch model checkpoints")
    parser.add_argument("--load_pickle", type=str, default=None,
                       help="Load sample from pickle file (fastest option)")
    args = parser.parse_args()

    setup_logging()

    print("🚀 Testing Single Example Training for JAX vs PyTorch Comparison")
    print("=" * 70)
    print(f"📁 Model: {args.model_name}")
    print(f"📁 JAX checkpoint: {args.jax_checkpoint_dir}")
    print(f"📁 PyTorch checkpoint: {args.pytorch_checkpoint_dir}")
    
    print("🚫 Preprocessing disabled: Image augmentations and resizing are bypassed for fair comparison...")

    # Get model configuration
    train_config = openpi.training.config.get_config(args.model_name)
    model_config = train_config.model

    # Generate fixed noise and time
    noise, time = create_fixed_noise_and_time(
        batch_size=1, 
        action_horizon=model_config.action_horizon, 
        action_dim=model_config.action_dim
    )

    # Test JAX
    jax_success, jax_losses_1, jax_losses_2 = test_jax_single_example(
        noise, time, args.model_name, args.jax_checkpoint_dir, args.load_pickle
    )
    
    # Clear JAX memory
    jax.clear_caches()

    # Test PyTorch
    pytorch_success, pytorch_losses_1, pytorch_losses_2 = test_pytorch_single_example(
        noise, time, args.model_name, args.pytorch_checkpoint_dir, args.load_pickle
    )
    torch.cuda.empty_cache()

    

    # Compare losses
    if pytorch_success and jax_success:
        compare_losses(pytorch_losses_1, pytorch_losses_2, jax_losses_1, jax_losses_2)

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    if pytorch_success and jax_success:
        print("✅ Both JAX and PyTorch implementations work on the single example!")
        print("✅ Forward pass, backward pass, and second forward pass completed successfully!")
        print("🔍 Loss comparison completed above.")
    elif pytorch_success:
        print("❌ PyTorch works but JAX failed. Check JAX implementation.")
    elif jax_success:
        print("❌ JAX works but PyTorch failed. Check PyTorch implementation.")
    else:
        print("❌ Both implementations failed. Check the error messages above.")

    print("\n💡 Next steps:")
    print("1. Run this script to verify both implementations work")
    print("2. Analyze the loss comparison results above")
    print("3. Check if the backward pass gradients are reasonable")
    print("4. If losses differ significantly, investigate the differences")
    print("5. Check if the noise and time handling is consistent between implementations")
    print("6. Use the same example in full training runs")
    print("7. Note: Preprocessing (image augmentations) is disabled for this comparison")


if __name__ == "__main__":
    main() 
