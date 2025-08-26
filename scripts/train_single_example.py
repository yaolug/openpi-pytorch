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
    --jax_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid \
    --pytorch_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch

# Test pi0_aloha_sim model
python scripts/train_single_example.py \
    --model_name pi0_aloha_sim \
    --jax_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim \
    --pytorch_checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch

This script:
- Creates a fixed example with deterministic random values
- Uses the same noise and time values for both JAX and PyTorch
- Disables preprocessing for fair comparison
- Compares losses between implementations
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

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
import openpi.training.config


def setup_logging():
    """Setup logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


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


def test_pytorch_single_example(noise, time, model_name, pytorch_checkpoint_dir):
    """Test PyTorch training on single example."""
    print("\n=== Testing PyTorch on Single Example ===")

    # Create model using the training config
    train_config = openpi.training.config.get_config(model_name)
    model = PI0Pytorch(train_config.model)  # Use train_config.model instead of train_config

    # Load pre-trained weights
    pytorch_checkpoint_dir = pytorch_checkpoint_dir + "/model.safetensors"
    print(f"Loading PyTorch weights from: {pytorch_checkpoint_dir}")

    safetensors.torch.load_model(model, pytorch_checkpoint_dir)

    # Create fixed example
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
    observation = _model.Observation.from_dict(pytorch_example)
    actions = pytorch_example["actions"]

    print(f"Observation state shape: {observation.state.shape}")
    print(f"Observation state dtype: {observation.state.dtype}")
    print(f"Actions shape: {actions.shape}")
    print(f"Actions dtype: {actions.dtype}")
    print(f"Noise shape: {noise_tensor.shape}, dtype: {noise_tensor.dtype}")
    print(f"Time shape: {time_tensor.shape}, dtype: {time_tensor.dtype}")

    # Test forward pass with fixed noise and time
    model.eval()
    with torch.no_grad():
        try:
            # Use mock to disable preprocessing
            with patch('openpi.models.model.preprocess_observation_pytorch', side_effect=mock_preprocess_observation_pytorch):
                losses = model(observation, actions, noise=noise_tensor, time=time_tensor)
                print(f"PyTorch forward pass successful!")
                print(f"Losses shape: {losses.shape}")
                print(f"Losses dtype: {losses.dtype}")
                mean_loss = losses.to(torch.float32).mean().item()
                print(f"Mean loss: {mean_loss:.6f}")
                return True, losses
        except Exception as e:
            print(f"PyTorch forward pass failed: {e}")
            return False, None


def test_jax_single_example(noise, time, model_name, jax_checkpoint_dir):
    """Test JAX training on single example."""
    print("\n=== Testing JAX on Single Example ===")

    # Create model using the training config
    train_config = openpi.training.config.get_config(model_name)
    rng = jax.random.key(42)
    model = train_config.model.create(rng)  # Use train_config.model instead of config.create

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

    # Create fixed example
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

    # Test forward pass with fixed noise and time
    try:
        # Use the modified compute_loss method that accepts external noise and time
        # Use mock to disable preprocessing
        with patch('openpi.models.model.preprocess_observation', side_effect=mock_preprocess_observation):
            losses = model.compute_loss(rng, observation, actions, train=False, noise=noise_jax, time=time_jax)
            print(f"JAX forward pass successful!")
            print(f"Losses shape: {losses.shape}")
            print(f"Losses dtype: {losses.dtype}")
            mean_loss = jnp.mean(losses).item()
            print(f"Mean loss: {mean_loss:.6f}")
            return True, losses
    except Exception as e:
        print(f"JAX forward pass failed: {e}")
        return False, None


def compare_losses(pytorch_loss, jax_loss):
    """Compare losses and compute relative differences."""
    if pytorch_loss is None or jax_loss is None:
        return

    print("\n" + "=" * 70)
    print("📊 LOSS COMPARISON")
    print("=" * 70)

    print(f"PyTorch loss: {pytorch_loss}")
    print(f"JAX loss: {jax_loss}")

    # Additional tensor analysis if both are tensors
    pytorch_loss = pytorch_loss.to(torch.float32)
    jax_loss = jax_loss.astype(jnp.float32)
    if hasattr(pytorch_loss, 'shape') and hasattr(jax_loss, 'shape'):
        print(f"\n📐 Tensor Analysis:")

        # Check if shapes match
        if pytorch_loss.shape == jax_loss.shape:
            print(f"✅ Tensor shapes match: {pytorch_loss.shape}")

            # Element-wise comparison
            if hasattr(pytorch_loss, 'flatten') and hasattr(jax_loss, 'flatten'):
                # Convert to numpy for element-wise analysis
                try:
                    pytorch_flat = pytorch_loss.detach().cpu().numpy().flatten()
                    jax_flat = jax_loss.flatten()

                    # Element-wise differences
                    element_diff = np.abs(pytorch_flat - jax_flat)
                    print(f"element_diff[0]: {element_diff[0:2048*816:2048]}")
                    max_element_diff = np.max(element_diff)
                    mean_element_diff = np.mean(element_diff)

                    print(f"  Max element-wise difference: {max_element_diff:.8f}")
                    print(f"  Mean element-wise difference: {mean_element_diff:.8f}")

                    # Element-wise relative differences
                    # Avoid division by zero by adding small epsilon
                    epsilon = 1e-12
                    pytorch_flat_safe = pytorch_flat + epsilon
                    jax_flat_safe = jax_flat + epsilon

                    # Compute relative differences for each element
                    rel_diff_pytorch_elements = (element_diff / np.abs(pytorch_flat_safe)) * 100
                    rel_diff_jax_elements = (element_diff / np.abs(jax_flat_safe)) * 100

                    # Compute mean of relative differences
                    mean_rel_diff_pytorch = np.mean(rel_diff_pytorch_elements)
                    mean_rel_diff_jax = np.mean(rel_diff_jax_elements)

                    print(f"  Mean relative difference (w.r.t. PyTorch elements): {mean_rel_diff_pytorch:.4f}%")
                    print(f"  Mean relative difference (w.r.t. JAX elements): {mean_rel_diff_jax:.4f}%")

                    # Count elements with significant differences
                    significant_threshold = 1e-4
                    significant_count = np.sum(element_diff > significant_threshold)
                    total_elements = len(element_diff)
                    significant_percentage = (significant_count / total_elements) * 100

                    print(f"  Elements with diff > {significant_threshold}: {significant_count}/{total_elements} ({significant_percentage:.2f}%)")

                    # Additional relative difference analysis
                    significant_rel_threshold = 1.0  # 1%
                    significant_rel_count_pytorch = np.sum(rel_diff_pytorch_elements > significant_rel_threshold)
                    significant_rel_count_jax = np.sum(rel_diff_jax_elements > significant_rel_threshold)

                    print(f"  Elements with rel diff > {significant_rel_threshold}% (w.r.t. PyTorch): {significant_rel_count_pytorch}/{total_elements} ({(significant_rel_count_pytorch/total_elements)*100:.2f}%)")
                    print(f"  Elements with rel diff > {significant_rel_threshold}% (w.r.t. JAX): {significant_rel_count_jax}/{total_elements} ({(significant_rel_count_jax/total_elements)*100:.2f}%)")

                except Exception as e:
                    print(f"  ⚠️  Could not perform element-wise analysis: {e}")
        else:
            print(f"❌ Tensor shapes don't match: PyTorch {pytorch_loss.shape} vs JAX {jax_loss.shape}")


def main():
    """Main function to test both implementations."""
    parser = argparse.ArgumentParser(description="Train on a single example for JAX vs PyTorch comparison")
    parser.add_argument("--model_name", type=str, default="pi05_droid", 
                       choices=["pi0_aloha_sim", "pi0_aloha_towel", "pi0_base", "pi05_droid", "pi0_droid", "pi0_libero", "pi05_libero"],
                       help="Model name to use")
    parser.add_argument("--jax_checkpoint_dir", type=str, required=True,
                       help="Directory containing JAX model checkpoints")
    parser.add_argument("--pytorch_checkpoint_dir", type=str, required=True,
                       help="Directory containing PyTorch model checkpoints")
    args = parser.parse_args()

    setup_logging()

    print("🚀 Testing Single Example Training for JAX vs PyTorch Comparison")
    print("=" * 70)
    print(f"📁 Model: {args.model_name}")
    print(f"📁 JAX checkpoint: {args.jax_checkpoint_dir}")
    print(f"📁 PyTorch checkpoint: {args.pytorch_checkpoint_dir}")
    print("🎯 Using fixed noise and time values for deterministic comparison...")
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

    # Test PyTorch
    pytorch_success, pytorch_losses = test_pytorch_single_example(noise, time, args.model_name, args.pytorch_checkpoint_dir)
    torch.cuda.empty_cache()

    # Test JAX
    jax_success, jax_losses = test_jax_single_example(noise, time, args.model_name, args.jax_checkpoint_dir)

    # Compare losses
    if pytorch_success and jax_success:
        compare_losses(pytorch_losses, jax_losses)

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)

    if pytorch_success and jax_success:
        print("✅ Both JAX and PyTorch implementations work on the single example!")
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
    print("3. If losses differ significantly, investigate the differences")
    print("4. Check if the noise and time handling is consistent between implementations")
    print("5. Use the same example in full training runs")
    print("6. Note: Preprocessing (image augmentations) is disabled for this comparison")


if __name__ == "__main__":
    main() 
