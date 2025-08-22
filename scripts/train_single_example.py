"""
Train on a single example for debugging JAX vs PyTorch comparison.
This script creates a deterministic dataset with one example and trains on it
to help debug differences between JAX and PyTorch implementations.
"""

import logging
import numpy as np
import torch
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def setup_logging():
    """Setup logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_fixed_example():
    """Create a fixed example for debugging."""
    np.random.seed(42)

    batch_size = 1
    action_dim = 32
    action_horizon = 10
    image_size = 224
    max_token_len = 48

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


def test_pytorch_single_example(noise, time):
    """Test PyTorch training on single example."""
    print("\n=== Testing PyTorch on Single Example ===")

    # Create model
    config = Pi0Config(action_dim=32, action_horizon=10, pi05=True)
    model = PI0Pytorch(config)

    # Load pre-trained weights
    weight_path = "/home/jasonlu/.cache/openpi/openpi-assets-preview/checkpoints/pi05_base_pytorch2/model.safetensors"
    print(f"Loading PyTorch weights from: {weight_path}")

    from safetensors.torch import load_model
    load_model(model, weight_path)

    # Create fixed example
    example = create_fixed_example()

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
        #try:
        losses = model(observation, actions, noise=noise_tensor, time=time_tensor)
        print(f"PyTorch forward pass successful!")
        print(f"Losses shape: {losses.shape}")
        print(f"Losses dtype: {losses.dtype}")
        # mean_loss = losses.to(torch.float32).mean().item()
        # print(f"Mean loss: {mean_loss:.6f}")
        return True, losses
        # except Exception as e:
        #     print(f"PyTorch forward pass failed: {e}")
        #     return False, None


def test_jax_single_example(noise, time, debug_single_layer=False):
    """Test JAX training on single example."""
    print("\n=== Testing JAX on Single Example ===")

    # Create model
    config = Pi0Config(action_dim=32, action_horizon=10, pi05=True)
    if debug_single_layer:
        print("🔧 Debug mode: Using only 1 encoder layer")

    # Create a custom model with modified siglip depth for debugging
    if debug_single_layer:
        # Import the Pi0 model class
        from openpi.models.pi0 import Pi0
        import openpi.models.gemma as _gemma
        import openpi.models.siglip as _siglip
        import flax.nnx.bridge as nnx_bridge

        # Create the model manually with custom siglip variant
        rng = jax.random.key(42)
        rngs = flax.nnx.Rngs(rng)

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # Create LLM
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        # Create custom siglip model with depth=1
        # We'll use the same variant but override the depth parameter
        siglip_params = _siglip.decode_variant("So400m/14")
        siglip_params["depth"] = 1  # Override depth to 1 for debugging

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant=None,  # Don't use variant, use explicit params
                pool_type="none",
                scan=False,  # Disable scan for single layer
                dtype_mm=config.dtype,
                **siglip_params,  # Pass the modified parameters
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # Create the full model
        model = Pi0(config, rngs)
        # Replace the siglip model with our custom one
        model.PaliGemma.img = img

        print("🔧 Created single-layer SigLIP model (depth=1) for debugging...")
    else:
        rng = jax.random.key(42)
        model = config.create(rng)

    # Load pre-trained weights
    weight_path = "/home/jasonlu/.cache/openpi/openpi-assets-preview/checkpoints/pi05_base/params"
    print(f"Loading JAX weights from: {weight_path}")

    # try:
    # Use the same approach as in policy_config.py
    params = _model.restore_params(weight_path, dtype=jnp.bfloat16)

    # Filter params to only include the first encoder layer for debugging
    if debug_single_layer:
        filtered_params = {}

        # The parameters are nested, so we need to traverse the structure
        def filter_nested_params(params_dict, key_path=""):
            result = {}
            for key, value in params_dict.items():
                current_path = f"{key_path}.{key}" if key_path else key

                if isinstance(value, dict):
                    # Recursive case - traverse deeper
                    filtered_sub = filter_nested_params(value, current_path)
                    if filtered_sub:  # Only include if there are sub-parameters
                        result[key] = filtered_sub
                else:
                    # Leaf case - check if this parameter should be included
                    if 'Transformer' in current_path:
                        # Only keep the first encoder block (encoderblock_0) and encoder_norm
                        if 'encoderblock_0' in current_path or 'encoder_norm' in current_path:
                            result[key] = value
                    else:
                        # Keep all non-Transformer params
                        result[key] = value
            return result

        filtered_params = filter_nested_params(params)
        params_to_use = filtered_params
        print("✅ JAX weights loaded successfully (first layer only)!")
        print("⚠️  Note: Using So400m variant with depth=1 (modified from depth=27)")
        print("⚠️  Only the first layer weights will be used, others will be randomly initialized")

        # Debug: Show what parameters we have
        print(f"📋 Available parameters for single-layer model:")
        transformer_params = []
        for key in sorted(params_to_use.keys()):
            if 'Transformer' in key:
                transformer_params.append(key)
                print(f"    {key}: {params_to_use[key].shape}")

        if not transformer_params:
            print("    No Transformer parameters found! Let's see all keys:")
            for key in sorted(params_to_use.keys())[:20]:  # Show first 20 keys
                print(f"    {key}")

            # Let's also check if PaliGemma has nested structure
            if 'PaliGemma' in params_to_use:
                print("    Checking PaliGemma structure:")
                paligemma_params = params_to_use['PaliGemma']
                if hasattr(paligemma_params, 'keys'):
                    for subkey in sorted(paligemma_params.keys()):
                        print(f"      PaliGemma.{subkey}")
                        if hasattr(paligemma_params[subkey], 'keys'):
                            for subsubkey in sorted(paligemma_params[subkey].keys()):
                                print(f"        PaliGemma.{subkey}.{subsubkey}")
                                if hasattr(paligemma_params[subkey][subsubkey], 'keys'):
                                    for subsubsubkey in sorted(paligemma_params[subkey][subsubkey].keys()):
                                        if 'Transformer' in subsubsubkey:
                                            print(f"          PaliGemma.{subkey}.{subsubkey}.{subsubsubkey}")

        # The issue is that with scan=False, the model expects different parameter names
        # We need to map from encoderblock_0 to encoderblock in the nested structure
        def adapt_nested_params(params_dict, key_path=""):
            result = {}
            for key, value in params_dict.items():
                current_path = f"{key_path}.{key}" if key_path else key

                if isinstance(value, dict):
                    # Recursive case - traverse deeper
                    result[key] = adapt_nested_params(value, current_path)
                else:
                    # Leaf case - adapt the key if needed
                    new_key = key
                    if 'Transformer' in current_path and 'encoderblock_0' in key:
                        # Map encoderblock_0 to encoderblock for non-scan mode
                        new_key = key.replace('encoderblock_0', 'encoderblock')
                    result[new_key] = value
            return result

        adapted_params = adapt_nested_params(params_to_use)
        params_to_use = adapted_params
        print("🔄 Adapted parameter names for non-scan mode")
        print(f"  Example mapping: encoderblock_0 -> encoderblock")
    else:
        params_to_use = params
        print("✅ JAX weights loaded successfully!")

    # Apply the params to the model using NNX state management
    import flax.nnx as nnx
    graphdef, model_state = nnx.split(model)

    # Debug: Let me check what the model actually expects first
    print(f"🔍 Checking what the model expects...")
    try:
        print(f"📋 Model parameter structure:")
        model_transformer_params = []
        for key in sorted(model_state.keys()):
            if 'Transformer' in key:
                model_transformer_params.append(key)
                print(f"    {key}: shape {getattr(model_state[key], 'shape', 'no shape')}")

        if not model_transformer_params:
            print("    No Transformer parameters found in model! Let's see all keys:")
            for key in sorted(model_state.keys())[:20]:  # Show first 20 keys
                print(f"    {key}")

            # Let's also check if PaliGemma has nested structure in model
            if 'PaliGemma' in model_state:
                print("    Checking PaliGemma structure in model:")
                paligemma_state = model_state['PaliGemma']
                if hasattr(paligemma_state, 'keys'):
                    for subkey in sorted(paligemma_state.keys()):
                        print(f"      PaliGemma.{subkey}")
                        if hasattr(paligemma_state[subkey], 'keys'):
                            for subsubkey in sorted(paligemma_state[subkey].keys()):
                                print(f"        PaliGemma.{subkey}.{subsubkey}")
                                if hasattr(paligemma_state[subkey][subsubkey], 'keys'):
                                    for subsubsubkey in sorted(paligemma_state[subkey][subsubkey].keys()):
                                        if 'Transformer' in subsubsubkey:
                                            print(f"          PaliGemma.{subkey}.{subsubkey}.{subsubsubkey}")
    except Exception as e:
        print(f"    Could not inspect model parameters: {e}")

    # Now try to load parameters
    try:
        model_state.replace_by_pure_dict(params_to_use)
        model = nnx.merge(graphdef, model_state)
        print("✅ Parameters loaded successfully!")
    except Exception as e:
        print(f"❌ Parameter loading failed: {e}")
        print("🔄 Continuing with random initialization...")
        model = nnx.merge(graphdef, model_state)
    # except Exception as e:
    #     print(f"❌ Failed to load JAX weights: {e}")
    #     print("Continuing with random initialization...")

    # Create fixed example
    example = create_fixed_example()

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
    # try:
    # Use the modified compute_loss method that accepts external noise and time
    losses = model.compute_loss(rng, observation, actions, train=False, noise=noise_jax, time=time_jax)
    print(f"JAX forward pass successful!")
    print(f"Losses shape: {losses.shape}")
    print(f"Losses dtype: {losses.dtype}")
    mean_loss = jnp.mean(losses).item()
    print(f"Mean loss: {mean_loss:.6f}")
    return True, losses
    # except Exception as e:
    #     print(f"JAX forward pass failed: {e}")
    #     return False, None


def compare_losses(pytorch_loss, jax_loss):
    """Compare losses and compute relative differences."""
    if pytorch_loss is None or jax_loss is None:
        return

    print("\n" + "=" * 70)
    print("📊 LOSS COMPARISON")
    print("=" * 70)

    # # Handle tensor inputs by computing mean if needed
    # if hasattr(pytorch_loss, 'mean'):
    #     pytorch_mean = pytorch_loss.to(torch.float32).mean().item()
    #     pytorch_std = pytorch_loss.to(torch.float32).std().item()
    #     print(f"PyTorch loss tensor - Mean: {pytorch_mean:.8f}, Std: {pytorch_std:.8f}")
    #     print(f"PyTorch loss shape: {pytorch_loss.shape}")
    # else:
    #     pytorch_mean = float(pytorch_loss)
    #     pytorch_std = 0.0
    #     print(f"PyTorch loss scalar: {pytorch_mean:.8f}")

    # if hasattr(jax_loss, 'mean'):
    #     jax_mean = jax_loss.mean().item()
    #     jax_std = jax_loss.std().item()
    #     print(f"JAX loss tensor - Mean: {jax_mean:.8f}, Std: {jax_std:.8f}")
    #     print(f"JAX loss shape: {jax_loss.shape}")
    # else:
    #     jax_mean = float(jax_loss)
    #     jax_std = 0.0
    #     print(f"JAX loss scalar: {jax_mean:.8f}")



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
    setup_logging()

    print("🚀 Testing Single Example Training for JAX vs PyTorch Comparison")
    print("=" * 70)
    print("📁 Loading pre-trained weights for both models...")
    print("🎯 Using fixed noise and time values for deterministic comparison...")
    print("🔧 Debug mode: JAX model will use only 1 encoder layer for faster debugging...")

    # Generate fixed noise and time
    noise, time = create_fixed_noise_and_time(
        batch_size=1, 
        action_horizon=10, 
        action_dim=32
    )

    # Test PyTorch
    pytorch_success, pytorch_losses = test_pytorch_single_example(noise, time)
    torch.cuda.empty_cache()

    # Test JAX
    jax_success, jax_losses = test_jax_single_example(noise, time, debug_single_layer=False)

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


if __name__ == "__main__":
    main() 
