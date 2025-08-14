#!/usr/bin/env python3
"""
Example script showing how to run inference with both JAX and PyTorch Pi0 models.

This demonstrates the basic usage patterns for both implementations.

pi0_droid
python inference_example.py --model_name pi0_droid --jax_checkpoint_dir /home/jasonlu/.cache/openpi/openpi-assets/checkpoints/pi0_droid --pytorch_checkpoint_dir /home/jasonlu/.cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch2

pi0_aloha_sim
python inference_example.py --model_name pi0_aloha_sim --jax_checkpoint_dir /home/jasonlu/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim --pytorch_checkpoint_dir /home/jasonlu/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch2

pi05
python inference_example.py --model_name pi05_droid --jax_checkpoint_dir /home/jasonlu/.cache/openpi/openpi-assets-preview/checkpoints/pi05_droid --pytorch_checkpoint_dir /home/jasonlu/.cache/openpi/openpi-assets-preview/checkpoints/pi05_droid_pytorch2

"""

import os
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ.pop("TORCH_LOGS", None)
os.environ.pop("TORCH_COMPILE_DEBUG", None)
os.environ.pop("TORCHDYNAMO_VERBOSE", None)
os.environ.pop("TRITON_DEBUG", None)
import logging
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)
import torch
torch._inductor.config.debug = False
torch._inductor.config.verbose_progress = False

# also make sure the older config flags aren’t printing
import torch._dynamo.config as dcfg
dcfg.verbose = False



import argparse
import numpy as np
import jax
import jax.numpy as jnp
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import time


def create_example_data(model_name: str = "pi0_aloha_sim") -> dict:
    """Create example input data matching the expected format."""
    
    if model_name == "pi0_aloha_sim":
        # Create example data for ALOHA sim environment (uses AlohaInputs format)
        example = {
            "images": {
                "cam_high": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
                "cam_low": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
            },
            "state": np.random.randn(14).astype(np.float32),  # 14 motors for ALOHA sim
            "prompt": "Pick up the cube and place it in the bin",
        }
    elif model_name == "pi0_aloha_towel":
        # Create example data for ALOHA towel task (uses AlohaInputs format)
        example = {
            "images": {
                "cam_high": np.random.randint(0, 256, size=(3, 224, 224), dtype=np.uint8),
                # Note: towel task typically only uses one camera
            },
            "state": np.random.randn(14).astype(np.float32),  # 14 motors for ALOHA
            "prompt": "Fold the towel neatly on the table",
        }
    elif model_name == "pi0_base":
        # Create example data for base model (uses basic observation format)
        example = {
            "image": {
                "base_0_rgb": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
                "left_wrist_0_rgb": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
                "right_wrist_0_rgb": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
            },
            "image_mask": {
                "base_0_rgb": True,
                "left_wrist_0_rgb": True,
                "right_wrist_0_rgb": False,  # This camera is often masked out
            },
            "state": np.random.randn(8).astype(np.float32),  # Joint + gripper positions
            "prompt": "Pick up the object and move it to the target location",
        }
    elif model_name == "pi0_droid" or model_name == "pi05_droid":
        # Create example data for droid policy (uses DroidInputs format)
        example = {
            "observation/exterior_image_1_left": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.random.randn(7).astype(np.float32),  # 7 joint positions
            "observation/gripper_position": np.random.randn(1).astype(np.float32),  # 1 gripper position
            "prompt": "Pick up the object and move it to the target location",
        }
    elif model_name == "pi0_libero" or model_name == "pi05_libero":
        # Create example data for libero policy (uses LiberoInputs format)
        example = {
            "observation/image": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8),
            "observation/state": np.random.randn(8).astype(np.float32),  # 8 joint positions
            "prompt": "do something",
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return example



def _print_jax_model_weights(policy) -> None:
    """Print all JAX model weights (keys, shapes, dtypes)."""
    try:
        from flax import nnx
        import flax.traverse_util as traverse_util
    except Exception as e:
        print(f"Could not import flax to print JAX weights: {e}")
        return

    try:
        _, state = nnx.split(policy._model)
        pure = state.to_pure_dict()
        flat = traverse_util.flatten_dict(pure, sep="/")
        print("\n=== JAX Model Weights ===")
        for key, value in flat.items():
            try:
                arr = np.asarray(value)
                if arr.size > 0:
                    first = arr.flat[0]
                    try:
                        first_val = first
                    except Exception:
                        first_val = float(first)
                    print(f"{key}: first={first_val.item()} dtype={first_val.dtype}")
                else:
                    print(f"{key}: first=<empty>")
            except Exception:
                print(f"{key}: first=<unavailable>")
    except Exception as e:
        print(f"Failed to print JAX model weights: {e}")


def _print_pytorch_model_weights(policy) -> None:
    """Print all PyTorch model weights (keys, shapes, dtypes)."""
    try:
        import torch
    except Exception as e:
        print(f"Could not import torch to print PyTorch weights: {e}")
        return

    try:
        state_dict = policy._model.state_dict()
        print("\n=== PyTorch Model Weights ===")
        for name, tensor in state_dict.items():
            try:
                if tensor.numel() > 0:
                    first_val = tensor.view(-1)[0]
                    print(f"{name}: first={first_val} dtype={first_val.dtype}")

                    if "action" in name or "state" in name:
                        print(f"{name}: tensor={first_val.to(torch.float16).to(torch.float32)}")
                        print(f"{name}: tensor={first_val.to(torch.bfloat16).to(torch.float32)}")
                else:
                    print(f"{name}: first=<empty>")
            except Exception:
                print(f"{name}: first=<unavailable>")
    except Exception as e:
        print(f"Failed to print PyTorch model weights: {e}")


def run_jax_inference_example(observation, model_name, checkpoint_dir):
    """Example of running inference with JAX Pi0 model."""
    print("=== JAX Pi0 Inference Example ===")

    try:
        import jax

        from openpi.models.pi0_config import Pi0Config
        from openpi.policies.policy import Policy

        config = _config.get_config(model_name)

        # Create trained policy
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)

        # Print all JAX weights
        _print_jax_model_weights(policy)

        rng_key = jax.random.key(42)
        noise_shape = (config.model.action_horizon, config.model.action_dim)  # Use model's expected dimension
        jax_noise = jax.random.normal(rng_key, noise_shape, dtype=jnp.float32)
        noise_np = np.array(jax_noise)
        policy._rng = rng_key

        # Run inference
        print("Running JAX inference...")
        result = policy.infer(observation, noise=noise_np)

        # Print results
        print("JAX inference completed!")
        print(f"  - Actions shape: {result['actions'].shape}")
        print(f"  - Actions range: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")

        return result, noise_np

    except ImportError as e:
        print(f"Failed to run JAX inference: {e}")
        return None

def run_pytorch_inference_example(observation, model_name, noise, checkpoint_dir):
    """Example of running inference with PyTorch Pi0 model."""
    print("\n=== PyTorch Pi0 Inference Example ===")

    try:
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.policies.policy import Policy

        config = _config.get_config(model_name)

        # Create trained policy
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)

        # Print all PyTorch weights
        # _print_pytorch_model_weights(policy)

        # Warm-up with 5 inference calls
        print("Running PyTorch inference...")
        print("  Warming up with 5 inference calls...")
        for _ in range(5):
            _ = policy.infer(observation, noise=noise)
        
        # Test inference with 5 calls and average timing
        print("  Testing with 5 inference calls...")
        times = []
        for i in range(5):
            t0 = time.perf_counter()
            result = policy.infer(observation, noise=noise)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        pytorch_time = np.mean(times)
        print(f"  Individual times: {[f'{t*1000:.2f}ms' for t in times]}")
        print(f"  Average time: {pytorch_time*1000:.2f} ms")

        # Print results
        print("PyTorch inference completed!")
        print(f"  - Actions shape: {result['actions'].shape}")
        print(f"  - Actions range: [{result['actions'].min():.3f}, {result['actions'].max():.3f}]")
        print(f"  - Inference time: {pytorch_time*1000:.2f} ms")

        return result, pytorch_time

    except ImportError as e:
        print(f"Failed to run PyTorch inference: {e}")
        return None, None


def compare_results(jax_result, pytorch_result):
    """Compare results from both implementations."""
    if jax_result is None or pytorch_result is None:
        print("Cannot compare results - one implementation failed")
        return

    print("\n=== Comparing Results ===")

    # Compare actions
    actions_diff = np.abs(jax_result["actions"] - pytorch_result["actions"])
    max_diff = np.max(actions_diff)
    mean_diff = np.mean(actions_diff)

    print(f"JAX actions: {jax_result['actions']}")
    print(f"PyTorch actions: {pytorch_result['actions']}")

    print("Actions comparison:")
    print(f"  - Max absolute difference: {max_diff:.6f}")
    print(f"  - Mean absolute difference: {mean_diff:.6f}")

    # Calculate relative differences
    relative_diff = np.abs((jax_result["actions"] - pytorch_result["actions"]) / pytorch_result["actions"])
    max_rel_diff = np.max(relative_diff)
    mean_rel_diff = np.mean(relative_diff)

    print(f"  - Max relative difference: {max_rel_diff:.6f}")
    print(f"  - Mean relative difference: {mean_rel_diff:.6f}")
    
    # Additional diagnostic info
    print(f"  - JAX actions stats: min={jax_result['actions'].min():.6f}, max={jax_result['actions'].max():.6f}, mean={jax_result['actions'].mean():.6f}")
    print(f"  - PyTorch actions stats: min={pytorch_result['actions'].min():.6f}, max={pytorch_result['actions'].max():.6f}, mean={pytorch_result['actions'].mean():.6f}")

    # Check if results are close with different tolerances
    if np.allclose(jax_result["actions"], pytorch_result["actions"], rtol=1e-5, atol=1e-6):
        print("✅ Results match within strict tolerance!")
    elif np.allclose(jax_result["actions"], pytorch_result["actions"], rtol=1e-4, atol=1e-5):
        print("⚠️  Results match within moderate tolerance (rtol=1e-4, atol=1e-5)")
    elif np.allclose(jax_result["actions"], pytorch_result["actions"], rtol=2e-2, atol=2e-3):
        print("⚠️  Results match within loose tolerance (rtol=2e-2, atol=2e-3)")
    else:
        print("❌ Results differ significantly even with loose tolerance!")


def run_jax_inference_compare_jit(observation, model_name, checkpoint_dir):
    """Run JAX inference both with JIT and without JIT, compare and time.

    Returns (jitted_result, nojit_result, noise_np, jitted_time_s, nojit_time_s)
    """
    print("\n=== JAX JIT vs No-JIT Comparison ===")
    try:
        # Common config and RNG/noise
        config = _config.get_config(model_name)
        rng_key = jax.random.key(42)
        noise_shape = (config.model.action_horizon, config.model.action_dim)
        jax_noise = jax.random.normal(rng_key, noise_shape, dtype=jnp.float32)
        noise_np = np.array(jax_noise)

        # JIT policy
        policy_jit = _policy_config.create_trained_policy(config, checkpoint_dir)
        policy_jit._rng = rng_key
        # Warm-up with 5 inference calls
        print("  Warming up JAX (JIT) with 5 inference calls...")
        for _ in range(5):
            _ = policy_jit.infer(observation, noise=noise_np)
        
        # Test inference with 5 calls and average timing
        print("  Testing JAX (JIT) with 5 inference calls...")
        jit_times = []
        for i in range(5):
            t0 = time.perf_counter()
            jitted_result = policy_jit.infer(observation, noise=noise_np)
            t1 = time.perf_counter()
            jit_times.append(t1 - t0)
        
        jitted_time = np.mean(jit_times)
        print(f"JAX (JIT) individual times: {[f'{t*1000:.2f}ms' for t in jit_times]}")
        print(f"JAX (JIT) average time: {jitted_time*1000:.2f} ms")

        # No-JIT policy by bypassing jitted wrapper
        policy_nojit = _policy_config.create_trained_policy(config, checkpoint_dir)
        policy_nojit._rng = rng_key
        # Force no-JIT path by using raw method
        policy_nojit._sample_actions = policy_nojit._model.sample_actions
        
        # Run inference once (no warm-up needed for no-JIT)
        print("  Running JAX (no-JIT) inference...")
        t0 = time.perf_counter()
        nojit_result = policy_nojit.infer(observation, noise=noise_np)
        t1 = time.perf_counter()
        nojit_time = t1 - t0
        print(f"JAX (no-JIT) time: {nojit_time*1000:.2f} ms")

        # Compare outputs
        actions_jit = jitted_result["actions"]
        actions_nj = nojit_result["actions"]
        diff = np.abs(actions_jit - actions_nj)
        print("Actions comparison (JIT vs no-JIT):")
        print(f"  - Max abs diff: {diff.max():.6f}")
        print(f"  - Mean abs diff: {diff.mean():.6f}")
        # Relative differences (relative to no-JIT)
        rel_diff = np.abs((actions_jit - actions_nj) / actions_nj)
        print(f"  - Max relative diff: {np.max(rel_diff):.6f}")
        print(f"  - Mean relative diff: {np.mean(rel_diff):.6f}")
        if np.allclose(actions_jit, actions_nj, rtol=1e-5, atol=1e-6):
            print("  ✅ Match within strict tolerance")
        elif np.allclose(actions_jit, actions_nj, rtol=1e-4, atol=1e-5):
            print("  ⚠️  Match within moderate tolerance")
        elif np.allclose(actions_jit, actions_nj, rtol=2e-2, atol=2e-3):
            print("  ⚠️  Match within loose tolerance")
        else:
            print("  ❌ Significant difference")

        return jitted_result, nojit_result, noise_np, jitted_time, nojit_time

    except Exception as e:
        print(f"Failed JAX JIT vs no-JIT comparison: {e}")
        return None, None, None, None, None


def main():
    parser = argparse.ArgumentParser(description="Run inference with both JAX and PyTorch Pi0 models")
    parser.add_argument("--model_name", type=str, default="pi0_aloha_sim", 
                       choices=["pi0_aloha_sim", "pi0_aloha_towel", "pi0_base", "pi05_droid", "pi0_droid", "pi0_libero", "pi05_libero"],
                       help="Model name to use")
    parser.add_argument("--jax_checkpoint_dir", type=str, default=None,
                       help="Directory containing JAX model checkpoints")
    parser.add_argument("--pytorch_checkpoint_dir", type=str, default=None,
                       help="Directory containing PyTorch model checkpoints")
    args = parser.parse_args()

    """Run both inference examples and compare results."""
    print("Pi0 Model Inference Comparison")
    print("=" * 50)

    # Set random seed for reproducibility
    np.random.seed(42)

    observation = create_example_data(args.model_name)


    # Compare JAX JIT vs no-JIT (and get noise for PyTorch)
    jax_jitted_result, jax_nojit_result, noise, jitted_time, nojit_time = run_jax_inference_compare_jit(
        observation, args.model_name, args.jax_checkpoint_dir
    )

    import torch
    torch.cuda.empty_cache()
    # Run PyTorch inference with same noise as JAX
    pytorch_result, pytorch_time = run_pytorch_inference_example(observation, args.model_name, noise, args.pytorch_checkpoint_dir)

    # Compare JAX (JIT) with PyTorch
    if jax_jitted_result is not None and pytorch_result is not None:
        compare_results(jax_jitted_result, pytorch_result)

    # Compare JAX (no-JIT) with PyTorch
    if jax_nojit_result is not None and pytorch_result is not None:
        compare_results(jax_nojit_result, pytorch_result)

    # Print timing comparison
    if jitted_time is not None and nojit_time is not None and pytorch_time is not None:
        print("\n=== Timing Comparison ===")
        print(f"JAX (JIT):     {jitted_time*1000:.2f} ms")
        print(f"JAX (no-JIT):  {nojit_time*1000:.2f} ms")
        print(f"PyTorch:       {pytorch_time*1000:.2f} ms")
        
        # Calculate speedup ratios
        if jitted_time > 0:
            jit_speedup = nojit_time / jitted_time
            print(f"JIT speedup:   {jit_speedup:.2f}x")
        
        if pytorch_time > 0:
            jit_vs_pytorch = pytorch_time / jitted_time
            nojit_vs_pytorch = pytorch_time / nojit_time
            print(f"JAX JIT vs PyTorch: {jit_vs_pytorch:.2f}x")
            print(f"JAX no-JIT vs PyTorch: {nojit_vs_pytorch:.2f}x")

    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main()
