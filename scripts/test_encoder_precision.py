#!/usr/bin/env python3
"""
Test script to compare compute precision between PyTorch SiglipEncoder and JAX Encoder1DBlock.

This script creates identical inputs and configurations for both implementations
and compares their outputs to assess numerical precision differences.
"""

import logging
import numpy as np
import torch
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple

# Import the encoder implementations
from transformers.models.siglip.modeling_siglip import SiglipEncoder, SiglipConfig
from openpi.models.siglip import Encoder1DBlock, Encoder


def setup_logging():
    """Setup logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_fixed_inputs(batch_size: int = 2, seq_len: int = 196, hidden_size: int = 768):
    """Create fixed inputs for deterministic comparison."""
    np.random.seed(42)
    
    # Create fixed input embeddings
    inputs_embeds = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32) * 0.1
    
    # Create fixed attention mask (all tokens are valid)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)
    
    return inputs_embeds, attention_mask


def create_pytorch_encoder_config(hidden_size: int = 768, num_hidden_layers: int = 12, 
                                 num_attention_heads: int = 12, intermediate_size: int = 3072):
    """Create PyTorch SiglipConfig."""
    config = SiglipConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=14,
        num_channels=3,
        qkv_bias=True,
        use_absolute_position_embeddings=True,
        use_relative_position_embeddings=False,
        use_mean_pooling=False,
        cls_token=False,
        num_labels=1000,
        classifier_dropout=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
        _attn_implementation="eager"
    )
    return config


def test_pytorch_encoder(inputs_embeds: np.ndarray, attention_mask: np.ndarray, 
                        config: SiglipConfig) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Test PyTorch SiglipEncoder."""
    print("\n=== Testing PyTorch SiglipEncoder ===")
    
    # Create model
    model = SiglipEncoder(config)
    model.eval()
    
    # Convert inputs to PyTorch tensors
    inputs_embeds_tensor = torch.from_numpy(inputs_embeds)
    attention_mask_tensor = torch.from_numpy(attention_mask).float()  # Convert to float
    
    print(f"Input shape: {inputs_embeds_tensor.shape}")
    print(f"Attention mask shape: {attention_mask_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        try:
            outputs = model(
                inputs_embeds=inputs_embeds_tensor,
                attention_mask=attention_mask_tensor,
                output_attentions=False,
                output_hidden_states=False
            )
            
            # Get the output
            last_hidden_state = outputs.last_hidden_state
            
            print(f"Output shape: {last_hidden_state.shape}")
            print(f"Output dtype: {last_hidden_state.dtype}")
            
            # Compute statistics
            output_stats = {
                'mean': last_hidden_state.mean().item(),
                'std': last_hidden_state.std().item(),
                'min': last_hidden_state.min().item(),
                'max': last_hidden_state.max().item(),
                'norm': torch.norm(last_hidden_state).item()
            }
            
            print(f"Output statistics:")
            for key, value in output_stats.items():
                print(f"  {key}: {value:.8f}")
            
            return last_hidden_state, output_stats
            
        except Exception as e:
            print(f"PyTorch encoder failed: {e}")
            # Return dummy output for comparison
            dummy_output = torch.zeros_like(inputs_embeds_tensor)
            dummy_stats = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'norm': 0.0
            }
            return dummy_output, dummy_stats


def test_jax_encoder(inputs_embeds: np.ndarray, attention_mask: np.ndarray, 
                    depth: int = 12, num_heads: int = 12, mlp_dim: int = 3072) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Test JAX Encoder with Encoder1DBlock."""
    print("\n=== Testing JAX Encoder ===")
    
    # Create model
    model = Encoder(
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=0.0,
        scan=False,
        dtype_mm="float32"
    )
    
    # Convert inputs to JAX arrays
    inputs_embeds_jax = jnp.array(inputs_embeds)
    attention_mask_jax = jnp.array(attention_mask)
    
    print(f"Input shape: {inputs_embeds_jax.shape}")
    print(f"Attention mask shape: {attention_mask_jax.shape}")
    
    # Initialize model parameters
    rng = jax.random.key(42)
    variables = model.init(rng, inputs_embeds_jax, deterministic=True)
    
    # Forward pass
    last_hidden_state, _ = model.apply(variables, inputs_embeds_jax, deterministic=True)
    
    print(f"Output shape: {last_hidden_state.shape}")
    print(f"Output dtype: {last_hidden_state.dtype}")
    
    # Compute statistics
    output_stats = {
        'mean': jnp.mean(last_hidden_state).item(),
        'std': jnp.std(last_hidden_state).item(),
        'min': jnp.min(last_hidden_state).item(),
        'max': jnp.max(last_hidden_state).item(),
        'norm': jnp.linalg.norm(last_hidden_state).item()
    }
    
    print(f"Output statistics:")
    for key, value in output_stats.items():
        print(f"  {key}: {value:.8f}")
    
    return last_hidden_state, output_stats


def compare_outputs(pytorch_output: torch.Tensor, jax_output: jnp.ndarray, 
                   pytorch_stats: Dict[str, Any], jax_stats: Dict[str, Any]):
    """Compare outputs from both implementations."""
    print("\n" + "=" * 70)
    print("📊 OUTPUT COMPARISON")
    print("=" * 70)
    
    # Convert to numpy for comparison
    pytorch_np = pytorch_output.detach().cpu().numpy()
    jax_np = np.array(jax_output)
    
    print(f"PyTorch output shape: {pytorch_np.shape}")
    print(f"JAX output shape: {jax_np.shape}")
    
    # Check if shapes match
    if pytorch_np.shape != jax_np.shape:
        print("❌ Output shapes don't match!")
        return
    
    print("✅ Output shapes match")
    
    # Element-wise comparison
    element_diff = np.abs(pytorch_np - jax_np)
    max_element_diff = np.max(element_diff)
    mean_element_diff = np.mean(element_diff)
    
    print(f"Element-wise differences:")
    print(f"  Max difference: {max_element_diff:.8f}")
    print(f"  Mean difference: {mean_element_diff:.8f}")
    print(f"  First 10 differences: {element_diff.flatten()[:10]}")
    
    # Relative differences
    epsilon = 1e-12
    pytorch_safe = np.abs(pytorch_np) + epsilon
    jax_safe = np.abs(jax_np) + epsilon
    
    rel_diff_pytorch = (element_diff / pytorch_safe) * 100
    rel_diff_jax = (element_diff / jax_safe) * 100
    
    mean_rel_diff_pytorch = np.mean(rel_diff_pytorch)
    mean_rel_diff_jax = np.mean(rel_diff_jax)
    
    print(f"Relative differences:")
    print(f"  Mean relative diff (w.r.t. PyTorch): {mean_rel_diff_pytorch:.4f}%")
    print(f"  Mean relative diff (w.r.t. JAX): {mean_rel_diff_jax:.4f}%")
    
    # Statistics comparison
    print(f"\nStatistics comparison:")
    for key in pytorch_stats:
        pytorch_val = pytorch_stats[key]
        jax_val = jax_stats[key]
        abs_diff = abs(pytorch_val - jax_val)
        rel_diff = abs_diff / (abs(pytorch_val) + epsilon) * 100
        print(f"  {key}: PyTorch={pytorch_val:.8f}, JAX={jax_val:.8f}, diff={abs_diff:.8f} ({rel_diff:.4f}%)")
    
    # Determine if differences are significant
    if max_element_diff < 1e-6:
        print("✅ Outputs are essentially identical (max diff < 1e-6)")
    elif max_element_diff < 1e-4:
        print("✅ Outputs are very close (max diff < 1e-4)")
    elif max_element_diff < 1e-2:
        print("⚠️  Outputs are reasonably close (max diff < 1e-2)")
    else:
        print("❌ Outputs differ significantly (max diff >= 1e-2)")
    
    # Check for systematic differences
    if mean_rel_diff_pytorch < 0.1:
        print("✅ Relative differences are very small (< 0.1%)")
    elif mean_rel_diff_pytorch < 1.0:
        print("⚠️  Relative differences are small (< 1%)")
    else:
        print("❌ Relative differences are significant (>= 1%)")


def test_single_block_comparison():
    """Test comparison of a single encoder block."""
    print("\n" + "=" * 70)
    print("🔬 SINGLE BLOCK COMPARISON")
    print("=" * 70)
    
    # Create inputs
    batch_size, seq_len, hidden_size = 2, 196, 768
    inputs_embeds, attention_mask = create_fixed_inputs(batch_size, seq_len, hidden_size)
    
    # Test PyTorch single layer
    print("\n--- PyTorch Single Layer ---")
    config = create_pytorch_encoder_config(hidden_size, 1, 12, 3072)
    pytorch_model = SiglipEncoder(config)
    pytorch_model.eval()
    
    inputs_embeds_tensor = torch.from_numpy(inputs_embeds)
    attention_mask_tensor = torch.from_numpy(attention_mask)
    
    with torch.no_grad():
        pytorch_output = pytorch_model(
            inputs_embeds=inputs_embeds_tensor,
            attention_mask=attention_mask_tensor
        ).last_hidden_state
    
    # Test JAX single layer
    print("\n--- JAX Single Layer ---")
    jax_model = Encoder(
        depth=1,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.0,
        scan=False,
        dtype_mm="float32"
    )
    
    inputs_embeds_jax = jnp.array(inputs_embeds)
    rng = jax.random.key(42)
    variables = jax_model.init(rng, inputs_embeds_jax, deterministic=True)
    jax_output, _ = jax_model.apply(variables, inputs_embeds_jax, deterministic=True)
    
    # Compare
    pytorch_np = pytorch_output.detach().cpu().numpy()
    jax_np = np.array(jax_output)
    
    element_diff = np.abs(pytorch_np - jax_np)
    max_diff = np.max(element_diff)
    mean_diff = np.mean(element_diff)
    
    print(f"\nSingle block comparison:")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")
    print(f"  First 5 differences: {element_diff.flatten()[:5]}")


def main():
    """Main function to run the precision comparison tests."""
    setup_logging()
    
    print("🚀 Testing Encoder Precision Comparison: PyTorch vs JAX")
    print("=" * 70)
    
    # Test parameters
    batch_size = 2
    seq_len = 196  # Typical for 224x224 images with 16x16 patches
    hidden_size = 1152
    num_layers = 27
    num_heads = 16
    mlp_dim = 4304
    
    print(f"Test parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Number of heads: {num_heads}")
    print(f"  MLP dimension: {mlp_dim}")
    
    # Create fixed inputs
    inputs_embeds, attention_mask = create_fixed_inputs(batch_size, seq_len, hidden_size)
    
    # Create PyTorch config
    config = create_pytorch_encoder_config(hidden_size, num_layers, num_heads, mlp_dim)
    
    # Test PyTorch encoder
    pytorch_output, pytorch_stats = test_pytorch_encoder(inputs_embeds, attention_mask, config)
    
    # Test JAX encoder
    jax_output, jax_stats = test_jax_encoder(inputs_embeds, attention_mask, num_layers, num_heads, mlp_dim)
    
    # Compare outputs
    compare_outputs(pytorch_output, jax_output, pytorch_stats, jax_stats)
    
    # Test single block comparison
    test_single_block_comparison()
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print("✅ Precision comparison completed!")
    print("💡 Check the results above to assess numerical differences between implementations")


if __name__ == "__main__":
    main() 