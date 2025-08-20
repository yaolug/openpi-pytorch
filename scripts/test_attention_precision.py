#!/usr/bin/env python3
"""
Simplified test script to compare attention mechanism precision between PyTorch and JAX.

This script focuses on the core attention computation without the full encoder stack.
"""

import logging
import numpy as np
import torch
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple


def setup_logging():
    """Setup logging for debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_fixed_inputs(batch_size: int = 2, seq_len: int = 196, hidden_size: int = 1152):
    """Create fixed inputs for deterministic comparison."""
    np.random.seed(42)
    
    # Create fixed input embeddings
    inputs_embeds = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32) * 0.1
    
    return inputs_embeds


def test_pytorch_attention(inputs_embeds: np.ndarray, num_heads: int = 16):
    """Test PyTorch multi-head attention."""
    print("\n=== Testing PyTorch Multi-Head Attention ===")
    
    # Create a simple multi-head attention layer
    hidden_size = inputs_embeds.shape[-1]
    head_dim = hidden_size // num_heads
    
    # Create attention layer
    attention = torch.nn.MultiheadAttention(
        embed_dim=hidden_size,
        num_heads=num_heads,
        dropout=0.0,
        batch_first=True
    )
    attention.eval()
    
    # Convert inputs to PyTorch tensors
    inputs_tensor = torch.from_numpy(inputs_embeds)
    
    print(f"Input shape: {inputs_tensor.shape}")
    print(f"Hidden size: {hidden_size}, Head dim: {head_dim}")
    
    # Forward pass
    with torch.no_grad():
        try:
            output, attention_weights = attention(
                query=inputs_tensor,
                key=inputs_tensor,
                value=inputs_tensor
            )
            
            print(f"Output shape: {output.shape}")
            print(f"Attention weights shape: {attention_weights.shape}")
            
            # Compute statistics
            output_stats = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'norm': torch.norm(output).item()
            }
            
            print(f"Output statistics:")
            for key, value in output_stats.items():
                print(f"  {key}: {value:.8f}")
            
            return output, attention_weights, output_stats
            
        except Exception as e:
            print(f"PyTorch attention failed: {e}")
            return None, None, {}


def test_jax_attention(inputs_embeds: np.ndarray, num_heads: int = 16):
    """Test JAX multi-head attention."""
    print("\n=== Testing JAX Multi-Head Attention ===")
    
    # Create a simple multi-head attention layer
    hidden_size = inputs_embeds.shape[-1]
    head_dim = hidden_size // num_heads
    
    # Create attention layer
    attention = jax.nn.MultiHeadDotProductAttention(
        num_heads=num_heads,
        kernel_init=jax.nn.initializers.xavier_uniform(),
        deterministic=True
    )
    
    # Convert inputs to JAX arrays
    inputs_jax = jnp.array(inputs_embeds)
    
    print(f"Input shape: {inputs_jax.shape}")
    print(f"Hidden size: {hidden_size}, Head dim: {head_dim}")
    
    # Initialize and apply
    rng = jax.random.key(42)
    variables = attention.init(rng, inputs_jax, inputs_jax)
    
    try:
        output = attention.apply(variables, inputs_jax, inputs_jax)
        
        print(f"Output shape: {output.shape}")
        
        # Compute statistics
        output_stats = {
            'mean': jnp.mean(output).item(),
            'std': jnp.std(output).item(),
            'min': jnp.min(output).item(),
            'max': jnp.max(output).item(),
            'norm': jnp.linalg.norm(output).item()
        }
        
        print(f"Output statistics:")
        for key, value in output_stats.items():
            print(f"  {key}: {value:.8f}")
        
        return output, output_stats
        
    except Exception as e:
        print(f"JAX attention failed: {e}")
        return None, {}


def compare_attention_outputs(pytorch_output: torch.Tensor, jax_output: jnp.ndarray, 
                            pytorch_stats: Dict[str, Any], jax_stats: Dict[str, Any]):
    """Compare attention outputs from both implementations."""
    print("\n" + "=" * 70)
    print("📊 ATTENTION OUTPUT COMPARISON")
    print("=" * 70)
    
    if pytorch_output is None or jax_output is None:
        print("❌ One or both attention implementations failed")
        return
    
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
        print("✅ Attention outputs are essentially identical (max diff < 1e-6)")
    elif max_element_diff < 1e-4:
        print("✅ Attention outputs are very close (max diff < 1e-4)")
    elif max_element_diff < 1e-2:
        print("⚠️  Attention outputs are reasonably close (max diff < 1e-2)")
    else:
        print("❌ Attention outputs differ significantly (max diff >= 1e-2)")


def main():
    """Main function to run the attention precision comparison tests."""
    setup_logging()
    
    print("🚀 Testing Attention Precision Comparison: PyTorch vs JAX")
    print("=" * 70)
    
    # Test parameters
    batch_size = 2
    seq_len = 196  # Typical for 224x224 images with 16x16 patches
    hidden_size = 1152
    num_heads = 16
    
    print(f"Test parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of heads: {num_heads}")
    
    # Create fixed inputs
    inputs_embeds = create_fixed_inputs(batch_size, seq_len, hidden_size)
    
    # Test PyTorch attention
    pytorch_output, pytorch_weights, pytorch_stats = test_pytorch_attention(inputs_embeds, num_heads)
    
    # Test JAX attention
    jax_output, jax_stats = test_jax_attention(inputs_embeds, num_heads)
    
    # Compare outputs
    compare_attention_outputs(pytorch_output, jax_output, pytorch_stats, jax_stats)
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print("✅ Attention precision comparison completed!")
    print("💡 Check the results above to assess numerical differences between implementations")


if __name__ == "__main__":
    main() 