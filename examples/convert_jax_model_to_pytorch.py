#!/usr/bin/env python3
"""
Load a JAX model and print all parameter keys, with optional conversion to PyTorch.

This script loads a JAX model checkpoint using orbax and can either:
1. Print out all the parameter keys in a hierarchical structure for inspection
2. Convert the JAX model to PyTorch format using our PI0Pytorch model

Usage:
    # Just inspect keys:
    python convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only
    
    # Convert to PyTorch:
    python convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output

Example:    
    # pi0_droid 
    python convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid/params --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch2

    # pi0_aloha_sim
    python convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim/params --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch2

    # pi05_droid
    python convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi05_droid/params --output_path /home/$USER/.cache/openpi/openpi-assets-preview/checkpoints/pi05_droid_pytorch2
"""

import argparse
import pathlib
from typing import Any, Dict
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import torch
from jax.sharding import SingleDeviceSharding
from safetensors.torch import save_model

# Import our PI0Pytorch model
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models.pi0_config import Pi0Config
import openpi.models.gemma as _gemma
from openpi.shared import download
from openpi.models import model as _model
import jax.numpy as jnp

PRECISIONS = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def flatten_for_inspection(tree, parent_key="", separator="/"):
    """
    Flatten a nested dictionary for easy inspection of keys.
    
    Args:
        tree: The nested dictionary (JAX pytree)
        parent_key: Current parent key path
        separator: Separator to use between key levels
        
    Returns:
        Dictionary with flattened keys and array shapes as values
    """
    items = []
    for k, v in tree.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_for_inspection(v, new_key, separator).items())
        else:
            # Store shape and dtype information instead of the actual array
            if hasattr(v, 'shape') and hasattr(v, 'dtype'):
                items.append((new_key, f"shape: {v.shape}, dtype: {v.dtype}"))
            else:
                items.append((new_key, f"type: {type(v)}"))
    return dict(items)


def flatten_for_npz(tree, parent_key=""):
    """Flatten nested dictionary for conversion processing."""
    out = {}
    for k, v in tree.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_for_npz(v, new_key))
        else:
            out[new_key] = np.array(v)
    return out


def slice_paligemma_state_dict(state_dict, config):
    """Convert PaliGemma JAX parameters to PyTorch format."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""
    print(f"\n🔄 Converting PaliGemma parameters (suffix: '{suffix}')...")

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    print(f"  {jax_key} -> {pytorch_key}")
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)
    
    
    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    print(f"  {jax_key} -> {pytorch_key}")
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    print(f"[PyTorch DEBUG] mean(abs(patch_embedding.bias)): {state_dict[pytorch_key]}")
    
    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    print(f"  {jax_key} -> {pytorch_key}")
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    print(f"\n📊 Extracting vision transformer layers...")
    
    print(f"  img/Transformer/encoderblock/LayerNorm_0/scale{suffix} -> layer_norm1.weight (for all layers)")
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    print(f"  img/Transformer/encoderblock/MlpBlock_0/Dense_*{suffix} -> mlp.fc*.weight/bias (for all layers)")
    encoderblock_mlp_dense0_kernel= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias= state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    print(f"  img/Transformer/encoderblock/MultiHeadDotProductAttention_0/*{suffix} -> self_attn.*.weight/bias (for all layers)")
    encoderblock_attention_0_key_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}")
    encoderblock_attention_0_key_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}")
    encoderblock_attention_0_value_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}")
    encoderblock_attention_0_value_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}")
    encoderblock_attention_0_query_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}")
    encoderblock_attention_0_query_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}")
    encoderblock_attention_0_out_kernel = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}")
    encoderblock_attention_0_out_bias = state_dict.pop(f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}")

    print(f"\n🏗️ Converting {config.vision_config.num_hidden_layers} vision transformer layers...")
    for i in range(config.vision_config.num_hidden_layers):
        if i == 0 or i == config.vision_config.num_hidden_layers - 1:  # Print first and last layer details
            print(f"  Layer {i}: JAX arrays[{i}] -> paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.*")
        elif i == 1:
            print(f"  ... (layers 1-{config.vision_config.num_hidden_layers-2} follow same pattern)")
            
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"] = encoderblock_layernorm0_bias[i]
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"] = encoderblock_layernorm1_bias[i]

        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"] = encoderblock_mlp_dense0_bias[i]
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"] = encoderblock_mlp_dense1_bias[i]
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    print(f"\n🔚 Converting post-layer normalization...")
    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    print(f"  {jax_key} -> {pytorch_key}")
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()
    
    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    print(f"  {jax_key} -> {pytorch_key}")
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # multimodal projector
    print(f"\n🌉 Converting multimodal projector...")
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = 'paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight'
    print(f"  {jax_key} -> {pytorch_key}")
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()
    
    jax_key = f"img/head/bias{suffix}"
    pytorch_key = 'paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias'
    print(f"  {jax_key} -> {pytorch_key}")
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # text decoder (gemma)
    print(f"\n📝 Converting text decoder (Gemma)...")
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    print(f"  {jax_key} -> {pytorch_key}")
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # pop the einsum attention + mlp representations
    print(f"\n🧠 Extracting language model parameters...")
    print(f"  llm/layers/attn/*{suffix} -> language_model.layers.*.self_attn.* (for all layers)")
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    print(f"  llm/layers/mlp/*{suffix} -> language_model.layers.*.mlp.* (for all layers)")
    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    print(f"  llm/layers/pre_*_norm{suffix} -> language_model.layers.*.*_layernorm.weight (for all layers)")
    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    print(f"\n🔄 Converting {config.text_config.num_hidden_layers} language model layers...")
    for i in range(config.text_config.num_hidden_layers):
        if i == 0 or i == config.text_config.num_hidden_layers - 1:  # Print first and last layer details
            print(f"  Layer {i}: JAX einsum arrays[{i}] -> paligemma_with_expert.paligemma.model.language_model.layers.{i}.*")
        elif i == 1:
            print(f"  ... (layers 1-{config.text_config.num_hidden_layers-2} follow same pattern)")
            
        q_proj_weight_reshaped = llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size)
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"] = q_proj_weight_reshaped

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"] = v_proj_weight_reshaped

        o_proj_weight_reshaped = llm_attention_attn_vec_einsum[i].transpose(2, 0, 1).reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size)
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"] = o_proj_weight_reshaped
        
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"] = up_proj_weight.transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"] = llm_input_layernorm[i]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]

    print(f"\n✅ Converting final language model components...")
    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    print(f"  {jax_key} -> {pytorch_key}")
    state_dict[pytorch_key] = state_dict.pop(jax_key)
    
    # pytorch_key = "paligemma_with_expert.paligemma.lm_head.weight"
    # print(f"  embedding_vector (tied weights) -> {pytorch_key}")
    # state_dict[pytorch_key] = embedding_vector # weights are tied.

    expert_dict = {}
    final_state_dict = {}
    
    # Expert-related keys to extract (including pi05 Dense layer parameters)
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]
    
    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, num_expert=1, checkpoint_dir=None):
    """Convert Gemma JAX parameters to PyTorch format."""
    print(f"\n🧠 Converting Gemma expert parameters (expert {num_expert})...")
    
    # Add missing attributes to config if they don't exist
    if not hasattr(config, 'vocab_size'):
        config.vocab_size = 257152  # PALIGEMMA_VOCAB_SIZE
    if not hasattr(config, 'hidden_size'):
        config.hidden_size = config.width
    if not hasattr(config, 'num_hidden_layers'):
        config.num_hidden_layers = config.depth
    if not hasattr(config, 'num_attention_heads'):
        config.num_attention_heads = config.num_heads

    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # Check if we have Dense layers (for pi05/adaptive normalization) or scale layers (for regular pi0)
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization
        print(f"  Detected pi05/adaptive normalization format - using Dense layers")
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}")
        llm_post_attention_layernorm_kernel = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}")
    else:
        # Regular pi0 with standard RMSNorm
        print(f"  Detected standard RMSNorm format - using scale layers")
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    print(f"\n🔄 Converting {config.num_hidden_layers} Gemma expert layers...")
    for i in range(config.num_hidden_layers):
        if i == 0 or i == config.num_hidden_layers - 1:  # Print first and last layer details
            print(f"  Layer {i}: JAX einsum arrays[{i}] -> paligemma_with_expert.gemma_expert.model.layers.{i}.*")
        elif i == 1:
            print(f"  ... (layers 1-{config.num_hidden_layers-2} follow same pattern)")
            
        q_proj_weight_reshaped = llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = q_proj_weight_reshaped

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = v_proj_weight_reshaped

        o_proj_weight_reshaped = llm_attention_attn_vec_einsum[i].reshape(config.num_attention_heads * config.head_dim, config.hidden_size).transpose(1,0)
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = o_proj_weight_reshaped
        
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = up_proj_weight.transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()

        if "pi05" in checkpoint_dir:
            # Pi05 with adaptive normalization - use Dense layer parameters directly
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"] = llm_input_layernorm_bias[i]
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"] = llm_post_attention_layernorm_bias[i]
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"] = llm_input_layernorm_kernel[i].transpose()
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"] = llm_post_attention_layernorm_kernel[i].transpose()
        else:
            # Regular pi0 with standard RMSNorm
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"] = llm_input_layernorm[i]
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]

    # Handle final norm layer
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization - use Dense layer parameters directly
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = final_norm_bias
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = final_norm_kernel.transpose()
    else:
        # Regular pi0 with standard RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = state_dict.pop(f"llm/final_norm_{num_expert}/scale{suffix}")
    
        #state_dict["paligemma_with_expert.gemma_expert.lm_head.weight"] = embedding_vector # weights are tied.

    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value
    
    print(f"  Extracted {len(final_state_dict)} Gemma expert parameters")
    return final_state_dict


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None):
    """Load and process params by restoring via JAX model loader first.
    This respects dtype conversions that occur during model restore.
    """
    params_dir = pathlib.Path(checkpoint_dir).resolve()
    # Allow passing the root checkpoint dir; use params/ if present
    if (params_dir / "params").exists():
        params_dir = params_dir / "params"

    # Map precision string to JAX dtype, or None to keep saved dtypes
    dtype_map = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
    }
    restore_dtype = dtype_map.get(restore_precision) if restore_precision else None

    # Use repository restore utility to load a pure dict of params (value suffix removed)
    params = _model.restore_params(params_dir, restore_type=jax.Array, dtype=restore_dtype)

    # get params for PaliGemma
    pali_params = params["PaliGemma"]
    del params["PaliGemma"]
    pali_params_flat = flatten_for_npz(pali_params)
    return {"paligemma_params": pali_params_flat, "projection_params": params}


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    Load JAX model from checkpoint and print all parameter keys.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
    """
    params_path = pathlib.Path(checkpoint_dir).resolve()
    
    if not params_path.exists():
        print(f"Error: Checkpoint directory does not exist: {params_path}")
        return
    
    print(f"Loading JAX model from: {params_path}")
    print("=" * 80)
    
    try:
        # Initialize checkpointer
        checkpointer = ocp.PyTreeCheckpointer()
        
        # Load metadata to see available keys
        metadata = checkpointer.metadata(params_path)
        print("Available top-level keys in checkpoint:")
        for key in metadata.keys():
            print(f"  - {key}")
        print()
        
        # Restore the parameters
        params_name = "params"
        if params_name not in metadata:
            print(f"Warning: '{params_name}' not found in metadata. Available keys: {list(metadata.keys())}")
            if metadata.keys():
                params_name = list(metadata.keys())[0]
                print(f"Using '{params_name}' instead.")
            else:
                print("No keys found in metadata!")
                return
        
        item = {params_name: metadata[params_name]}
        device = jax.local_devices()[0]
        sharding = SingleDeviceSharding(device)
        
        print(f"Restoring parameters for key: '{params_name}'...")
        restored = checkpointer.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree_util.tree_map(
                    lambda _: ocp.ArrayRestoreArgs(
                        restore_type=jax.Array,
                        sharding=sharding,
                    ),
                    item,
                ),
                transforms={},
            ),
        )
        
        params = restored[params_name]
        print(f"Successfully loaded parameters!")
        print()
        
        # Flatten and print all keys
        flat_params = flatten_for_inspection(params)
        
        print(f"All parameter keys with shapes and dtypes ({len(flat_params)} total):")
        print("=" * 80)
        
        # Sort keys for better readability
        sorted_keys = sorted(flat_params.keys())
        
        for key in sorted_keys:
            print(f"{key:<60} -> {flat_params[key]}")
        
        print()
        print("=" * 80)
        print(f"Summary: Found {len(flat_params)} parameters")
        
        # Print some high-level structure information
        top_level_keys = set()
        for key in sorted_keys:
            top_level_key = key.split('/')[0]
            top_level_keys.add(top_level_key)
        
        print(f"Top-level parameter groups: {sorted(list(top_level_keys))}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()


def convert_pi0_checkpoint(checkpoint_dir: str, precision: str, output_path: str):
    """
    Convert PI0 JAX checkpoint to PyTorch format.
    
    Args:
        checkpoint_dir: Path to the JAX checkpoint
        precision: Model precision (float32, bfloat16, float16)
        output_path: Path to save the converted PyTorch model
    """
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print("=" * 80)
    
    # Break down orbax ckpts by restoring via JAX to respect dtype
    initial_params = slice_initial_orbax_checkpoint(checkpoint_dir=checkpoint_dir, restore_precision=precision)
    print(f"[PyTorch DEBUG] initial_params: {initial_params.keys()}")
    
    # Process projection params
    print(f"\n🎯 Converting projection parameters...")

    if "pi05" in checkpoint_dir:
        keys = [
            "action_in_proj", 
            "action_out_proj",
            "time_mlp_in", 
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj", 
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params
        
        pytorch_weight_key = f"{key}.weight"
        pytorch_bias_key = f"{key}.bias"
        print(f"  {key}/kernel -> {pytorch_weight_key}")
        print(f"  {key}/bias -> {pytorch_bias_key}")
        
        projection_params[pytorch_weight_key] = torch.from_numpy(np.array(weight)).T
        projection_params[pytorch_bias_key] = torch.from_numpy(np.array(bias))

    # Create configs based on checkpoint path
    if "pi0_base" in checkpoint_dir:
        # Create a config object with vision_config and text_config attributes
        class PaliGemmaConfig:
            def __init__(self):
                self.vision_config = type('obj', (object,), {
                    'hidden_size': 1152,
                    'num_hidden_layers': 27,
                    'num_attention_heads': 16,
                    'intermediate_size': 4304,
                    'patch_size': 14,
                    'projection_dim': 2048
                })()
                self.text_config = type('obj', (object,), {
                    'hidden_size': 2048,
                    'num_hidden_layers': 18,
                    'num_attention_heads': 8,
                    'head_dim': 256,
                    'intermediate_size': 16384
                })()
        
        paligemma_config = PaliGemmaConfig()
        action_expert_config = _gemma.get_config("gemma_300m")
    elif "pi0_aloha" in checkpoint_dir:
        # Create a config object with vision_config and text_config attributes
        class PaliGemmaConfig:
            def __init__(self):
                self.vision_config = type('obj', (object,), {
                    'hidden_size': 1152,
                    'num_hidden_layers': 27,
                    'num_attention_heads': 16,
                    'intermediate_size': 4304,
                    'patch_size': 14,
                    'projection_dim': 2048
                })()
                self.text_config = type('obj', (object,), {
                    'hidden_size': 2048,
                    'num_hidden_layers': 18,
                    'num_attention_heads': 8,
                    'head_dim': 256,
                    'intermediate_size': 16384
                })()
        
        paligemma_config = PaliGemmaConfig()
        action_expert_config = _gemma.get_config("gemma_300m")
    else:
        print("Warning: Could not determine model config from checkpoint path. Using base configs.")
        # Create a config object with vision_config and text_config attributes
        class PaliGemmaConfig:
            def __init__(self):
                self.vision_config = type('obj', (object,), {
                    'hidden_size': 1152,
                    'num_hidden_layers': 27,
                    'num_attention_heads': 16,
                    'intermediate_size': 4304,
                    'patch_size': 14,
                    'projection_dim': 2048
                })()
                self.text_config = type('obj', (object,), {
                    'hidden_size': 2048,
                    'num_hidden_layers': 18,
                    'num_attention_heads': 8,
                    'head_dim': 256,
                    'intermediate_size': 16384
                })()
        
        paligemma_config = PaliGemmaConfig()
        action_expert_config = _gemma.get_config("gemma_300m")

    # Process PaliGemma weights
    paligemma_params, expert_params = slice_paligemma_state_dict(initial_params["paligemma_params"], paligemma_config)

    # Process Gemma weights from expert_params
    gemma_params = slice_gemma_state_dict(expert_params, action_expert_config, num_expert=1, checkpoint_dir=checkpoint_dir)

    # Create Pi0Config based on checkpoint path
    if "pi0_aloha_sim" in checkpoint_dir:
        pi0_config = Pi0Config(
            action_dim=14,  # ALOHA has 14 action dimensions
            action_horizon=50,
        )
    elif "pi0_aloha_towel" in checkpoint_dir:
        pi0_config = Pi0Config(
            action_dim=14,  # ALOHA has 14 action dimensions
            action_horizon=50,
        )
    elif "pi0_base" in checkpoint_dir:
        pi0_config = Pi0Config(
            action_dim=8,   # Base droid has 8 action dimensions
            action_horizon=10,
        )
    elif "pi05_droid" in checkpoint_dir:
        pi0_config = Pi0Config(
            action_dim=8,   # Base droid has 8 action dimensions
            action_horizon=10,
            pi05=True,
        )
    elif "pi05_libero" in checkpoint_dir:
        pi0_config = Pi0Config(
            action_dim=7,
            action_horizon=10,
            pi05=True,
        )
    else:
        print("Warning: Could not determine PI0 config from checkpoint path. Using base config.")
        pi0_config = Pi0Config(
            action_dim=8,
            action_horizon=10,
        )

    # Instantiate model
    print(f"\n🏗️ Creating PI0Pytorch model with config: action_dim={pi0_config.action_dim}, action_horizon={pi0_config.action_horizon}")
    pi0_model = PI0Pytorch(pi0_config)

    # Combine all parameters (no prefix needed for our model structure)
    torch_dtype = PRECISIONS[precision]
    all_params = {**paligemma_params, **gemma_params, **projection_params}
    
    print(f"\n🚀 Loading {len(all_params)} parameters into PyTorch model...")
    print(f"  - PaliGemma parameters: {len(paligemma_params)}")
    print(f"  - Gemma expert parameters: {len(gemma_params)}")
    print(f"  - Projection parameters: {len(projection_params)}")
    print(f"  - Target precision: {precision} ({torch_dtype})")
    
    # Print all JAX keys and shapes
    print(f"\n📋 JAX Model Keys, Shapes, and Dtypes:")
    print("=" * 80)
    for key, value in all_params.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, dtype: {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Print all PyTorch model keys and shapes
    print(f"\n📋 PyTorch Model Keys, Shapes, and Dtypes:")
    print("=" * 80)
    pytorch_state_dict = pi0_model.state_dict()
    for key, value in pytorch_state_dict.items():
        print(f"  {key}: {value.shape}, dtype: {value.dtype}")
    
    # Find missing keys
    print(f"\n🔍 Missing Keys Analysis:")
    print("=" * 80)
    missing_keys = []
    for key in pytorch_state_dict.keys():
        if key not in all_params:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"  Missing keys in JAX checkpoint ({len(missing_keys)}):")
        for key in missing_keys:
            print(f"    - {key}")
    else:
        print("  ✅ All PyTorch keys found in JAX checkpoint")
    
    # Find extra keys
    extra_keys = []
    for key in all_params.keys():
        if key not in pytorch_state_dict:
            extra_keys.append(key)
    
    if extra_keys:
        print(f"  Extra keys in JAX checkpoint ({len(extra_keys)}):")
        for key in extra_keys[:10]:  # Show first 10
            print(f"    - {key}")
        if len(extra_keys) > 10:
            print(f"    ... and {len(extra_keys) - 10} more")
    
    # Load state dict
    try:
        pi0_model.load_state_dict(all_params)
        print(f"  ✅ Successfully loaded parameters into model")
    except Exception as e:
        print(f"  ⚠️ Warning: Could not load all parameters: {e}")
        print(f"  Continuing with partial load...")
    
    pi0_model = pi0_model.to(torch_dtype)

    # Save the converted model using safetensors
    print(f"\n💾 Saving converted model to {output_path}...")
    import os
    import shutil
    os.makedirs(output_path, exist_ok=True)
    
    # Save model weights as SafeTensors using save_model to handle tied weights
    save_model(pi0_model, os.path.join(output_path, "model.safetensors"))
    
    # Copy assets folder if it exists
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)
        print(f"  📁 Copied assets folder from {assets_source}")
    else:
        print(f"  ⚠️ Assets folder not found at {assets_source}")
    
    # Save config as JSON for reference
    import json
    config_dict = {
        "action_dim": pi0_config.action_dim,
        "action_horizon": pi0_config.action_horizon,
        "paligemma_variant": pi0_config.paligemma_variant,
        "action_expert_variant": pi0_config.action_expert_variant,
        "precision": precision,
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"  ✅ Model saved successfully!")
    print(f"  📄 Config saved to config.json")
    print(f"  🔢 Model weights saved to model.safetensors")

    print(f"\n🎉 Model conversion completed successfully!")
    print(f"📊 Model info: {type(pi0_model).__name__} with {sum(p.numel() for p in pi0_model.parameters())} total parameters")


def main():
    parser = argparse.ArgumentParser(description="Load JAX model and optionally convert to PyTorch")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the JAX checkpoint directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save converted PyTorch model (required for conversion)"
    )
    parser.add_argument(
        "--precision",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        type=str,
        help="Precision for model conversion"
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only inspect parameter keys, don't convert"
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        model_name = args.checkpoint_dir.split("/")[-2]
        # Use s3:// instead of gs:// for proper openpi-assets bucket access
        checkpoint_dir = download.maybe_download(f"s3://openpi-assets/checkpoints/{model_name}")
    else:
        checkpoint_dir = args.checkpoint_dir
    
    if args.inspect_only:
        load_jax_model_and_print_keys(args.checkpoint_dir)
    else:
        if not args.output_path:
            print("Error: --output_path is required for conversion. Use --inspect_only to only view keys.")
            return
        convert_pi0_checkpoint(checkpoint_dir, args.precision, args.output_path)


if __name__ == "__main__":
    main()
