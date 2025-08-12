import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        elif target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


class PI0Pytorch(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control

    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)

    Designed by Physical Intelligence. Ported from Jax to Pytorch.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        # paligemma_with_export_config = PaliGemmaWithExpertConfig(
        #     freeze_vision_encoder=self.config.freeze_vision_encoder,
        #     train_expert_only=self.config.train_expert_only,
        #     attention_implementation=self.config.attention_implementation,
        # )
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(paligemma_config, action_expert_config, use_adarms=[False, True] if self.pi05 else [False, False])

        # Projections are float32
        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

    #     self.set_requires_grad()

    # def set_requires_grad(self):
    #     for params in self.state_proj.parameters():
    #         params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # Debug: Print PyTorch SigLIP model config and weights
        print(f"[PyTorch DEBUG] SigLIP Model Info:")
        vision_model = self.paligemma_with_expert.paligemma.vision_tower
        print(f"  - Vision model type: {type(vision_model)}")
        print(f"  - Vision model config: {vision_model.config}")
        
        # Print some key weights
        print(f"  - Vision model weights:")
        for name, param in vision_model.named_parameters():
            if "embed" in name or "patch" in name or "conv" in name:
                print(f"    {name}: {param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")
                break  # Just print first few
        
        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks):
            print(f"[PyTorch DEBUG] abs mean of image: {torch.mean(torch.abs(img))}")
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # # TODO: why we need to do this?
            # img_emb_dim = img_emb.shape[-1]
            # img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # embs = torch.cat(embs, dim=1)
        # return embs, pad_masks, att_masks

        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)

        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]

        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # Debug: embed_prefix outputs
        print(f"[PyTorch DEBUG] embed_prefix outputs:")
        print(f"  - embs shape: {embs.shape}")
        print(f"  - pad_masks shape: {pad_masks.shape}")
        print(f"  - att_masks shape: {att_masks.shape}")
        print(f"  - embs stats: min={embs.min():.6f}, max={embs.max():.6f}, mean={embs.mean():.6f}")

        # Print mean of embeddings along sequence length dimension (dim=1)
        # print(f"[PyTorch DEBUG] Mean embeddings across sequence length:")
        # torch.set_printoptions(threshold=float('inf'))
        # print(f"  {embs.mean(dim=1)[0, :]}")  # First 5 elements of first batch
        # torch.set_printoptions(threshold=10)
        # Debug: Print first 5 elements of first batch's embeddings
        print(f"[PyTorch DEBUG] First 5 elements of first batch's embeddings:")
        print(f"  {embs[0, 0:5, 0:5]}")
        print(f"  {embs[0, 769:775, 0:5]}")

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            # Embed state
            # self.state_proj = self.state_proj.to(dtype=torch.bfloat16).to(dtype=torch.float32)

            state_emb = self.state_proj(state)
            state_emb = state_emb.to(dtype=torch.bfloat16)
            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            dtype = state_emb.dtype
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        # self.action_in_proj = self.action_in_proj.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        action_emb = self.action_in_proj(noisy_actions)

        if not self.pi05:
            # self.action_time_mlp_in = self.action_time_mlp_in.to(dtype=torch.bfloat16).to(dtype=torch.float32)
            # self.action_time_mlp_out = self.action_time_mlp_out.to(dtype=torch.bfloat16).to(dtype=torch.float32)

            time_emb = time_emb[:, None, :].expand_as(action_emb)

            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)  # swish == silu
            action_time_emb = self.action_time_mlp_out(action_time_emb)

            # Convert to bfloat16 to match state embeddings
            action_time_emb = action_time_emb.to(dtype=torch.bfloat16)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)  # swish == silu
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Add head dimension to attention mask: [B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
        att_2d_masks_4d = att_2d_masks[:, None, :, :]

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond]
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)

        # self.action_out_proj = self.action_out_proj.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    @torch.no_grad()
    def sample_actions(self, observation, noise=None, num_steps=10) -> Tensor:
        #num_steps = 1

        # for key in observation:
        #     if isinstance(observation[key], torch.Tensor) and observation[key].dtype == torch.float32:
        #         observation[key] = observation[key].to(dtype=torch.bfloat16)

        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation['state'].shape[0]
        device = next(self.paligemma_with_expert.parameters()).device
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # TODO preprocess_observation
        # Observation(
        #     images=out_images,
        #     image_masks=out_masks,
        #     state=observation.state,
        #     tokenized_prompt=observation.tokenized_prompt,
        #     tokenized_prompt_mask=observation.tokenized_prompt_mask,
        #     token_ar_mask=observation.token_ar_mask,
        #     token_loss_mask=observation.token_loss_mask,
        # )

        for key in observation['image']:
            import numpy as np
            if observation["image"][key].dtype == np.uint8:
                observation["image"][key] = observation["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0

        # TODO: Move this after input_transform
        images = []
        for img in observation['image'].values():
            img_tensor = torch.from_numpy(np.array(img))
            
            # Handle different input formats
            if img_tensor.dim() == 4:  # (batch, H, W, C) -> (batch, C, H, W)
                img_tensor = img_tensor.permute(0, 3, 1, 2)
            elif img_tensor.dim() == 3:  # (H, W, C) -> (C, H, W) -> (1, C, H, W)
                img_tensor = img_tensor.permute(2, 0, 1)  # -> (C, H, W)
                img_tensor = img_tensor.unsqueeze(0)  # -> (1, C, H, W)
            
            # Ensure correct device and dtype
            img_tensor = img_tensor.to(device=next(self.paligemma_with_expert.parameters()).device, 
                                     dtype=torch.float32)
            images.append(img_tensor)
        
        img_masks = []
        for mask in observation['image_mask'].values():
            mask_tensor = torch.from_numpy(np.array(mask))
            # Ensure mask has batch dimension
            if mask_tensor.dim() == 0:
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.to(device=next(self.paligemma_with_expert.parameters()).device)
            img_masks.append(mask_tensor)
        
        lang_tokens = torch.from_numpy(np.array(observation['tokenized_prompt']))
        lang_masks = torch.from_numpy(np.array(observation['tokenized_prompt_mask']))
        state = torch.from_numpy(np.array(observation['state']))
        
        # Move language tokens and state to correct device
        lang_tokens = lang_tokens.to(device=next(self.paligemma_with_expert.parameters()).device)
        lang_masks = lang_masks.to(device=next(self.paligemma_with_expert.parameters()).device)
        state = state.to(device=next(self.paligemma_with_expert.parameters()).device)

        #print(f"[PyTorch DEBUG] images: {images}")
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        #return prefix_embs.to(torch.float32)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        # Add head dimension to attention mask: [B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
        prefix_att_2d_masks_4d = prefix_att_2d_masks[:, None, :, :]
        prefix_att_2d_masks_4d = torch.where(prefix_att_2d_masks_4d, 0.0, -2.3819763e38)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        output, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        print(f"[PyTorch DEBUG] output shape: {output[0].shape}")
        print(f"[PyTorch DEBUG] output stats: min={output[0].min():.6f}, max={output[0].max():.6f}, mean={output[0].mean():.6f}")
        # Print mean of output along sequence length dimension
        # if output[0] is not None:
        #     seq_mean = torch.mean(output[0], dim=1)  # Average across sequence length dimension
        #     print(f"[PyTorch DEBUG] Mean across sequence length (first batch):")
        #     torch.set_printoptions(threshold=float('inf'))
        #     print(f"  {seq_mean[0, :]}")  # Print first 5 elements of first batch
        #     torch.set_printoptions(threshold=30)

        dt = -1.0 / num_steps
        model_device = next(self.paligemma_with_expert.parameters()).device
        dt = torch.tensor(dt, dtype=torch.float32, device=model_device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=model_device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        print(f"[PyTorch DEBUG] suffix_embs shape: {suffix_embs.shape}")
        print(f"[PyTorch DEBUG] suffix_embs dtype: {suffix_embs.dtype}")
        print(f"[PyTorch DEBUG] suffix_embs stats: min={suffix_embs.min():.6f}, max={suffix_embs.max():.6f}, mean={suffix_embs.mean():.6f}")

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        print(f"[PyTorch DEBUG] full_att_2d_masks shape: {full_att_2d_masks.shape}")
        print(f"[PyTorch DEBUG] past_key_values[0][0].shape: {past_key_values[0][0].shape}")
        print(f"[PyTorch DEBUG] suffix_len: {suffix_len}")
        print(f"[PyTorch DEBUG] prefix_len: {prefix_len}")
        print(f"[PyTorch DEBUG] batch_size: {batch_size}")

        # # When using past_key_values, we need to account for the full sequence length
        # # The transformer expects attention mask to cover: past_key_values + new suffix tokens
        # # Adjust the attention mask to match the expected sequence length
        # if past_key_values is not None:
        #     # Get the actual key length from past_key_values
        #     # past_key_values is [prefix_past_key_values, suffix_past_key_values]
        #     # We want the prefix past_key_values: past_key_values[0]
        #     # Then access layer 0, keys tensor: past_key_values[0][0][0]
        #     # Format: [prefix/suffix][layer][key/value][batch, heads, seq_len, head_dim]
        #     if past_key_values[0] is not None and len(past_key_values[0]) > 0:
        #         past_seq_len = past_key_values[0][0][0].shape[2]
        #     else:
        #         # Fallback: if prefix is None, try suffix
        #         past_seq_len = past_key_values[1][0][0].shape[2] if past_key_values[1] is not None else 0
        #     current_seq_len = full_att_2d_masks.shape[2]
        #     expected_seq_len = past_seq_len + suffix_len

        #     if current_seq_len != expected_seq_len:
        #         # Pad the attention mask to match expected length
        #         pad_size = expected_seq_len - current_seq_len
        #         padding = torch.ones(
        #             batch_size, suffix_len, pad_size, dtype=full_att_2d_masks.dtype, device=full_att_2d_masks.device
        #         )
        #         full_att_2d_masks = torch.cat([full_att_2d_masks, padding], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Add head dimension to attention mask: [B, seq_len, seq_len] -> [B, 1, seq_len, seq_len]
        full_att_2d_masks_4d = full_att_2d_masks[:, None, :, :]
        full_att_2d_masks_4d = torch.where(full_att_2d_masks_4d, 0.0, -2.3819763e38)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond]
        )

        print(f"[PyTorch DEBUG] outputs_embeds shape: {outputs_embeds[1].shape}")
        print(f"[PyTorch DEBUG] outputs_embeds stats: min={outputs_embeds[1].min():.6f}, max={outputs_embeds[1].max():.6f}, mean={outputs_embeds[1].mean():.6f}")
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        
        # Debug: diffusion model output
        print(f"[PyTorch DEBUG] diffusion model output:")
        print(f"  - suffix_out shape: {suffix_out.shape}")
        print(f"  - suffix_out stats: min={suffix_out.min():.6f}, max={suffix_out.max():.6f}, mean={suffix_out.mean():.6f}")
        print(f"  - v_t (action output) shape: {v_t.shape}")
        print(f"  - v_t stats: min={v_t.min():.6f}, max={v_t.max():.6f}, mean={v_t.mean():.6f}")
        
        return v_t
