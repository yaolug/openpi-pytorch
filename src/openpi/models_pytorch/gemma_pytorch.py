from pytest import Cache
import torch
from torch import nn
import torch.version
# from transformers import Gemma3ForConditionalGeneration
from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration
from transformers import PreTrainedModel


import openpi.models.gemma as _gemma
from openpi.models import gemma as gemma_jax

from transformers.models.auto import CONFIG_MAPPING


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(self, vlm_config, action_expert_config, use_adarms=[False, False]):
        super().__init__()

        # TODO simplify config, to get the config from the name and then override necessary fields
        # maybe only need to override the fields in action expert
        vlm_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": vlm_config.width,
                    "intermediate_size": vlm_config.mlp_dim,
                    "model_type": "gemma",
                    "num_attention_heads": vlm_config.num_heads,
                    "head_dim": vlm_config.head_dim,
                    "num_hidden_layers": vlm_config.depth,
                    "num_image_tokens": 256,
                    "num_key_value_heads": vlm_config.num_kv_heads,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                    "use_adarms": use_adarms[0],
                    "adarms_cond_dim": vlm_config.width if use_adarms[0] else None,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        action_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=action_expert_config.head_dim,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=action_expert_config.width,
                initializer_range=0.02,
                intermediate_size=action_expert_config.mlp_dim,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=action_expert_config.num_heads,
                num_hidden_layers=action_expert_config.depth,
                num_key_value_heads=action_expert_config.num_kv_heads,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
                use_adarms=use_adarms[1],
                adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
            )

        # TODO convert to pytorch config

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config)
        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()
        # self.set_requires_grad()

        # TODO: remove the following 3 lines
        self.paligemma.vision_tower.eval()
        self.paligemma.eval()
        self.gemma_expert.eval()

    # def set_requires_grad(self):
    #     if self.config.freeze_vision_encoder:
    #         self.paligemma.vision_tower.eval()
    #         for params in self.paligemma.vision_tower.parameters():
    #             params.requires_grad = False

    #     if self.config.train_expert_only:
    #         self.paligemma.eval()
    #         for params in self.paligemma.parameters():
    #             params.requires_grad = False

    # def train(self, mode: bool = True):
    #     super().train(mode)

    #     if self.config.freeze_vision_encoder:
    #         self.paligemma.vision_tower.eval()

    #     if self.config.train_expert_only:
    #         self.paligemma.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self = self.to(dtype=torch.bfloat16)
        # self.paligemma = self.paligemma.to(dtype=torch.bfloat16)
        # self.gemma_expert = self.gemma_expert.to(dtype=torch.bfloat16)
        # return

        params_to_change_dtype = [
            "language_model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "multi_modal",
            "language_model.embed_tokens",
        ]

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                if any(selector in name for selector in params_to_keep_float32):
                    param.data = param.data.to(dtype=torch.float32)
                else:
                    param.data = param.data.to(dtype=torch.bfloat16)
            else:
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        # Handle different transformers versions
        if hasattr(self.paligemma, "get_image_features"):
            return self.paligemma.get_image_features(image)
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if inputs_embeds[0] is not None:
            print(f"[DEBUG] PaliGemma forward - inputs_embeds[0] shape: {inputs_embeds[0].shape}")
            print(f"[DEBUG] PaliGemma forward - attention_mask shape: {attention_mask.shape if attention_mask is not None else 'None'}")
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            print(f"[DEBUG] PaliGemma forward - prefix_output shape: {prefix_output.shape}")
        else:
            prefix_output = None
            prefix_past_key_values = None

        if inputs_embeds[1] is not None:
            print(f"[DEBUG] Gemma expert forward - inputs_embeds[1] shape: {inputs_embeds[1].shape}")
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            print(f"[DEBUG] Gemma expert forward - suffix_output shape: {suffix_output.shape}")
        else:
            suffix_output = None

        return [prefix_output, suffix_output], prefix_past_key_values