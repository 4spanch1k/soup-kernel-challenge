"""Fuse each Qwen2 MLP residual add with the norm at the next layer boundary."""

import torch

from .add_rmsnorm import MAX_WIDTH, add_rmsnorm
from .amp import autocast_dtype


def _fuse_add_norm(residual, delta, norm):
    y_dtype = autocast_dtype() or torch.promote_types(residual.dtype, norm.weight.dtype)
    return add_rmsnorm(residual, delta, norm.weight, norm.variance_epsilon, y_dtype)


def patch_qwen2_layer_tail():
    """Patch Qwen2Model.forward for the supported training path; keep the original as fallback."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2Model
    if getattr(cls.forward, "_soup_layer_tail_patched", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        supplied_inputs_embeds = inputs_embeds

        def fallback():
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=supplied_inputs_embeds, use_cache=use_cache, **kwargs)

        if ((input_ids is None) == (inputs_embeds is None)
                or type(self) is not modeling_qwen2.Qwen2Model
                or not self.training or not torch.is_grad_enabled()
                or self.gradient_checkpointing or past_key_values is not None or kwargs
                or getattr(self.config, "output_hidden_states", False)
                or getattr(self.config, "output_attentions", False)
                or (self.config.use_cache if use_cache is None else use_cache)
                or not isinstance(attention_mask, (torch.Tensor, dict, type(None)))
                or not isinstance(position_ids, (torch.Tensor, type(None)))
                or not self.layers or len(self.layers) != self.config.num_hidden_layers
                or len(self.config.layer_types) != self.config.num_hidden_layers
                or self.config.hidden_size > MAX_WIDTH
                or type(self.norm) is not modeling_qwen2.Qwen2RMSNorm):
            return fallback()

        for layer in self.layers:
            if (type(layer) is not modeling_qwen2.Qwen2DecoderLayer
                    or type(layer.self_attn) is not modeling_qwen2.Qwen2Attention
                    or type(layer.mlp) is not modeling_qwen2.Qwen2MLP
                    or getattr(layer, "gradient_checkpointing", False)
                    or type(layer.input_layernorm) is not modeling_qwen2.Qwen2RMSNorm
                    or type(layer.post_attention_layernorm) is not modeling_qwen2.Qwen2RMSNorm
                    or not layer.input_layernorm.weight.is_cuda
                    or not layer.post_attention_layernorm.weight.is_cuda):
                return fallback()

        if not self.norm.weight.is_cuda or not self.embed_tokens.weight.is_cuda:
            return fallback()

        if inputs_embeds is None:
            if not torch.is_tensor(input_ids) or not input_ids.is_cuda:
                return fallback()
            inputs_embeds = self.embed_tokens(input_ids)
        if (not torch.is_tensor(inputs_embeds) or inputs_embeds.dim() != 3
                or not inputs_embeds.is_cuda or inputs_embeds.shape[-1] > MAX_WIDTH
                or inputs_embeds.dtype not in (torch.float16, torch.float32)):
            return fallback()

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": modeling_qwen2.create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    modeling_qwen2.create_sliding_window_causal_mask(**mask_kwargs)
                )
        else:
            causal_mask_mapping = attention_mask

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        residual = inputs_embeds
        layer_input = self.layers[0].input_layernorm(residual)

        for index, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            attn_out, _ = layer.self_attn(
                hidden_states=layer_input,
                attention_mask=causal_mask_mapping[self.config.layer_types[index]],
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            mlp_input, residual = _fuse_add_norm(residual, attn_out, layer.post_attention_layernorm)
            mlp_out = layer.mlp(mlp_input)
            next_norm = (self.layers[index + 1].input_layernorm
                         if index + 1 < self.config.num_hidden_layers else self.norm)
            layer_input, residual = _fuse_add_norm(residual, mlp_out, next_norm)

        return modeling_qwen2.BaseModelOutputWithPast(last_hidden_state=layer_input, past_key_values=None)

    forward._soup_layer_tail_patched = True
    forward._soup_layer_tail_original = original
    cls.forward = forward
