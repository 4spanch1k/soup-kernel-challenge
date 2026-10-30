"""Fuse each Qwen2 MLP residual add with the norm at the next layer boundary."""

import torch
import torch.nn.functional as F

from .add_rmsnorm import MAX_WIDTH, add_rmsnorm
from .amp import autocast_dtype


def _fuse_add_norm(residual, delta, norm):
    y_dtype = autocast_dtype() or torch.promote_types(residual.dtype, norm.weight.dtype)
    return add_rmsnorm(residual, delta, norm.weight, norm.variance_epsilon, y_dtype)


def _causal_attention(attention, q, k, v):
    """Causal SDPA on the padded [batch, heads, seq, dim] layout with no mask tensor.

    The default path hands SDPA a 4D mask (the full square is computed) or, for a batch without
    padding, `enable_gqa` with no mask, which falls back to the slow math kernel on a T4. Expanding the
    KV heads and asking for `is_causal` runs the memory-efficient kernel that skips the upper triangle.
    Rows to the right of the valid tokens only feed padded rows, which are dropped afterwards."""
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    groups = q.shape[1] // k.shape[1]
    out = F.scaled_dot_product_attention(q, repeat_kv(k, groups), repeat_kv(v, groups),
                                         dropout_p=0.0, scale=attention.scaling, is_causal=True)
    return out.transpose(1, 2).contiguous()


def _dense_causal_attention(attention, hidden_states, position_embeddings, apply_rotary_pos_emb):
    """Qwen2Attention.forward for a batch without padding, with `_causal_attention`."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    q = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    out = _causal_attention(attention, q, k, v)
    return attention.o_proj(out.reshape(*input_shape, -1).contiguous())


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

        packed_idx = None
        no_padding = attention_mask is None
        right_padded = False
        if (torch.is_tensor(attention_mask) and attention_mask.dim() == 2
                and attention_mask.shape == inputs_embeds.shape[:2]
                and attention_mask.device == inputs_embeds.device):
            valid = attention_mask.bool()
            packed_idx = valid.reshape(-1).nonzero().squeeze(1)
            total_rows = inputs_embeds.shape[0] * inputs_embeds.shape[1]
            no_padding = packed_idx.numel() == total_rows
            right_padded = bool((valid[:, 1:] <= valid[:, :-1]).all())
            if packed_idx.numel() == 0 or no_padding:
                packed_idx = None

        # Plain causal attention is exact when padding only sits to the right of every row.
        causal_ok = ((no_padding or right_padded)
                     and self.config._attn_implementation == "sdpa"
                     and self.config.num_attention_heads % self.config.num_key_value_heads == 0
                     and all(layer.self_attn.attention_dropout == 0.0
                             and layer.self_attn.sliding_window is None
                             for layer in self.layers))

        if packed_idx is not None:
            batch, seq, width = inputs_embeds.shape
            rows = inputs_embeds.reshape(batch * seq, width).index_select(0, packed_idx)
            residual = rows
            layer_input = self.layers[0].input_layernorm(residual)
            cos, sin = position_embeddings

            for index, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                attention = layer.self_attn
                q = attention.q_proj(layer_input)
                k = attention.k_proj(layer_input)
                v = attention.v_proj(layer_input)
                q = q.new_zeros(batch * seq, q.shape[-1]).index_copy(0, packed_idx, q)
                k = k.new_zeros(batch * seq, k.shape[-1]).index_copy(0, packed_idx, k)
                v = v.new_zeros(batch * seq, v.shape[-1]).index_copy(0, packed_idx, v)
                q = q.view(batch, seq, -1, attention.head_dim).transpose(1, 2)
                k = k.view(batch, seq, -1, attention.head_dim).transpose(1, 2)
                v = v.view(batch, seq, -1, attention.head_dim).transpose(1, 2)
                q, k = modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)

                if causal_ok:
                    attn_out = _causal_attention(attention, q, k, v)
                else:
                    attention_interface = modeling_qwen2.ALL_ATTENTION_FUNCTIONS.get_interface(
                        self.config._attn_implementation, modeling_qwen2.eager_attention_forward)
                    attn_out, _ = attention_interface(
                        attention,
                        q,
                        k,
                        v,
                        causal_mask_mapping[self.config.layer_types[index]],
                        dropout=0.0 if not attention.training else attention.attention_dropout,
                        scaling=attention.scaling,
                        sliding_window=attention.sliding_window,
                    )
                attn_out = attn_out.reshape(batch * seq, -1).index_select(0, packed_idx)
                attn_out = attention.o_proj(attn_out)

                mlp_input, residual = _fuse_add_norm(
                    residual, attn_out, layer.post_attention_layernorm)
                mlp_out = layer.mlp(mlp_input)
                next_norm = (self.layers[index + 1].input_layernorm
                             if index + 1 < self.config.num_hidden_layers else self.norm)
                layer_input, residual = _fuse_add_norm(residual, mlp_out, next_norm)

            output = layer_input.new_zeros(batch * seq, width).index_copy(0, packed_idx, layer_input)
            return modeling_qwen2.BaseModelOutputWithPast(
                last_hidden_state=output.view(batch, seq, width), past_key_values=None)

        residual = inputs_embeds
        layer_input = self.layers[0].input_layernorm(residual)

        for index, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if causal_ok and no_padding:
                attn_out = _dense_causal_attention(
                    layer.self_attn, layer_input, position_embeddings, modeling_qwen2.apply_rotary_pos_emb)
            else:
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
