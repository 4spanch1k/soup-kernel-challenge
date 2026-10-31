"""Fuse each Qwen2 MLP residual add with the norm at the next layer boundary."""

import os

import torch

from .add_rmsnorm import MAX_WIDTH, add_rmsnorm
from .amp import autocast_dtype


def _call_with_checkpoint(function, *args, enabled):
    if not enabled:
        return function(*args)
    from torch.utils.checkpoint import checkpoint

    return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True)


def _fuse_add_norm(residual, delta, norm, exact_residual=False):
    y_dtype = autocast_dtype() or torch.promote_types(residual.dtype, norm.weight.dtype)
    return add_rmsnorm(residual, delta, norm.weight, norm.variance_epsilon, y_dtype,
                       exact_residual)


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
        original_kwargs = dict(kwargs)
        # PEFT's forward passes these three flags down as None; with their default values they change
        # nothing, so they must not push every call onto the original path.
        for name in ("output_attentions", "output_hidden_states"):
            if kwargs.get(name) is None or kwargs.get(name) is False:
                kwargs.pop(name, None)
        if self.config.use_return_dict and kwargs.get("return_dict") in (None, True):
            kwargs.pop("return_dict", None)

        def fallback():
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=supplied_inputs_embeds, use_cache=use_cache,
                            **original_kwargs)

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
        if (torch.is_tensor(attention_mask) and attention_mask.dim() == 2
                and attention_mask.shape == inputs_embeds.shape[:2]
                and attention_mask.device == inputs_embeds.device):
            packed_idx = attention_mask.bool().reshape(-1).nonzero().squeeze(1)
            total_rows = inputs_embeds.shape[0] * inputs_embeds.shape[1]
            if packed_idx.numel() == 0 or packed_idx.numel() == total_rows:
                packed_idx = None

        if packed_idx is not None:
            batch, seq, width = inputs_embeds.shape
            rows = inputs_embeds.reshape(batch * seq, width).index_select(0, packed_idx)
            residual = rows
            layer_input = self.layers[0].input_layernorm(residual)
            cos, sin = position_embeddings
            varlen_plan = None
            if os.environ.get("SOUP_X_VARLEN", "0") == "1":
                from .varlen_bridge import prepare_plan
                varlen_plan = prepare_plan(attention_mask, cos, sin)
            attention_checkpoint = os.environ.get("SOUP_X_ATTN_CKPT", "0")
            exact_residual = os.environ.get("SOUP_X_EXACT_RESID") == "1"

            for index, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                attention = layer.self_attn
                attention_interface = modeling_qwen2.ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation, modeling_qwen2.eager_attention_forward)

                def attention_output(hidden, cos, sin, mask, checkpoint_sdpa,
                                     attention=attention, attention_interface=attention_interface,
                                     packed_idx=packed_idx, batch=batch, seq=seq, varlen_plan=varlen_plan):
                    q = attention.q_proj(hidden)
                    k = attention.k_proj(hidden)
                    v = attention.v_proj(hidden)
                    if varlen_plan is not None:
                        from .varlen_bridge import packed_attention
                        compact = packed_attention(q, k, v, attention, varlen_plan)
                        if compact is not None:
                            return attention.o_proj(compact)
                    q = q.new_zeros(batch * seq, q.shape[-1]).index_copy(0, packed_idx, q)
                    k = k.new_zeros(batch * seq, k.shape[-1]).index_copy(0, packed_idx, k)
                    v = v.new_zeros(batch * seq, v.shape[-1]).index_copy(0, packed_idx, v)
                    q = q.view(batch, seq, -1, attention.head_dim).transpose(1, 2)
                    k = k.view(batch, seq, -1, attention.head_dim).transpose(1, 2)
                    v = v.view(batch, seq, -1, attention.head_dim).transpose(1, 2)
                    q, k = modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)

                    def run_attention(q, k, v, mask):
                        output, _ = attention_interface(
                            attention,
                            q,
                            k,
                            v,
                            mask,
                            dropout=0.0 if not attention.training else attention.attention_dropout,
                            scaling=attention.scaling,
                            sliding_window=attention.sliding_window,
                        )
                        return output

                    if checkpoint_sdpa:
                        attn_out = _call_with_checkpoint(
                            run_attention, q, k, v, mask, enabled=True)
                    else:
                        attn_out = run_attention(q, k, v, mask)
                    attn_out = attn_out.reshape(batch * seq, -1).index_select(0, packed_idx)
                    return attention.o_proj(attn_out)

                attention_mask_for_layer = causal_mask_mapping[self.config.layer_types[index]]
                if attention_checkpoint == "1":
                    attn_out = _call_with_checkpoint(
                        attention_output, layer_input, cos, sin, attention_mask_for_layer, False,
                        enabled=True)
                else:
                    attn_out = attention_output(
                        layer_input, cos, sin, attention_mask_for_layer,
                        attention_checkpoint == "2")

                mlp_input, residual = _fuse_add_norm(
                    residual, attn_out, layer.post_attention_layernorm)
                mlp_out = layer.mlp(mlp_input)
                next_norm = (self.layers[index + 1].input_layernorm
                             if index + 1 < self.config.num_hidden_layers else self.norm)
                layer_input, residual = _fuse_add_norm(
                    residual, mlp_out, next_norm, exact_residual)

            output = layer_input.new_zeros(batch * seq, width).index_copy(0, packed_idx, layer_input)
            return modeling_qwen2.BaseModelOutputWithPast(
                last_hidden_state=output.view(batch, seq, width), past_key_values=None)

        residual = inputs_embeds
        layer_input = self.layers[0].input_layernorm(residual)
        exact_residual = os.environ.get("SOUP_X_EXACT_RESID") == "1"
        dense_checkpoint = os.environ.get("SOUP_X_DENSE_CKPT") == "1"

        for index, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = causal_mask_mapping[self.config.layer_types[index]]

            def run_dense_attention(hidden, layer=layer, layer_mask=layer_mask):
                out, _ = layer.self_attn(
                    hidden_states=hidden,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                return out

            attn_out = _call_with_checkpoint(run_dense_attention, layer_input,
                                             enabled=dense_checkpoint)
            mlp_input, residual = _fuse_add_norm(residual, attn_out, layer.post_attention_layernorm)
            mlp_out = layer.mlp(mlp_input)
            next_norm = (self.layers[index + 1].input_layernorm
                         if index + 1 < self.config.num_hidden_layers else self.norm)
            layer_input, residual = _fuse_add_norm(
                residual, mlp_out, next_norm, exact_residual)

        return modeling_qwen2.BaseModelOutputWithPast(last_hidden_state=layer_input, past_key_values=None)

    forward._soup_layer_tail_patched = True
    forward._soup_layer_tail_original = original
    cls.forward = forward
