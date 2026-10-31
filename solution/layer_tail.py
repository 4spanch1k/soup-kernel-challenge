"""Fuse each Qwen2 MLP residual add with the norm at the next layer boundary."""

import os

import torch

from .add_rmsnorm import MAX_WIDTH, add_rmsnorm
from .amp import autocast_dtype


def _varlen_why(detail):
    """SOUP_X_WARM_LOG=1: say why the single-call attention path is not used."""
    if os.environ.get("SOUP_X_WARM_LOG"):
        print(f"[varlen] off: {detail}"[:400], flush=True)


_LAYER_TAIL_FALLBACK_COUNTS = {}


def _note_layer_tail_fallback(reason):
    count = _LAYER_TAIL_FALLBACK_COUNTS.get(reason, 0) + 1
    _LAYER_TAIL_FALLBACK_COUNTS[reason] = count
    if count == 1 and os.environ.get("SOUP_X_WARM_LOG"):
        print(f"[layer_tail] fallback: {reason} (count={count})", flush=True)


def _fuse_add_norm(residual, delta, norm):
    y_dtype = autocast_dtype() or torch.promote_types(residual.dtype, norm.weight.dtype)
    return add_rmsnorm(residual, delta, norm.weight, norm.variance_epsilon, y_dtype)


def _efficient_attention_operator():
    try:
        operator = torch.ops.aten._efficient_attention_forward.default
        schema = str(operator._schema)
        expected_schema = (
            "aten::_efficient_attention_forward(Tensor query, Tensor key, Tensor value, "
            "Tensor? bias, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, "
            "SymInt? max_seqlen_q, SymInt? max_seqlen_k, float dropout_p, "
            "int custom_mask_type, bool compute_log_sumexp=False, *, float? scale=None"
        )
        if not schema.startswith(expected_schema):
            _varlen_why(f"schema differs: {schema}")
            return None
        names = [argument.name for argument in operator._schema.arguments]
        expected = ["query", "key", "value", "bias", "cu_seqlens_q", "cu_seqlens_k",
                    "max_seqlen_q", "max_seqlen_k", "dropout_p", "custom_mask_type",
                    "compute_log_sumexp"]
        if names[:len(expected)] != expected or names[len(expected)] != "scale":
            _varlen_why(f"argument names differ: {names}")
            return None
        if not operator._schema.arguments[len(expected)].kwarg_only:
            _varlen_why("scale is not keyword-only")
            return None
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::_efficient_attention_forward", "Autograd"):
            _varlen_why("no Autograd kernel for the operator")
            return None
        return operator
    except (AttributeError, IndexError, RuntimeError, TypeError) as exc:
        _varlen_why(f"operator lookup raised {exc!r}")
        return None


_VARLEN_SEEN = {"ok": 0, "fallback": 0}


def _varlen_note(kind, detail=""):
    """SOUP_X_WARM_LOG=1: count the single-call attention path and say why it stopped."""
    _VARLEN_SEEN[kind] += 1
    if os.environ.get("SOUP_X_WARM_LOG") and (kind == "fallback" or _VARLEN_SEEN["ok"] in (1, 28, 1000)):
        print(f"[varlen] {kind} ok={_VARLEN_SEEN['ok']} fallback={_VARLEN_SEEN['fallback']} {detail}",
              flush=True)


def _call_varlen_attention(operator, query, key, value, cu_seqlens, max_seqlen, scale):
    try:
        output = operator(
            query,
            key,
            value,
            None,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            0.0,
            1,
            True,
            scale=scale,
        )[0]
    except Exception as exc:
        _varlen_note("fallback", f"operator raised {exc!r}"[:300])
        return None
    if (output.shape != query.shape or output.dtype != query.dtype
            or (query.requires_grad or key.requires_grad or value.requires_grad)
            and not output.requires_grad):
        _varlen_note("fallback", "output shape/dtype/grad mismatch")
        return None
    _varlen_note("ok")
    return output


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
        original_kwargs = kwargs.copy()

        for name in ("output_attentions", "output_hidden_states"):
            if kwargs.get(name) is None or kwargs.get(name) is False:
                kwargs.pop(name, None)
        return_dict = kwargs.get("return_dict")
        if self.config.use_return_dict and (return_dict is None or return_dict is True):
            kwargs.pop("return_dict", None)

        def fallback(reason):
            _note_layer_tail_fallback(reason)
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=supplied_inputs_embeds, use_cache=use_cache,
                            **original_kwargs)

        if (input_ids is None) == (inputs_embeds is None):
            return fallback("input_ids/inputs_embeds mode")
        if type(self) is not modeling_qwen2.Qwen2Model:
            return fallback("Qwen2Model subclass")
        if not self.training:
            return fallback("model is not training")
        if not torch.is_grad_enabled():
            return fallback("gradients are disabled")
        if self.gradient_checkpointing:
            return fallback("model gradient checkpointing")
        if past_key_values is not None:
            return fallback("past_key_values is present")
        if kwargs:
            return fallback(f"unsupported kwargs: {', '.join(sorted(kwargs))}")
        if getattr(self.config, "output_hidden_states", False):
            return fallback("config.output_hidden_states")
        if getattr(self.config, "output_attentions", False):
            return fallback("config.output_attentions")
        cache_enabled = self.config.use_cache if use_cache is None else use_cache
        if cache_enabled:
            return fallback("use_cache is enabled")
        if not isinstance(attention_mask, (torch.Tensor, dict, type(None))):
            return fallback("unsupported attention_mask type")
        if not isinstance(position_ids, (torch.Tensor, type(None))):
            return fallback("unsupported position_ids type")
        if not self.layers or len(self.layers) != self.config.num_hidden_layers:
            return fallback("Qwen2 layer count")
        if len(self.config.layer_types) != self.config.num_hidden_layers:
            return fallback("layer_types count")
        if self.config.hidden_size > MAX_WIDTH:
            return fallback("hidden_size exceeds kernel width")
        if type(self.norm) is not modeling_qwen2.Qwen2RMSNorm:
            return fallback("final norm type")

        for index, layer in enumerate(self.layers):
            if type(layer) is not modeling_qwen2.Qwen2DecoderLayer:
                return fallback(f"layer {index} type")
            if type(layer.self_attn) is not modeling_qwen2.Qwen2Attention:
                return fallback(f"layer {index} attention type")
            if type(layer.mlp) is not modeling_qwen2.Qwen2MLP:
                return fallback(f"layer {index} MLP type")
            if getattr(layer, "gradient_checkpointing", False):
                return fallback(f"layer {index} gradient checkpointing")
            if type(layer.input_layernorm) is not modeling_qwen2.Qwen2RMSNorm:
                return fallback(f"layer {index} input norm type")
            if type(layer.post_attention_layernorm) is not modeling_qwen2.Qwen2RMSNorm:
                return fallback(f"layer {index} post-attention norm type")
            if not layer.input_layernorm.weight.is_cuda:
                return fallback(f"layer {index} input norm is not CUDA")
            if not layer.post_attention_layernorm.weight.is_cuda:
                return fallback(f"layer {index} post-attention norm is not CUDA")

        if not self.norm.weight.is_cuda or not self.embed_tokens.weight.is_cuda:
            return fallback("final norm or embedding is not CUDA")

        if inputs_embeds is None:
            if not torch.is_tensor(input_ids) or not input_ids.is_cuda:
                return fallback("input_ids is not a CUDA tensor")
            inputs_embeds = self.embed_tokens(input_ids)
        if (not torch.is_tensor(inputs_embeds) or inputs_embeds.dim() != 3
                or not inputs_embeds.is_cuda or inputs_embeds.shape[-1] > MAX_WIDTH
                or inputs_embeds.dtype not in (torch.float16, torch.float32)):
            return fallback("inputs_embeds shape, device, width, or dtype")

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

        cos, sin = position_embeddings
        packed_idx = None
        valid_mask = None
        if (torch.is_tensor(attention_mask) and attention_mask.dim() == 2
                and attention_mask.shape == inputs_embeds.shape[:2]
                and attention_mask.device == inputs_embeds.device):
            valid_mask = attention_mask.bool()
            packed_idx = valid_mask.reshape(-1).nonzero().squeeze(1)
            total_rows = inputs_embeds.shape[0] * inputs_embeds.shape[1]
            if packed_idx.numel() == 0 or packed_idx.numel() == total_rows:
                packed_idx = None

        if packed_idx is not None:
            batch, seq, width = inputs_embeds.shape
            varlen_operator = None
            varlen_active = os.environ.get("SOUP_X_VARLEN") == "1"
            if varlen_active:
                right_padded = bool(
                    (valid_mask[:, 1:] <= valid_mask[:, :-1]).all() & valid_mask[:, 0].all()
                )
                varlen_active = (
                    right_padded
                    and inputs_embeds.is_cuda
                    and not self.has_sliding_layers
                    and all(layer.self_attn.sliding_window is None
                            and layer.self_attn.attention_dropout == 0
                            for layer in self.layers)
                    and cos.dim() == 3
                    and cos.shape[1] == seq
                    and cos.shape[0] in (1, batch)
                    and cos.shape == sin.shape
                    and getattr(self.config, "num_key_value_heads", 0) > 0
                    and getattr(self.config, "num_attention_heads", 0)
                    % getattr(self.config, "num_key_value_heads", 1) == 0
                )
                if not varlen_active:
                    _varlen_why("batch/config conditions not met (right padding, sliding window, dropout, cos shape)")
                if varlen_active:
                    varlen_operator = _efficient_attention_operator()
                    varlen_active = varlen_operator is not None
                if varlen_active:
                    lengths = valid_mask.sum(dim=1, dtype=torch.int32)
                    cu_seqlens = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
                    max_seqlen = int(lengths.max().item())
                    position_cos = cos.expand(batch, -1, -1).reshape(-1, cos.shape[-1])
                    position_sin = sin.expand(batch, -1, -1).reshape(-1, sin.shape[-1])
                    position_cos = position_cos.index_select(0, packed_idx).unsqueeze(0)
                    position_sin = position_sin.index_select(0, packed_idx).unsqueeze(0)
                    repeat_kv = self.config.num_attention_heads // self.config.num_key_value_heads

            rows = inputs_embeds.reshape(batch * seq, width).index_select(0, packed_idx)
            residual = rows
            layer_input = self.layers[0].input_layernorm(residual)
            for index, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                attention = layer.self_attn
                q_rows = attention.q_proj(layer_input)
                k_rows = attention.k_proj(layer_input)
                v_rows = attention.v_proj(layer_input)
                attn_out = None

                if varlen_active:
                    q_heads = getattr(self.config, "num_attention_heads", 0)
                    kv_heads = getattr(self.config, "num_key_value_heads", 0)
                    if (q_rows.dtype != torch.float16 or k_rows.dtype != torch.float16
                            or v_rows.dtype != torch.float16 or not kv_heads
                            or q_rows.shape[-1] != q_heads * attention.head_dim
                            or k_rows.shape[-1] != kv_heads * attention.head_dim
                            or v_rows.shape[-1] != kv_heads * attention.head_dim):
                        varlen_active = False
                        _varlen_why("dtype/shape of q,k,v projections not as expected")
                    else:
                        try:
                            query = q_rows.view(-1, q_heads, attention.head_dim).unsqueeze(0)
                            query = query.transpose(1, 2)
                            key = k_rows.view(-1, kv_heads, attention.head_dim).unsqueeze(0)
                            key = key.transpose(1, 2)
                            value = v_rows.view(1, -1, kv_heads, attention.head_dim)
                            query, key = modeling_qwen2.apply_rotary_pos_emb(
                                query, key, position_cos, position_sin)
                            query = query.transpose(1, 2).contiguous()
                            key = key.transpose(1, 2).contiguous()
                            if query.dtype == torch.float16 and key.dtype == torch.float16:
                                key = key.repeat_interleave(repeat_kv, dim=2)
                                value = value.repeat_interleave(repeat_kv, dim=2)
                                attn_out = _call_varlen_attention(
                                    varlen_operator, query, key, value, cu_seqlens,
                                    max_seqlen, attention.scaling)
                        except Exception as exc:
                            _varlen_note("fallback", f"prepare raised {exc!r}"[:300])
                            attn_out = None
                        if attn_out is None:
                            varlen_active = False
                        else:
                            attn_out = attn_out.reshape(-1, width)

                if attn_out is None:
                    q = q_rows.new_zeros(batch * seq, q_rows.shape[-1]).index_copy(
                        0, packed_idx, q_rows)
                    k = k_rows.new_zeros(batch * seq, k_rows.shape[-1]).index_copy(
                        0, packed_idx, k_rows)
                    v = v_rows.new_zeros(batch * seq, v_rows.shape[-1]).index_copy(
                        0, packed_idx, v_rows)
                    q = q.view(batch, seq, -1, attention.head_dim).transpose(1, 2)
                    k = k.view(batch, seq, -1, attention.head_dim).transpose(1, 2)
                    v = v.view(batch, seq, -1, attention.head_dim).transpose(1, 2)
                    q, k = modeling_qwen2.apply_rotary_pos_emb(q, k, cos, sin)

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
