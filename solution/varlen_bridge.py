"""Optional native PyTorch varlen attention; adapted from our c8/varlen branch."""
import torch

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
            return None
        names = [argument.name for argument in operator._schema.arguments]
        expected = ["query", "key", "value", "bias", "cu_seqlens_q", "cu_seqlens_k",
                    "max_seqlen_q", "max_seqlen_k", "dropout_p", "custom_mask_type",
                    "compute_log_sumexp"]
        if names[:len(expected)] != expected or names[len(expected)] != "scale":
            return None
        if not operator._schema.arguments[len(expected)].kwarg_only:
            return None
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::_efficient_attention_forward", "Autograd"):
            return None
        return operator
    except (AttributeError, IndexError, RuntimeError, TypeError):
        return None


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
    except Exception:
        return None
    if (output.shape != query.shape or output.dtype != query.dtype
            or (query.requires_grad or key.requires_grad or value.requires_grad)
            and not output.requires_grad):
        return None
    return output



def prepare_plan(mask, cos, sin):
    operator = _efficient_attention_operator()
    if operator is None or mask.dim() != 2:
        return None
    valid = mask.bool()
    if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        return None
    lengths = valid.sum(1, dtype=torch.int32)
    if bool((lengths == 0).any()):
        return None
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    maximum = int(lengths.max().item())
    batch, seq = valid.shape
    pc = cos.expand(batch, seq, cos.shape[-1])[valid].unsqueeze(0).contiguous()
    ps = sin.expand(batch, seq, sin.shape[-1])[valid].unsqueeze(0).contiguous()
    return operator, cumulative, maximum, pc, ps


def packed_attention(q, k, v, attention, plan):
    if (plan is None or attention.attention_dropout != 0 or attention.sliding_window is not None
            or any(x.dtype != torch.float16 for x in (q,k,v))):
        return None
    from transformers.models.qwen2 import modeling_qwen2
    operator, cumulative, maximum, cos, sin = plan
    dim = attention.head_dim
    heads, kv_heads = q.shape[-1] // dim, k.shape[-1] // dim
    if not kv_heads or heads % kv_heads:
        return None
    query = q.view(1,-1,heads,dim).transpose(1,2)
    key = k.view(1,-1,kv_heads,dim).transpose(1,2)
    query,key = modeling_qwen2.apply_rotary_pos_emb(query,key,cos,sin)
    query=query.transpose(1,2).contiguous()
    key=key.transpose(1,2).repeat_interleave(heads//kv_heads,dim=2).contiguous()
    value=v.view(1,-1,kv_heads,dim).repeat_interleave(heads//kv_heads,dim=2).contiguous()
    result=_call_varlen_attention(operator,query,key,value,cumulative,maximum,attention.scaling)
    return None if result is None else result.reshape(q.shape)
