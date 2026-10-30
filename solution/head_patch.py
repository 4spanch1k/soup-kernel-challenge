"""Plugs the chunked head + loss (ce_loss.py) into Qwen2 and TRL's SFTTrainer.

Two patches that work together, both looked up at call time:

1. `Qwen2ForCausalLM.forward`: when the trainer arms the fast path, the labelled training forward
   returns the loss from `chunked_head_loss` and no logits.
2. `SFTTrainer.compute_loss`: arms the fast path, then logs `entropy`, `num_tokens` and
   `mean_token_accuracy` by TRL's formulas from what the kernel returned (TRL reads them from the
   logits, which no longer exist), and leaves the step where Soup's rewind recorder looks for the
   per-token losses.

Anything that is not exactly this case (evaluation, another model, liger, label smoothing, a
fake module in Soup's tests) takes the original code path untouched.
"""

from dataclasses import dataclass
from typing import Any

import torch

from .ce_loss import chunked_head_loss

_FAST = type("FastPath", (), {"armed": False, "step": None})()
_WEIGHT_CACHE = {}


@dataclass(frozen=True)
class HeadStep:
    """What a chunked step leaves for the trainer's metrics (same fields Soup's rewind reads)."""

    token_entropy: Any  # [B, T] entropy of every position's distribution, 0 on padding
    token_loss: Any  # [B, T] float32, position t = cross-entropy of token t + 1, 0 if unlabelled
    correct: Any  # answer tokens whose argmax is the label
    labelled: Any  # answer tokens


def _autocast_dtype():
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except (TypeError, AttributeError):  # older torch: no device argument
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return None


def _head_weight(lm_head, dtype):
    """The head weight in the matmul dtype. Autocast would cast it on every call; it is frozen
    (LoRA is on q/v only), so one cast per version is enough."""
    weight = lm_head.weight
    if weight.dtype == dtype:
        return weight
    key = (weight.data_ptr(), weight._version, dtype)
    cached = _WEIGHT_CACHE.get("w")
    if cached is None or cached[0] != key:
        _WEIGHT_CACHE["w"] = (key, weight.detach().to(dtype))
    return _WEIGHT_CACHE["w"][1]


def _head_ok(model):
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    return (isinstance(head, torch.nn.Linear) and head.bias is None and weight.is_cuda
            and not weight.requires_grad and weight.dtype in (torch.float16, torch.float32)
            and not hasattr(weight, "quant_state"))


def patch_forward():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls.forward, "_soup_head_patched", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        if not (_FAST.armed and labels is not None and self.training and logits_to_keep == 0
                and _head_ok(self)):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        num_items = kwargs.pop("num_items_in_batch", None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        dtype = _autocast_dtype() or self.lm_head.weight.dtype
        mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None
        loss, entropy, token_loss, correct, labelled = chunked_head_loss(
            hidden, _head_weight(self.lm_head, dtype), labels, mask, num_items)
        _FAST.step = HeadStep(entropy, token_loss, correct, labelled)
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    forward._soup_head_patched = True
    cls.forward = forward


def _step_from_logits(logits, labels):
    """The forward fell back to full logits (an unusual head): TRL's own formulas."""
    from trl.trainer.utils import entropy_from_logits

    with torch.no_grad():
        shift_labels = labels[..., 1:]
        mask = shift_labels != -100
        hits = ((logits[..., :-1, :].argmax(dim=-1) == shift_labels) & mask).sum()
        return HeadStep(entropy_from_logits(logits), None, hits, mask.sum())


def _log_metrics(trainer, inputs, step):
    """TRL's `compute_loss` bookkeeping for the three logged columns, from the kernel's numbers."""
    metrics = trainer._metrics["train"]
    attention_mask = inputs["attention_mask"]
    gather = trainer.accelerator.gather_for_metrics
    with torch.no_grad():
        entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
        metrics["entropy"].append(gather(entropy).mean().item())
        trainer._total_train_tokens += gather(attention_mask.sum()).sum().item()
        metrics["num_tokens"] = [trainer._total_train_tokens]
        correct, total = gather(step.correct), gather(step.labelled)
        total_sum = total.sum()
        metrics["mean_token_accuracy"].append(
            (correct.sum() / total_sum).item() if total_sum > 0 else 0.0)


def _eligible(trainer, model, inputs):
    try:
        from transformers.models.qwen2 import modeling_qwen2

        unwrapped = trainer.accelerator.unwrap_model(model)
        core = unwrapped.get_base_model() if hasattr(unwrapped, "get_base_model") else unwrapped
        return (torch.cuda.is_available() and isinstance(core, modeling_qwen2.Qwen2ForCausalLM)
                and trainer.model.training and "labels" in inputs and "attention_mask" in inputs
                and "shift_labels" not in inputs and not trainer.args.use_liger_kernel
                and trainer.compute_loss_func is None and trainer.label_smoother is None
                and getattr(trainer, "num_virtual_tokens", 0) == 0)
    except Exception:  # noqa: BLE001 - anything unexpected (fake modules in tests): no fast path
        return False


def patch_compute_loss():
    import transformers
    from trl import SFTTrainer

    if getattr(SFTTrainer.compute_loss, "_soup_head_patched", False):
        return
    original = SFTTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _eligible(self, model, inputs):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        inputs.pop("_prediction_loss_only", None)  # as TRL's compute_loss does first
        inputs["use_cache"] = False
        _FAST.step, _FAST.armed = None, True
        try:
            loss, outputs = transformers.Trainer.compute_loss(
                self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        finally:
            _FAST.armed = False
        step, _FAST.step = _FAST.step, None
        if step is None:
            step = _step_from_logits(outputs.logits, inputs["labels"])
        else:
            try:  # where Soup's rewind recorder looks for the per-token losses
                from soup_cli.utils.stream_loss_chunk import STEP_ATTR
                setattr(self, STEP_ATTR, step)
            except ImportError:
                pass
        _log_metrics(self, inputs, step)
        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_head_patched = True
    SFTTrainer.compute_loss = compute_loss


def patch_head():
    patch_forward()
    patch_compute_loss()
