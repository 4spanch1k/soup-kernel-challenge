# LoRA fast path crash review

## Finding

`SOUP_X_LORA=1` removed PEFT's defensive clone and then modified the output of the
bitsandbytes 4-bit base layer in place with `result.add_`. PEFT 0.20.0 explains that its clone is
needed because backward can fail when the 4-bit result is a manipulated view. The fast path
therefore had a concrete autograd risk that its fake linear test did not reproduce: that test's
base layer is `nn.Linear`, not bitsandbytes `Linear4bit`.

The fast path now clones the base result before the in-place scaled addition. It still uses
`add_(..., alpha=scaling)` to avoid a separate scaled update tensor. The test now checks that the
fast result does not alias the base result.

## Confidence and limits

Confidence is **high that this was a credible cause**, because the patch removed exactly the
defensive operation PEFT documents for a backward failure mode. The Kaggle exit-1 log is
unavailable, so this does not prove it was the observed failure. Other runtime differences are
still possible. The CUDA test and a T4 training run have not been run here; keep `SOUP_X_LORA`
disabled until the T4 run confirms the fix.

Source checked: [PEFT v0.20.0 `Linear4bit.forward`](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/bnb.py#L463-L510).
