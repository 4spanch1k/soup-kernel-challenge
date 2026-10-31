"""Memory probe: runs harness/_train.py UNCHANGED (same argv) and records, for every training step, the valid
token count of its batch, the memory still allocated after the model forward (saved activations + weights) and the
peak allocated/reserved memory of that step. Writes /tmp/memprobe.json and prints the steps sorted by peak.
    python scripts/t4_mem_probe.py --config <run_dir>/task.yaml --metrics /tmp/m.json --solution
Diagnostics only: not part of solution/, harness/ or tests/."""
import json
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIB = 2**20
rows, cur = [], {}


def install():
    import torch
    import transformers
    from torch.nn.modules.module import register_module_forward_hook, register_module_forward_pre_hook

    def pre(module, args):
        # embed_tokens(input_ids) is called positionally with the batch ids: B x T of the padded batch
        if (type(module).__name__ == "Embedding" and args and torch.is_tensor(args[0]) and args[0].dim() == 2
                and torch.is_grad_enabled() and module.weight.shape[0] > 100000):
            cur["B"], cur["T"] = int(args[0].shape[0]), int(args[0].shape[1])
            cur["valid"] = int((args[0] != 151643).sum().item()) if args[0].dtype == torch.long else None

    def post(module, args, output):
        if type(module).__name__ == "Qwen2Model" and module.training and torch.is_grad_enabled():
            cur["after_model_fwd_mib"] = round(torch.cuda.memory_allocated() / MIB, 1)

    register_module_forward_pre_hook(pre)
    register_module_forward_hook(post)

    class Probe(transformers.TrainerCallback):
        def on_step_begin(self, train_args, state, control, **kw):
            cur.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, train_args, state, control, **kw):
            torch.cuda.synchronize()
            rows.append({"step": len(rows) + 1, **cur,
                         "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
                         "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1)})

    original_init = transformers.Trainer.__init__

    def init(self, *a, **k):
        original_init(self, *a, **k)
        self.add_callback(Probe())

    transformers.Trainer.__init__ = init


def report():
    json.dump(rows, open("/tmp/memprobe.json", "w"), indent=1)
    print("\nMEMPROBE steps sorted by peak allocated MiB (valid tokens, B x T, after model fwd, peak alloc, peak reserved):", flush=True)
    for r in sorted(rows, key=lambda r: -r["peak_alloc_mib"])[:12]:
        print(f'  step {r["step"]:2d}  valid {r.get("valid"):5}  {r.get("B")}x{r.get("T"):<4}  fwd {r.get("after_model_fwd_mib")}  '
              f'peak {r["peak_alloc_mib"]}  reserved {r["peak_reserved_mib"]}', flush=True)


if __name__ == "__main__":
    install()
    sys.argv = [str(REPO / "harness" / "_train.py")] + sys.argv[1:]
    try:
        runpy.run_path(str(REPO / "harness" / "_train.py"), run_name="__main__")
    except SystemExit:
        pass
    finally:
        report()
