#!/usr/bin/env python3
"""Launch one solution run and add allocator memory tracing to its _train.py process."""

import argparse
import atexit
import collections
import contextlib
import functools
import gc
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIB = 2**20
MAX_TRACE_ENTRIES = 100_000
LIVE_STATES = {"active_allocated", "active_awaiting_free"}


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _frame_text(frame):
    filename = _field(frame, "filename", "?")
    line = _field(frame, "line", "?")
    name = _field(frame, "name", "?")
    return f"{filename}:{line} ({name})"


def _block_frames(block):
    return tuple(_frame_text(frame) for frame in (block.get("frames") or ()))


def _tensor_index(torch):
    by_storage = collections.defaultdict(list)
    visible = 0
    for value in gc.get_objects():
        if not isinstance(value, torch.Tensor):
            continue
        try:
            if value.device.type != "cuda":
                continue
            address = value.untyped_storage().data_ptr()
            if not address:
                continue
            visible += 1
            metadata = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": value.numel() * value.element_size(),
                "requires_grad": bool(value.requires_grad),
            }
            if metadata not in by_storage[address]:
                by_storage[address].append(metadata)
        except (AttributeError, RuntimeError, TypeError, NotImplementedError):
            continue
    return by_storage, visible


def _live_blocks(snapshot):
    blocks = []
    for segment in snapshot.get("segments", ()):
        for block in segment.get("blocks", ()):
            if block.get("state") in LIVE_STATES:
                blocks.append(block)
    return blocks


def _print_snapshot(snapshot, torch, metadata):
    tensor_map, tensor_count = _tensor_index(torch)
    blocks = _live_blocks(snapshot)
    groups = collections.defaultdict(lambda: {"bytes": 0, "count": 0})
    for block in blocks:
        frames = _block_frames(block) or ("<no Python frame>",)
        group = groups[frames]
        group["bytes"] += int(block.get("size", 0))
        group["count"] += 1

    print(
        f"[mem] snapshot step={metadata['step']} phase={metadata['phase']} "
        f"peak_allocated_mib={metadata['allocated_mib']:.1f} "
        f"peak_reserved_mib={metadata['reserved_mib']:.1f} "
        f"live_blocks={len(blocks)} gc_cuda_tensors={tensor_count}",
        flush=True,
    )
    print("[mem] top live allocation stacks (bytes aggregated by full Python stack):", flush=True)
    for index, (frames, group) in enumerate(
            sorted(groups.items(), key=lambda item: -item[1]["bytes"])[:30], 1):
        print(
            f"[mem] stack {index:02d} size_mib={group['bytes'] / MIB:.1f} "
            f"blocks={group['count']} at={frames[-1]}",
            flush=True,
        )
        for frame in reversed(frames[-5:-1]):
            print(f"[mem]   <- {frame}", flush=True)

    print("[mem] top 30 live blocks by size:", flush=True)
    for index, block in enumerate(sorted(blocks, key=lambda item: -item.get("size", 0))[:30], 1):
        address = int(block.get("address", 0))
        tensors = tensor_map.get(address, ())
        if tensors:
            tensor_text = "; ".join(
                f"{tensor['dtype']} shape={tensor['shape']} grad={tensor['requires_grad']}"
                for tensor in tensors[:3]
            )
        else:
            tensor_text = "tensor metadata unavailable through gc"
        frames = _block_frames(block)
        site = frames[-1] if frames else "<no Python frame>"
        print(
            f"[mem] block {index:02d} size_mib={block.get('size', 0) / MIB:.1f} "
            f"requested_mib={block.get('requested_size', 0) / MIB:.1f} "
            f"state={block.get('state')} address=0x{address:x} "
            f"tensor={tensor_text} site={site}",
            flush=True,
        )
        for frame in reversed(frames[-5:-1]):
            print(f"[mem]   <- {frame}", flush=True)


class MemoryRecorder:
    def __init__(self, torch, output_dir):
        self.torch = torch
        self.read_peak_allocated = torch.cuda.max_memory_allocated
        self.read_peak_reserved = torch.cuda.max_memory_reserved
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.output_dir / "mempeak_steps.jsonl"
        self.snapshot_path = self.output_dir / "mempeak.pickle"
        self.metadata_path = self.output_dir / "mempeak_meta.json"
        self.active = False
        self.training_done = False
        self.phase_name = None
        self.max_allocated = 0
        self.max_reserved = 0
        self.max_allocated_step = None
        self.max_reserved_step = None
        self.snapshot_step = None

    def install_peak_readers(self):
        @functools.wraps(self.read_peak_allocated)
        def max_memory_allocated(*args, **kwargs):
            if self.training_done and self.max_allocated_step is not None:
                return self.max_allocated
            return self.read_peak_allocated(*args, **kwargs)

        @functools.wraps(self.read_peak_reserved)
        def max_memory_reserved(*args, **kwargs):
            if self.training_done and self.max_reserved_step is not None:
                return self.max_reserved
            return self.read_peak_reserved(*args, **kwargs)

        self.torch.cuda.max_memory_allocated = max_memory_allocated
        self.torch.cuda.max_memory_reserved = max_memory_reserved

    def _reset_peak(self):
        self.torch.cuda.reset_peak_memory_stats()

    def start_step(self, step):
        if self.active:
            self.finish_step()
        self.active = True
        self.step = int(step)
        self.tokens = 0
        self.token_count_known = False
        self.token_slots = 0
        self.phase_peaks = {}
        self.step_allocated = 0
        self.step_reserved = 0
        self.peak_phase = "other"
        self.phase_name = "other"
        self._reset_peak()

    def capture_batch_tokens(self, inputs):
        if not self.active or not isinstance(inputs, dict):
            return
        mask = inputs.get("attention_mask")
        input_ids = inputs.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "numel"):
            self.token_slots += int(input_ids.numel())
        if mask is None or not hasattr(mask, "sum"):
            self.token_count_known = False
        elif mask.device.type == "cpu":
            self.tokens += int(mask.sum().item())
        else:
            self.token_count_known = False

    def _save_peak_snapshot(self, allocated, reserved):
        snapshot = self.torch.cuda.memory._snapshot()
        metadata = {
            "step": self.step,
            "phase": self.phase_name,
            "allocated_mib": round(allocated / MIB, 1),
            "reserved_mib": round(reserved / MIB, 1),
            "tokens": self.tokens if self.token_count_known else None,
            "token_slots": self.token_slots,
            "phase_peaks": self.phase_peaks,
            "max_allocated_step": self.max_allocated_step,
            "max_reserved_step": self.max_reserved_step,
        }
        with self.snapshot_path.open("wb") as handle:
            pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.snapshot_step = self.step
        _print_snapshot(snapshot, self.torch, metadata)
        print(f"[mem] snapshot_pickle={self.snapshot_path}", flush=True)

    def _close_phase(self):
        if not self.active or self.phase_name is None:
            return
        allocated = int(self.read_peak_allocated())
        reserved = int(self.read_peak_reserved())
        previous = self.phase_peaks.get(self.phase_name, {"allocated": 0, "reserved": 0})
        self.phase_peaks[self.phase_name] = {
            "allocated": max(previous["allocated"], allocated),
            "reserved": max(previous["reserved"], reserved),
        }
        if allocated > self.step_allocated:
            self.step_allocated = allocated
            self.peak_phase = self.phase_name
        self.step_reserved = max(self.step_reserved, reserved)
        new_allocated_max = allocated > self.max_allocated
        new_reserved_max = reserved > self.max_reserved
        if new_allocated_max:
            self.max_allocated = allocated
            self.max_allocated_step = self.step
        if new_reserved_max:
            self.max_reserved = reserved
            self.max_reserved_step = self.step
        if new_allocated_max or new_reserved_max:
            self._save_peak_snapshot(allocated, reserved)

    @contextlib.contextmanager
    def phase(self, name):
        if not self.active:
            yield
            return
        previous = self.phase_name
        self._close_phase()
        self.phase_name = name
        self._reset_peak()
        try:
            yield
        finally:
            self._close_phase()
            self.phase_name = previous
            self._reset_peak()

    def call_with_forward_phase(self, model, call, args, kwargs):
        if not hasattr(model, "forward"):
            return call(*args, **kwargs)
        original_forward = model.forward
        had_instance_forward = "forward" in getattr(model, "__dict__", {})
        old_instance_forward = getattr(model, "__dict__", {}).get("forward")

        @functools.wraps(original_forward)
        def forward(*forward_args, **forward_kwargs):
            with self.phase("forward"):
                return original_forward(*forward_args, **forward_kwargs)

        model.forward = forward
        try:
            return call(*args, **kwargs)
        finally:
            if had_instance_forward:
                model.forward = old_instance_forward
            else:
                del model.forward

    def finish_step(self):
        if not self.active:
            return
        self._close_phase()
        row = {
            "step": self.step,
            "tokens": self.tokens if self.token_count_known else None,
            "token_slots": self.token_slots,
            "max_memory_allocated_mib": round(self.step_allocated / MIB, 1),
            "max_memory_reserved_mib": round(self.step_reserved / MIB, 1),
            "phase": self.peak_phase,
            "phase_peaks_mib": {
                name: {
                    "allocated": round(values["allocated"] / MIB, 1),
                    "reserved": round(values["reserved"] / MIB, 1),
                }
                for name, values in self.phase_peaks.items()
            },
        }
        with self.steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
        token_text = str(row["tokens"]) if row["tokens"] is not None else "unknown"
        print(
            f"[mem] step={self.step} tokens={token_text} slots={self.token_slots} "
            f"max_allocated_mib={row['max_memory_allocated_mib']:.1f} "
            f"max_reserved_mib={row['max_memory_reserved_mib']:.1f} "
            f"phase={self.peak_phase}",
            flush=True,
        )
        self.active = False
        self.phase_name = None

    def report(self):
        if self.active:
            self.finish_step()
        if self.max_allocated_step is not None:
            print(
                f"[mem] overall_peak_allocated_step={self.max_allocated_step} "
                f"allocated_mib={self.max_allocated / MIB:.1f}",
                flush=True,
            )
            print(
                f"[mem] overall_peak_reserved_step={self.max_reserved_step} "
                f"reserved_mib={self.max_reserved / MIB:.1f} "
                f"snapshot_step={self.snapshot_step} metadata={self.metadata_path}",
                flush=True,
            )


def _install_trainer_patches(transformers, accelerate, recorder):
    Trainer = transformers.Trainer
    TrainerCallback = transformers.TrainerCallback

    class MemoryCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            recorder.start_step(state.global_step + 1)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            recorder.finish_step()
            return control

        def on_train_end(self, args, state, control, **kwargs):
            recorder.finish_step()
            recorder.training_done = True
            return control

    original_init = Trainer.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._t4_mempeak_recorder = recorder
        self.add_callback(MemoryCallback())

    Trainer.__init__ = init

    original_prepare_inputs = Trainer._prepare_inputs

    @functools.wraps(original_prepare_inputs)
    def prepare_inputs(self, inputs):
        active_recorder = getattr(self, "_t4_mempeak_recorder", None)
        if active_recorder is None or not active_recorder.active:
            return original_prepare_inputs(self, inputs)
        active_recorder.capture_batch_tokens(inputs)
        with active_recorder.phase("input"):
            return original_prepare_inputs(self, inputs)

    Trainer._prepare_inputs = prepare_inputs

    original_training_step = Trainer.training_step

    @functools.wraps(original_training_step)
    def training_step(self, model, inputs, *args, **kwargs):
        active_recorder = getattr(self, "_t4_mempeak_recorder", None)
        if active_recorder is None or not active_recorder.active:
            return original_training_step(self, model, inputs, *args, **kwargs)
        had_instance_loss = "compute_loss" in self.__dict__
        old_instance_loss = self.__dict__.get("compute_loss")
        original_compute_loss = self.compute_loss

        @functools.wraps(original_compute_loss)
        def compute_loss(*loss_args, **loss_kwargs):
            current_model = loss_args[0] if loss_args else model
            with active_recorder.phase("loss"):
                return active_recorder.call_with_forward_phase(
                    current_model, original_compute_loss, loss_args, loss_kwargs
                )

        self.compute_loss = compute_loss
        try:
            with active_recorder.phase("other"):
                return original_training_step(self, model, inputs, *args, **kwargs)
        finally:
            if had_instance_loss:
                self.compute_loss = old_instance_loss
            else:
                del self.compute_loss

    Trainer.training_step = training_step

    original_create_optimizer = Trainer.create_optimizer

    @functools.wraps(original_create_optimizer)
    def create_optimizer(self, *args, **kwargs):
        result = original_create_optimizer(self, *args, **kwargs)
        optimizer = self.optimizer
        if optimizer is not None and not getattr(optimizer, "_t4_mempeak_wrapped", False):
            original_step = optimizer.step

            @functools.wraps(original_step)
            def step(*step_args, **step_kwargs):
                if recorder.active:
                    with recorder.phase("optimizer"):
                        return original_step(*step_args, **step_kwargs)
                return original_step(*step_args, **step_kwargs)

            optimizer.step = step
            optimizer._t4_mempeak_wrapped = True
        return result

    Trainer.create_optimizer = create_optimizer

    original_backward = accelerate.Accelerator.backward

    @functools.wraps(original_backward)
    def backward(self, loss, *args, **kwargs):
        if recorder.active:
            with recorder.phase("backward"):
                return original_backward(self, loss, *args, **kwargs)
        return original_backward(self, loss, *args, **kwargs)

    accelerate.Accelerator.backward = backward

    original_clip_grad_norm = getattr(accelerate.Accelerator, "clip_grad_norm_", None)
    if original_clip_grad_norm is not None:
        @functools.wraps(original_clip_grad_norm)
        def clip_grad_norm(self, *args, **kwargs):
            if recorder.active:
                with recorder.phase("backward"):
                    return original_clip_grad_norm(self, *args, **kwargs)
            return original_clip_grad_norm(self, *args, **kwargs)

        accelerate.Accelerator.clip_grad_norm_ = clip_grad_norm


def install_if_training_process():
    if (os.environ.get("SOUP_T4_MEMPEAK") != "1"
            or Path(sys.argv[0]).name != "_train.py"
            or "--solution" not in sys.argv):
        return

    import torch

    torch.cuda.memory._record_memory_history(
        max_entries=MAX_TRACE_ENTRIES,
        stacks="python",
    )
    import accelerate
    import transformers

    output_dir = os.environ["SOUP_MEMPEAK_DIR"]
    recorder = MemoryRecorder(torch, output_dir)
    recorder.install_peak_readers()
    _install_trainer_patches(transformers, accelerate, recorder)
    atexit.register(recorder.report)
    print(
        f"[mem] allocator history enabled max_entries={MAX_TRACE_ENTRIES} "
        f"output_dir={output_dir}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/kaggle/working/t4_mempeak/runs")
    parser.add_argument("--train-file", default=str(REPO / "data" / "train.jsonl"))
    parser.add_argument("--eval-file", default=str(REPO / "data" / "eval_public.jsonl"))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    output_dir = Path(args.out).resolve()
    training_file = Path(args.train_file).resolve()
    evaluation_file = Path(args.eval_file).resolve()
    env = dict(os.environ)
    scripts_dir = str(Path(__file__).resolve().parent)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (scripts_dir, existing_pythonpath) if part
    )
    env["SOUP_T4_MEMPEAK"] = "1"
    env["SOUP_MEMPEAK_DIR"] = str(output_dir / "solution_1" / "mempeak")

    command = [
        sys.executable,
        str(REPO / "harness" / "run_task.py"),
        "--only", "solution",
        "--out", str(output_dir),
        "--train-file", str(training_file),
        "--eval-file", str(evaluation_file),
        "--timeout", str(args.timeout),
    ]
    print(f"[mem] launching: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=REPO, env=env, check=False)
    print(f"[mem] harness_exit_code={result.returncode}", flush=True)
    train_log = output_dir / "solution_1" / "train.log"
    if train_log.exists():
        print(f"[mem] train_log={train_log}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
