#!/usr/bin/env python3
"""Time Soup's real model setup calls on the current CUDA device; do not train or save."""

import importlib
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def timed(label, call, torch=None):
    if torch is not None:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    if torch is not None:
        torch.cuda.synchronize()
    print(f"{label}: {time.perf_counter() - start:.3f} s", flush=True)
    return result


def main():
    total_start = time.perf_counter()
    torch = timed("import torch", lambda: importlib.import_module("torch"))
    if not torch.cuda.is_available():
        raise SystemExit("This diagnostic needs a CUDA GPU; run it on the T4.")
    start = time.perf_counter()
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    print(f"CUDA context: {time.perf_counter() - start:.3f} s", flush=True)
    transformers = timed("import transformers", lambda: importlib.import_module("transformers"))
    timed("import solution (no apply)", lambda: importlib.import_module("solution"))
    timed("import soup_cli.cli", lambda: importlib.import_module("soup_cli.cli"))
    timed("import peft", lambda: importlib.import_module("peft"))
    timed("import trl", lambda: importlib.import_module("trl"))
    timed("import bitsandbytes", lambda: importlib.import_module("bitsandbytes"))
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    load_config = importlib.import_module("soup_cli.config.loader").load_config
    cfg = timed("load task/task.yaml", lambda: load_config(ROOT / "task" / "task.yaml"))
    train_command = importlib.import_module("soup_cli.commands.train")
    capture_data_hash = importlib.import_module(
        "soup_cli.utils.training_identity"
    ).capture_training_data_hash
    timed("Soup training data hash", lambda: capture_data_hash(cfg))
    loader_data_config = importlib.import_module(
        "soup_cli.utils.eval_schedule"
    ).loader_data_config
    data_config = train_command.data_config_for_task(loader_data_config(cfg), cfg.task)
    dataset = timed(
        "Soup load_dataset",
        lambda: train_command.load_dataset(
            data_config,
            preserve_source_columns=train_command.task_preserves_source_columns(cfg.task),
        ),
    )

    sft = importlib.import_module("soup_cli.trainer.sft")
    wrapper = timed("Soup SFT wrapper setup", lambda: sft.SFTTrainerWrapper(cfg))
    apply_training_seed = importlib.import_module("soup_cli.utils.seeding").apply_training_seed
    timed("Soup training seed", lambda: apply_training_seed(cfg.training))

    tokenizer = timed(
        "AutoTokenizer.from_pretrained",
        lambda: transformers.AutoTokenizer.from_pretrained(
            cfg.base,
            trust_remote_code=wrapper._trust_remote_code,
            **sft.base_revision_kwargs(cfg.base),
        ),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    build_quantization_config = importlib.import_module(
        "soup_cli.utils.quant_menu"
    ).build_quantization_config_for_loader
    quantization_config = timed(
        "Soup BitsAndBytesConfig",
        lambda: build_quantization_config(
            tcfg=cfg.training,
            base=cfg.base,
        ),
    )
    resolve_device_map = importlib.import_module("soup_cli.utils.gpu").resolve_device_map
    model_kwargs = {
        "trust_remote_code": wrapper._trust_remote_code,
        "device_map": resolve_device_map(wrapper.device),
        "torch_dtype": wrapper._resolve_load_dtype(cfg.training),
        "quantization_config": quantization_config,
    }
    model = timed(
        "AutoModelForCausalLM.from_pretrained",
        lambda: transformers.AutoModelForCausalLM.from_pretrained(
            cfg.base,
            **model_kwargs,
            **sft.base_revision_kwargs(cfg.base),
        ),
        torch,
    )

    apply_vocab_expansion = importlib.import_module(
        "soup_cli.utils.data_pipeline"
    ).apply_vocab_expansion
    timed(
        "Soup apply_vocab_expansion",
        lambda: apply_vocab_expansion(tokenizer, model, cfg.data),
        torch,
    )

    peft = importlib.import_module("peft")
    should_enable_checkpointing = importlib.import_module(
        "soup_cli.utils.layer_stream"
    ).should_enable_hf_gradient_checkpointing
    model = timed(
        "prepare_model_for_kbit_training",
        lambda: peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=should_enable_checkpointing(
                cfg.training.gradient_checkpointing,
                stream_layers=cfg.training.stream_layers,
            ),
        ),
        torch,
    )

    wiring = importlib.import_module("soup_cli.utils.peft_wiring")
    targets = wiring.resolve_lora_target_modules(
        model, cfg.training.lora.target_modules, sft.console
    )
    target_parameters = wiring.resolve_lora_target_parameters(
        model, cfg.training.lora.target_parameters
    )
    targets = importlib.import_module(
        "soup_cli.utils.moe"
    ).resolve_moe_lora_targets(model, cfg.training, targets, sft.console)
    lora_config = wiring.build_lora_config(
        cfg.training.lora,
        target_modules=targets,
        target_parameters=target_parameters,
        task_type=peft.TaskType.CAUSAL_LM,
    )
    timed(
        "Soup pre-LoRA wiring",
        lambda: wiring.apply_pre_lora_patches(model, cfg.base),
        torch,
    )
    model = timed(
        "get_peft_model",
        lambda: peft.get_peft_model(model, lora_config),
        torch,
    )
    timed("Soup post-LoRA wiring", lambda: wiring.apply_post_lora_patches(model), torch)
    timed(
        "Soup fast-LoRA wiring",
        lambda: wiring.apply_fast_lora_patches(model, cfg.training),
        torch,
    )
    print(
        f"Loaded {len(dataset['train'])} training rows. Stopped after LoRA setup: tokenization, "
        "trainer init, training, save, and process exit "
        "are intentionally not run.",
        flush=True,
    )
    print(f"Total through LoRA setup: {time.perf_counter() - total_start:.3f} s", flush=True)


if __name__ == "__main__":
    main()
