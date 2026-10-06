"""Assistant-only SFT. Optional LoRA. Run from the repository root."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bank_agent.training import (
    add_model_arguments, append_jsonl, assert_sft_disjoint, dependency_versions, load_model,
    load_sft_examples, load_tokenizer, padded_batch, restore_checkpoint,
    save_checkpoint, seed_everything, sha256_file,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--data", required=True, help="JSONL with messages lists")
    parser.add_argument("--validation-data", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional total optimizer-step cap, useful for smoke checks")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional first-N subset, useful for smoke checks")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.gradient_accumulation, args.max_length) < 1:
        parser.error("Epochs, batch size, accumulation and max length must be positive")
    if any(value is not None and value < 1 for value in (args.max_steps, args.max_samples)):
        parser.error("Optional max steps/samples must be positive")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        parser.error("Learning rate and max grad norm must be positive")

    import torch
    seed_everything(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args)
    examples, counts = load_sft_examples(args.data, tokenizer, args.max_length)
    if args.max_samples:
        examples = examples[:args.max_samples]
        counts["selected_for_run"] = len(examples)
    validation = None
    validation_counts = None
    if args.validation_data:
        validation, validation_counts = load_sft_examples(
            args.validation_data, tokenizer, args.max_length, allowed_splits=("valid", "validation")
        )
        assert_sft_disjoint(examples, validation)
    model = load_model(args, resume=args.resume)
    device = str(next(model.parameters()).device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )
    configuration = vars(args) | {
        "trainer": "assistant_only_sft", "data_sha256": sha256_file(args.data),
        "validation_sha256": sha256_file(args.validation_data) if args.validation_data else None,
    }
    state = {
        "completed_epochs": 0, "optimizer_steps": 0, "epoch_indices": None,
        "next_batch_index": 0, "epoch_loss_sum": 0.0, "epoch_supervised_tokens": 0,
    }
    if args.resume:
        state = restore_checkpoint(
            optimizer, args.resume, configuration, {"resume", "output", "epochs", "max_steps"}
        )
    (output / "run_metadata.json").write_text(json.dumps({
        "configuration": configuration, "dependencies": dependency_versions(),
        "data": counts, "validation": validation_counts,
        "result_scope": "LLM SFT run; task success requires separate agent evaluation",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.max_steps is not None and state["optimizer_steps"] >= args.max_steps:
        parser.error("--max-steps must exceed the resumed optimizer-step count")
    for epoch in range(state["completed_epochs"], args.epochs):
        model.train()
        indices = state["epoch_indices"]
        if indices is None:
            indices = list(range(len(examples)))
            random.shuffle(indices)
            state["epoch_indices"] = indices
        batches = [indices[i:i + args.batch_size]
                   for i in range(0, len(indices), args.batch_size)]
        loss_sum = state["epoch_loss_sum"]
        supervised_tokens = state["epoch_supervised_tokens"]
        reached_cap = False
        optimizer.zero_grad(set_to_none=True)
        for batch_index in range(state["next_batch_index"], len(batches)):
            batch_indices = batches[batch_index]
            batch = padded_batch(
                [examples[i] for i in batch_indices], tokenizer.pad_token_id, device
            )
            raw_loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(raw_loss):
                raise FloatingPointError("Non-finite SFT loss")
            # The last accumulation window may contain fewer microbatches.
            window_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            window_size = min(args.gradient_accumulation, len(batches) - window_start)
            (raw_loss / window_size).backward()
            count = int((batch["labels"][:, 1:] != -100).sum())
            loss_sum += float(raw_loss.detach()) * count
            supervised_tokens += count
            if (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state["optimizer_steps"] += 1
                state["next_batch_index"] = batch_index + 1
                if args.max_steps is not None and state["optimizer_steps"] >= args.max_steps:
                    reached_cap = True
                    break
        completed_epoch = state["next_batch_index"] == len(batches)
        record = {
            "completed_epoch": completed_epoch,
            "epoch": epoch + 1, "optimizer_steps": state["optimizer_steps"],
            "train_token_nll": loss_sum / max(supervised_tokens, 1),
            "supervised_tokens": supervised_tokens,
        }
        if validation:
            model.eval()
            total_loss, total_tokens = 0.0, 0
            with torch.no_grad():
                for index in range(0, len(validation), args.batch_size):
                    batch = padded_batch(
                        validation[index:index + args.batch_size],
                        tokenizer.pad_token_id, device,
                    )
                    loss = model(**batch, use_cache=False).loss
                    count = int((batch["labels"][:, 1:] != -100).sum())
                    total_loss += float(loss) * count
                    total_tokens += count
            record["validation_token_nll"] = total_loss / max(total_tokens, 1)
        append_jsonl(output / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        if completed_epoch:
            state.update({
                "completed_epochs": epoch + 1, "epoch_indices": None,
                "next_batch_index": 0, "epoch_loss_sum": 0.0, "epoch_supervised_tokens": 0,
            })
            checkpoint_name = f"epoch-{epoch + 1:03d}"
        else:
            state["epoch_loss_sum"] = loss_sum
            state["epoch_supervised_tokens"] = supervised_tokens
            checkpoint_name = f"step-{state['optimizer_steps']:06d}"
        save_checkpoint(
            model, tokenizer, optimizer, output / checkpoint_name, state, configuration
        )
        if reached_cap:
            break
    print(f"Checkpoints: {output.resolve()}. No agent-success claim is inferred from SFT loss.")


if __name__ == "__main__":
    main()
