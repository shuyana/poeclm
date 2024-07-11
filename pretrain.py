"""Script to pre-train a chemical language model."""

import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
import wandb
from omegaconf import OmegaConf

from poeclm.data import SMILESDataModule, SMILESTokenizer
from poeclm.model import GPT, GPTConfig


@dataclass
class TopConfig:
    """Configuration for pre-training."""

    model: GPTConfig
    vocab_file: str = "data/vocab.txt"
    dataset_dir: str = "data/datasets/enum_16M"
    filter_expr: str | None = None
    randomize_smiles: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42
    num_workers: int = 2
    batch_size: int = 512
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    use_fused: bool = True
    grad_clip: float = 1.0
    min_lr: float = 2e-5
    max_iters: int = 300000
    warmup_iters: int = 2000
    log_interval: int = 10
    eval_interval: int = 2000
    eval_iters: int = 100
    eval_batch_size: int = 512
    output_dir: str = "outputs/pretraining/enum_16M/85M"


def main() -> None:
    """Pre-train a GPT model."""
    base_cfg = OmegaConf.structured(TopConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    config = cast(TopConfig, OmegaConf.to_object(cfg))

    vocab_file = Path(config.vocab_file)
    dataset_dir = Path(config.dataset_dir)
    output_dir = Path(config.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "pretraining.yaml")
    wandb.init(
        project=f"poeclm-pretraining-{dataset_dir.name}",
        config=OmegaConf.to_container(cfg),  # type: ignore
    )
    wandb.save(str(output_dir / "pretraining.yaml"), str(output_dir))

    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in config.device else "cpu"
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config.dtype]
    ctx = torch.autocast(device_type=device_type, dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and dtype == torch.float16))

    model = GPT(config.model)
    model.to(config.device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = model.configure_optimizers(
        config.learning_rate,
        config.weight_decay,
        (config.beta1, config.beta2),
        config.use_fused,
    )

    model = torch.compile(model)  # type: ignore

    tokenizer = SMILESTokenizer.from_vocab_file(vocab_file)
    assert len(tokenizer) <= model.config.vocab_size

    datamodule = SMILESDataModule(
        dataset_dir,
        tokenizer,
        filter_expr=config.filter_expr,
        randomize_smiles=config.randomize_smiles,
    )

    def loss_fn(idx: torch.Tensor) -> torch.Tensor:
        logits = model(idx)
        logits = logits[:, :-1].contiguous()
        targets = idx[:, 1:].contiguous()
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=tokenizer.pad_idx,
        )

    @torch.inference_mode()
    def estimate_losses(batch_iters: dict[str, Iterator[torch.Tensor]]) -> dict[str, torch.Tensor]:
        out = {}
        model.eval()
        for split, batch_iter in batch_iters.items():
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                idx = next(batch_iter)
                with ctx:
                    loss = loss_fn(idx)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it: int) -> float:
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        if it > config.max_iters:
            return config.min_lr
        decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)

    def get_batch_iter(
        split: Literal["train", "val", "test"],
        batch_size: int,
    ) -> Iterator[torch.Tensor]:
        dataloader = datamodule.get_dataloader(
            split,
            batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        while True:
            for batch in dataloader:
                yield batch.to(config.device)

    train_batch_iter = get_batch_iter("train", config.batch_size)
    eval_batch_iter = {
        "train": get_batch_iter("train", config.eval_batch_size),
        "val": get_batch_iter("val", config.eval_batch_size),
    }
    best_val_loss = 1e9
    t0 = time.perf_counter()
    for iter_num in range(config.max_iters + 1):
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % config.eval_interval == 0:
            losses = estimate_losses(eval_batch_iter)
            train_loss, val_loss = losses["train"].item(), losses["val"].item()
            print(f"iter {iter_num}: train_loss {train_loss:.4f}, val_loss {val_loss:.4f}")
            wandb.log(
                {"train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr},
                step=iter_num,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"config": asdict(model.config), "model": model.state_dict()},
                    output_dir / "best.ckpt",
                )
                wandb.save(str(output_dir / "best.ckpt"), str(output_dir))

        for _ in range(config.gradient_accumulation_steps):
            idx = next(train_batch_iter)
            with ctx:
                loss = loss_fn(idx) / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if iter_num % config.log_interval == 0:
            t1 = time.perf_counter()
            dt = t1 - t0
            t0 = t1
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt:.2f} s")


if __name__ == "__main__":
    main()
