"""Definition of GPT models.

Based on the following implementations:
- nanoGPT: https://github.com/karpathy/nanoGPT
- llama2.c: https://github.com/karpathy/llama2.c
- Lit-LLaMA: https://github.com/Lightning-AI/lit-llama
- Lit-GPT: https://github.com/Lightning-AI/lit-gpt
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TypeAlias

import torch
from torch import nn

from .utils import round_up

RoPECache: TypeAlias = tuple[torch.Tensor, torch.Tensor]


def build_rope_cache(max_seq_length: int, head_size: int) -> RoPECache:
    """Builds a cache for Rotary Positional Embedding (RoPE).

    Args:
        max_seq_length (int): The maximum sequence length.
        head_size (int): The size of a single head.

    Returns:
        RoPECache: A tuple of cos and sin tensors. Each of shape
            (max_seq_length, head_size).
    """
    theta = 1.0 / (10000 ** (torch.arange(0, head_size, 2) / head_size))
    pos = torch.arange(max_seq_length)
    x = torch.outer(pos, theta).repeat(1, 2)
    return torch.cos(x), torch.sin(x)


def apply_rope(x: torch.Tensor, rope: RoPECache) -> torch.Tensor:
    """Applies RoPE to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, nh, T, hs).
        rope (RoPECache): A tuple of cos and sin tensors. Each of shape
            (T, hs).

    Returns:
        torch.Tensor: Output tensor of shape (B, nh, T, hs).
    """
    head_size = x.size(-1)
    cos, sin = rope
    x1 = x[..., : head_size // 2]
    x2 = x[..., head_size // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)


@dataclass
class GPTConfig:
    """Configuration for GPT model."""

    block_size: int = 128
    vocab_size: int = 40
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    norm_eps: float = 1e-5

    @property
    def head_size(self) -> int:
        """The size of a single head."""
        return self.n_embd // self.n_head

    @property
    def intermediate_size(self) -> int:
        """The intermediate size of the MLP."""
        return round_up(int(self.n_embd * 8 / 3), 32)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, size: int, eps: float):
        """Initializer.

        Args:
            size (int): The size of the input.
            eps (float): A small value for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): A tensor of shape (B, T, C).

        Returns:
            torch.Tensor: A tensor of shape (B, T, C).
        """
        x_normed = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.weight * x_normed


class MLP(nn.Module):
    """MLP block."""

    def __init__(self, config: GPTConfig):
        """Initializer.

        Args:
            config (GPTConfig): Configuration for GPT model.
        """
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): A tensor of shape (B, T, C).

        Returns:
            torch.Tensor: A tensor of shape (B, T, C).
        """
        return self.proj(torch.nn.functional.silu(self.fc_1(x)) * self.fc_2(x))


class CausalSelfAttention(nn.Module):
    """Causal self-attention."""

    def __init__(self, config: GPTConfig) -> None:
        """Initializer.

        Args:
            config (GPTConfig): Configuration for GPT model.
        """
        super().__init__()
        self.config = config
        self.attn = nn.Linear(config.n_embd, 3 * config.n_head * config.head_size, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor, rope: RoPECache) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): A tensor of shape (B, T, C).
            rope (RoPECache): A tuple of cos and sin tensors. Each of
                shape (T, hs).

        Returns:
            torch.Tensor: A tensor of shape (B, T, C).
        """
        B, T, C = x.size()

        qkv = self.attn(x).view(B, T, 3 * self.config.n_head, self.config.head_size)
        qkv = qkv.permute(0, 2, 1, 3)  # (B, 3 * nh, T, hs)
        q, k, v = qkv.split(self.config.n_head, dim=1)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(y)


class Block(nn.Module):
    """A transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        """Initializer.

        Args:
            config (GPTConfig): Configuration for GPT model.
        """
        super().__init__()
        self.config = config
        self.norm_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, rope: RoPECache) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): A tensor of shape (B, T, C).
            rope (RoPECache): A tuple of cos and sin tensors. Each of
                shape (T, hs).

        Returns:
            torch.Tensor: A tensor of shape (B, T, C).
        """
        x = x + self.attn(self.norm_1(x), rope)
        return x + self.mlp(self.norm_2(x))


class GPT(nn.Module):
    """GPT model."""

    def __init__(self, config: GPTConfig):
        """Initializer.

        Args:
            config (GPTConfig): A configuration for GPT model.
        """
        super().__init__()
        self.config = config

        rope_cos, rope_sin = build_rope_cache(config.block_size, config.head_size)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                "ln_f": RMSNorm(config.n_embd, config.norm_eps),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        def _init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self.apply(_init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    @classmethod
    def from_checkpoint_file(cls, checkpoint_file: str | Path) -> Self:
        """Creates a GPT model from a checkpoint file.

        Args:
            checkpoint_file (str | Path): A path to a checkpoint file.
                The checkpoint file should contain a dictionary with
                keys "config" and "model" containing the configuration
                and the state dict of the model, respectively.

        Returns:
            GPT: A GPT model.
        """
        checkpoint = torch.load(checkpoint_file)
        model = cls(GPTConfig(**checkpoint["config"]))
        model.load_state_dict(
            {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
        )
        return model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            idx (torch.Tensor): A tensor of shape (B, T).

        Returns:
            torch.Tensor: A tensor of shape (B, T, V).
        """
        T = idx.size(1)

        x = self.transformer["wte"](idx)
        rope = (self.rope_cos[:T], self.rope_sin[:T])
        for block in self.transformer["h"]:
            x = block(x, rope)
        x = self.transformer["ln_f"](x)

        return self.lm_head(x)

    def configure_optimizers(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
        use_fused: bool,
    ) -> torch.optim.Optimizer:
        """Configures the optimizer.

        Args:
            learning_rate (float): A learning rate.
            weight_decay (float): A weight decay.
            betas (tuple[float, float]): A tuple of beta1 and beta2.
            use_fused (bool): Whether to use fused optimizer.

        Returns:
            torch.optim.Optimizer: An optimizer.
        """
        params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        nodecay_params = [p for p in params if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generates new tokens.

        Args:
            idx (torch.Tensor): A tensor of shape (B, T).
            max_new_tokens (int): The maximum number of new tokens to
                generate.
            temperature (float, optional): The temperature for sampling.
                Defaults to 1.0.
            top_k (int | None, optional): The number of top tokens to
                consider. Defaults to None.

        Returns:
            torch.Tensor: A tensor of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :]
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class CompGPT(nn.Module):
    """CompGPT model."""

    def __init__(self, models: Iterable[GPT], weights: Iterable[float]):
        """Initializer.

        Args:
            models (Iterable[GPT]): An iterable of GPT models.
            weights (Iterable[float]): An iterable of weights.
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = list(weights)

    @classmethod
    def from_checkpoint_files(
        cls,
        checkpoint_files: Iterable[str],
        weights: Iterable[float],
    ) -> Self:
        """Creates a CompGPT model from checkpoint files.

        Args:
            checkpoint_files (Iterable[str]): An iterable of paths to
                checkpoint files.
            weights (Iterable[float]): An iterable of weights.

        Returns:
            CompGPT: A CompGPT model.
        """
        models = [GPT.from_checkpoint_file(path) for path in checkpoint_files]
        return cls(models, weights)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            idx (torch.Tensor): A tensor of shape (B, T).

        Returns:
            torch.Tensor: A tensor of shape (B, T, V).
        """
        return sum(
            weight * model(idx) for model, weight in zip(self.models, self.weights, strict=True)
        )

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generates new tokens.

        Args:
            idx (torch.Tensor): A tensor of shape (B, T).
            max_new_tokens (int): The maximum number of new tokens to
                generate.
            temperature (float, optional): The temperature for sampling.
                Defaults to 1.0.
            top_k (int | None, optional): The number of top tokens to
                consider. Defaults to None.

        Returns:
            torch.Tensor: A tensor of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :]
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
