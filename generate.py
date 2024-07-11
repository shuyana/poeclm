"""Script to generate samples with GPT models."""

import math
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import cast

import datamol as dm
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import trange

from poeclm.chem import compute_descriptors, process_smiles
from poeclm.model import CompGPT
from poeclm.tokenizer import SMILESTokenizer


@dataclass
class SingleModelConfig:
    """Configuration for a single GPT model."""

    checkpoint_file: str
    weight: float


@dataclass
class TopConfig:
    """Configuration for sample generation."""

    models: list[SingleModelConfig]
    vocab_file: str = "data/vocab.txt"
    device: str = "cuda"
    dtype: str = "bfloat16"
    seed: int = 42
    max_length: int = 96
    temperature: float = 1.0
    top_k: int | None = None
    batch_size: int = 512
    num_samples: int = 2**15
    output_dir: str = "outputs/generation/Prior/85M"


def main() -> None:
    """Generates samples by combining multiple GPT models."""
    base_cfg = OmegaConf.structured(TopConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_cfg, cli_cfg)
    config = cast(TopConfig, OmegaConf.to_object(cfg))

    vocab_file = Path(config.vocab_file)
    output_dir = Path(config.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "generation.yaml")

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

    checkpoint_files, weights = zip(*map(astuple, config.models), strict=True)
    model = CompGPT.from_checkpoint_files(checkpoint_files, weights)
    model.eval()
    model.to(config.device)

    model = torch.compile(model)  # type: ignore

    tokenizer = SMILESTokenizer.from_vocab_file(vocab_file)

    smiles_list: list[str] = []
    bos_idx = torch.full(
        (config.batch_size, 1),
        tokenizer.bos_idx,
        dtype=torch.long,
        device=config.device,
    )
    for _ in trange(math.ceil(config.num_samples / config.batch_size), desc="Generating samples"):
        with torch.inference_mode(), ctx:
            idx = model.generate(
                bos_idx,
                config.max_length - 1,
                temperature=config.temperature,
                top_k=config.top_k,
            )
        smiles_list.extend(tokenizer.decode(idx_list) for idx_list in idx.cpu().tolist())
    data = {
        "smiles": smiles_list[: config.num_samples],
        "id": [f"sample{i}" for i in range(1, config.num_samples + 1)],
    }
    df = pd.DataFrame(data)

    df["standard_smiles"] = dm.parallelized(
        process_smiles,
        df["smiles"],
        progress=True,
        tqdm_kwargs={"desc": "Processing SMILES"},
    )

    descriptors = dm.parallelized(
        lambda smiles: compute_descriptors(smiles) if smiles is not None else {},
        df["standard_smiles"],
        progress=True,
        tqdm_kwargs={"desc": "Computing descriptors"},
    )
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame.from_records(descriptors)], axis=1)

    df.to_parquet(
        output_dir / "samples.parquet",
        engine="pyarrow",
        index=False,
        row_group_size=2**20,
    )

    # Save the first 10K unique SMILES for evaluation
    df = df.dropna(subset="standard_smiles")
    df = df.drop_duplicates(subset="standard_smiles", ignore_index=True)
    if len(df) < 10_000:
        raise RuntimeError(f"Unique SMILES less than 10,000: {len(df):,}")
    df = df[["standard_smiles", "id"]].head(10_000)
    df.to_csv(output_dir / "unique.smi", sep=" ", header=False, index=False)


if __name__ == "__main__":
    main()
