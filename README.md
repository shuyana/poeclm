# Product-of-Experts Chemical Language Models

Source code for the paper "Navigating Ultra-Large Virtual Chemical Spaces with Product-of-Experts Chemical Language Models"

## Installation

Install [Poetry](https://python-poetry.org/docs/#installation) and run:

```bash
poetry install
```

Or use the provided Dockerfile:

```bash
docker build -t poeclm -f docker/Dockerfile .
```

```bash
docker run --ipc=host --gpus all --rm -it \
    -e UID=$(id -u) \
    -e GID=$(id -g) \
    -v $(pwd):/home/user/workspace \
    -w /home/user/workspace \
    poeclm:latest \
    /bin/bash
```

## Reproducing the datasets

### Enumerated library used for pre-training

Enumerate a library of compounds:

```bash
poetry run python scripts/enumerate_library.py \
    -b data/chemical_space/enaminebbUS.smi.gz \
    -r data/chemical_space/hartenfeller.csv \
    -o data/chemical_space/enum_16M.smi \
    -n 16_000_000 \
    --prop_filters Ro5 Veber \
    --rd_filters Inpharmatica PAINS \
    --threshold 0.65
```

Prepare a dataset for pre-training:

```bash
poetry run python scripts/prepare_dataset.py \
    -i data/chemical_space/enum_16M.smi \
    -o data/datasets/enum_16M \
    -f 0.96 0.02 0.02 \
    --vocab_file data/vocab.txt \
    --filter_expr 'n_heavy_atoms <= 50'
```

### DOCKSTRING dataset used for fine-tuning

Download the DOCKSTRING dataset:

```bash
poetry run python scripts/download_dockstring.py \
    -o data/dockstring/dataset.csv
```

Prepare a dataset for fine-tuning:

```bash
poetry run python scripts/prepare_dataset.py \
    -i data/dockstring/dataset.csv \
    -o data/datasets/dockstring \
    -f 0.8 0.1 0.1 \
    --vocab_file data/vocab.txt \
    --filter_expr 'n_heavy_atoms <= 50'
```

## Reproducing the models

### Prior models

Pre-train a 6M parameter model on the enumerated library:

```bash
poetry run python pretrain.py \
    "model.n_embd=288" \
    "model.n_layer=6" \
    "model.n_head=6" \
    "output_dir=outputs/pretraining/enum_16M/6M"
```

Pre-train a 25M parameter model on the enumerated library:

```bash
poetry run python pretrain.py \
    "model.n_embd=512" \
    "model.n_layer=8" \
    "model.n_head=8" \
    "output_dir=outputs/pretraining/enum_16M/25M"
```

Pre-train a 85M parameter model on the enumerated library:

```bash
poetry run python pretrain.py \
    "model.n_embd=768" \
    "model.n_layer=12" \
    "model.n_head=12" \
    "output_dir=outputs/pretraining/enum_16M/85M"
```

### Expert and anti-expert models

#### DRD2

Fine-tune the 85M parameter model for DRD2 docking:

```bash
poetry run python finetune.py \
    "checkpoint_file=outputs/pretraining/enum_16M/85M/best.ckpt" \
    "dataset_dir=data/datasets/dockstring" \
    "filter_expr='DRD2 <= -11.1'" \
    "batch_size=32" \
    "output_dir=outputs/finetuning/DRD2+/85M"
```

```bash
poetry run python finetune.py \
    "checkpoint_file=outputs/pretraining/enum_16M/85M/best.ckpt" \
    "dataset_dir=data/datasets/dockstring" \
    "filter_expr='DRD2 > -11.1'" \
    "batch_size=512" \
    "output_dir=outputs/finetuning/DRD2-/85M"
```

#### BBB

Fine-tune the 85M parameter model for BBB permeability:

```bash
poetry run python finetune.py \
    "checkpoint_file=outputs/pretraining/enum_16M/85M/best.ckpt" \
    "dataset_dir=data/datasets/dockstring" \
    "filter_expr='0.5159 * clogp - 0.0277 * tpsa - 0.3462 > 0.0'" \
    "batch_size=256" \
    "output_dir=outputs/finetuning/BBB+/85M"
```

```bash
poetry run python finetune.py \
    "checkpoint_file=outputs/pretraining/enum_16M/85M/best.ckpt" \
    "dataset_dir=data/datasets/dockstring" \
    "filter_expr='0.5159 * clogp - 0.0277 * tpsa - 0.3462 <= 0.0'" \
    "batch_size=512" \
    "output_dir=outputs/finetuning/BBB-/85M"
```

#### QED

Fine-tune the 85M parameter model for QED:

```bash
poetry run python finetune.py \
    "checkpoint_file=outputs/pretraining/enum_16M/85M/best.ckpt" \
    "dataset_dir=data/datasets/dockstring" \
    "filter_expr='qed > 0.6'" \
    "batch_size=256" \
    "output_dir=outputs/finetuning/QED+/85M"
```

```bash
poetry run python finetune.py \
    "checkpoint_file=outputs/pretraining/enum_16M/85M/best.ckpt" \
    "dataset_dir=data/datasets/dockstring" \
    "filter_expr='qed <= 0.6'" \
    "batch_size=256" \
    "output_dir=outputs/finetuning/QED-/85M"
```

## Reproducing the results

### Compound generation

#### Baseline

Generate compounds with the random baseline:

```python
from pathlib import Path

import pandas as pd

df = pd.read_parquet("data/datasets/enum_16M/train.parquet", engine="pyarrow")
df = df.sample(n=2**15, random_state=42, ignore_index=True)

out = Path("outputs/generation/Baseline/Random")
out.mkdir(parents=True, exist_ok=True)

df.to_parquet(out / "samples.parquet", engine="pyarrow", index=False, row_group_size=2**20)

unique_10K = df.head(10_000)
unique_10K[["standard_smiles", "id"]].to_csv(
    out / "unique.smi",
    sep=" ",
    header=False,
    index=False,
)
```

#### Prior

Generate compounds with the 6M parameter model:

```bash
poetry run python generate.py \
    "models=[{checkpoint_file: outputs/pretraining/enum_16M/6M/best.ckpt, weight: 1.0}]" \
    "output_dir=outputs/generation/Prior/6M"
```

Generate compounds with the 25M parameter model:

```bash
poetry run python generate.py \
    "models=[{checkpoint_file: outputs/pretraining/enum_16M/25M/best.ckpt, weight: 1.0}]" \
    "output_dir=outputs/generation/Prior/25M"
```

Generate compounds with the 85M parameter model:

```bash
poetry run python generate.py \
    "models=[{checkpoint_file: outputs/pretraining/enum_16M/85M/best.ckpt, weight: 1.0}]" \
    "output_dir=outputs/generation/Prior/85M"
```

#### Expert (DRD2)

Generate compounds with the expert model (DRD2):

```bash
poetry run python generate.py \
    "models=[{checkpoint_file: outputs/finetuning/DRD2+/85M/best.ckpt, weight: 1.0}]" \
    "output_dir=outputs/generation/Expert/DRD2+"
```

#### PoE

Generate compounds with the PoE model using different configurations:

```bash
poetry run python generate.py \
    "models=[
        {checkpoint_file: outputs/pretraining/enum_16M/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/DRD2+/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/DRD2-/85M/best.ckpt, weight: -1.0},
    ]" \
    "output_dir=outputs/generation/PoE/DRD2=1.0"
```

```bash
poetry run python generate.py \
    "models=[
        {checkpoint_file: outputs/pretraining/enum_16M/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/DRD2+/85M/best.ckpt, weight: 1.5},
        {checkpoint_file: outputs/finetuning/DRD2-/85M/best.ckpt, weight: -1.5},
    ]" \
    "output_dir=outputs/generation/PoE/DRD2=1.5"
```

```bash
poetry run python generate.py \
    "models=[
        {checkpoint_file: outputs/pretraining/enum_16M/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/DRD2+/85M/best.ckpt, weight: 2.0},
        {checkpoint_file: outputs/finetuning/DRD2-/85M/best.ckpt, weight: -2.0},
    ]" \
    "output_dir=outputs/generation/PoE/DRD2=2.0"
```

```bash
poetry run python generate.py \
    "models=[
        {checkpoint_file: outputs/pretraining/enum_16M/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/DRD2+/85M/best.ckpt, weight: 2.5},
        {checkpoint_file: outputs/finetuning/DRD2-/85M/best.ckpt, weight: -2.5},
    ]" \
    "output_dir=outputs/generation/PoE/DRD2=2.5"
```

```bash
poetry run python generate.py \
    "models=[
        {checkpoint_file: outputs/pretraining/enum_16M/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/DRD2+/85M/best.ckpt, weight: 2.0},
        {checkpoint_file: outputs/finetuning/DRD2-/85M/best.ckpt, weight: -2.0},
        {checkpoint_file: outputs/finetuning/BBB+/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/BBB-/85M/best.ckpt, weight: -1.0},
    ]" \
    "output_dir=outputs/generation/PoE/DRD2=2.0_BBB=1.0"
```

```bash
poetry run python generate.py \
    "models=[
        {checkpoint_file: outputs/pretraining/enum_16M/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/DRD2+/85M/best.ckpt, weight: 2.0},
        {checkpoint_file: outputs/finetuning/DRD2-/85M/best.ckpt, weight: -2.0},
        {checkpoint_file: outputs/finetuning/QED+/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/QED-/85M/best.ckpt, weight: -1.0},
    ]" \
    "output_dir=outputs/generation/PoE/DRD2=2.0_QED=1.0"
```

```bash
poetry run python generate.py \
    "models=[
        {checkpoint_file: outputs/pretraining/enum_16M/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/DRD2+/85M/best.ckpt, weight: 2.0},
        {checkpoint_file: outputs/finetuning/DRD2-/85M/best.ckpt, weight: -2.0},
        {checkpoint_file: outputs/finetuning/BBB+/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/BBB-/85M/best.ckpt, weight: -1.0},
        {checkpoint_file: outputs/finetuning/QED+/85M/best.ckpt, weight: 1.0},
        {checkpoint_file: outputs/finetuning/QED-/85M/best.ckpt, weight: -1.0},
    ]" \
    "output_dir=outputs/generation/PoE/DRD2=2.0_BBB=1.0_QED=1.0"
```

### Evaluation

#### Similarity search with SpaceLight

Download SpaceLight from [the AMD Software Server](https://software.zbh.uni-hamburg.de/) (requires a license) and extract it to the `third_party` directory.

Generate a topological fragment space for similarity search with SpaceLight:

```bash
poetry run python scripts/generate_spacelight_config.py \
    -b data/chemical_space/enaminebbUS.smi.gz \
    -r data/chemical_space/hartenfeller.csv \
    -o data/chemical_space/JSon
```

```bash
./third_party/SpaceLight_1.2.2/SpaceLight generate \
    -i data/chemical_space/JSon \
    -f data/chemical_space/fragspace.tfsdb
```

Run similarity search with SpaceLight:

```bash
for path in $(find outputs/generation -name unique.smi | sort); do
    ./third_party/SpaceLight_1.2.2/SpaceLight search \
        -f data/chemical_space/fragspace.tfsdb \
        -i $path \
        -o ${path%.smi}_SpaceLight.csv
done
```

#### Molecular docking with DOCKSTRING

Run molecular docking with DOCKSTRING (requires openbabel):

```bash
for path in $(find outputs/generation -name unique.smi | sort); do
    poetry run python scripts/run_docking.py \
        -t DRD2 \
        -i $path \
        -o ${path%.smi}_DRD2.sdf
done
```

## Acknowledgements

The code is based on:

- https://github.com/karpathy/nanoGPT/
- https://github.com/karpathy/llama2.c/
- https://github.com/Lightning-AI/lit-llama/
- https://github.com/Lightning-AI/litgpt/
- http://www.dalkescientific.com/writings/diary/archive/2020/10/07/intersection_popcount.html/

The building blocks are from: https://zinc20.docking.org/catalogs/enaminebbUS/

The chemical reactions are from: https://doi.org/10.1021/ci200379p/

The DOCKSTRING dataset is from: https://doi.org/10.1021/acs.jcim.1c01334/
