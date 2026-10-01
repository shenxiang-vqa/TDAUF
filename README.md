This document provides the implementation details required to reproduce the experiments reported for **TDAUF: Two-stage Dynamic Alignment with Uncertainty-aware Fusion**.

To improve reproducibility, we explicitly document:

1. project organization;
2. model and training configuration;
3. random seeds and deterministic settings;
4. parameter initialization;
5. task-specific preprocessing;
6. software and hardware environment;
7. training and evaluation commands;
8. multi-seed result organization.

---

## 1. Project Structure

The implementation is organized as follows:

```text
TDAUF/
├── configs/
│   ├── vqa/
│   │   ├── vqa2_tdauf.yaml
│   │   └── gqa_tdauf.yaml
│   ├── grounding/
│   │   ├── refcoco_tdauf.yaml
│   │   ├── refcocop_tdauf.yaml
│   │   └── refcocog_tdauf.yaml
│   └── retrieval/
│       └── flickr30k_tdauf.yaml
│
├── tdauf/
│   ├── core/
│   │   ├── config.py
│   │   ├── seed.py
│   │   └── initialization.py
│   │
│   ├── datasets/
│   │   ├── vqa/
│   │   ├── gqa/
│   │   ├── grounding/
│   │   └── retrieval/
│   │
│   ├── models/
│   │   └── tdauf/
│   │       ├── net.py
│   │       ├── pgdc.py
│   │       ├── cdsr.py
│   │       ├── uadf.py
│   │       └── modules.py
│   │
│   ├── engine/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   │
│   └── utils/
│       ├── checkpoint.py
│       ├── logger.py
│       └── metrics.py
│
├── data/
│   ├── vqa/
│   ├── gqa/
│   ├── grounding/
│   └── flickr30k/
│
├── scripts/
│   ├── train_vqa2.sh
│   ├── train_gqa.sh
│   ├── train_refcoco.sh
│   ├── train_refcocop.sh
│   ├── train_refcocog.sh
│   └── train_flickr30k.sh
│
├── checkpoints/
├── results/
├── requirements.txt
├── requirements-lock.txt
├── run.py
└── REPRODUCIBILITY.md
## 2. Model Configuration

TDAUF is implemented using a Transformer encoder--decoder architecture. The main architectural and training settings are summarized below.

### 2.1 Transformer Architecture

- Encoder layers: `6`
- Decoder layers: `6`
- Attention heads: `8`
- Dimensions per head: `64`
- Hidden dimension: `512`
- Final multimodal representation dimension: `1024`

### 2.2 Visual Representation

The main experiments use two complementary visual representations:

- **Grid-level features:** CLIP-RN50x16
- **Region-level features:** VinVL-X152-C4

The extracted visual features are projected into the common Transformer representation space before cross-modal reasoning.

### 2.3 Text Representation

- **Text encoder:** CLIP-RN50x16 text encoder
- **Training strategy:** the pretrained text encoder is frozen during task-specific training.

### 2.4 Stage-I: Prompt-Guided Dynamic Calibration (PGDC)

Stage-I performs language-guided visual evidence calibration before deeper cross-modal reasoning.

The main hyperparameters are:

| Hyperparameter | Value |
|---|---:|
| Top-k ratio | 0.3 |
| LoRA rank | 8 |
| Prompt tokens | 4 |

FiLM-based modulation is used for language-conditioned visual adjustment, while LoRA provides parameter-efficient task-specific adaptation.

### 2.5 Stage-II: Capsule-Driven Semantic Refinement (CDSR)

Stage-II performs visual-feedback-driven semantic refinement based on the calibrated visual representation produced by Stage-I.

The main hyperparameters are:

| Hyperparameter | Value |
|---|---:|
| Number of capsules | 10 |
| Routing iterations | 4 |
| Geometric decay coefficient | 0.6 |

Dynamic capsule routing is used to model structured associations between visual evidence and linguistic semantic units.

### 2.6 Uncertainty-Aware Dynamic Fusion (UADF)

The UADF module adaptively combines multimodal representations according to both representational relevance and predictive reliability.

The complete fusion strategy incorporates:

- modality-level confidence;
- KL-based reliability estimation;
- adaptive multimodal weighting.

### 2.7 Optimization

The common optimization settings are:

| Setting | Value |
|---|---:|
| Optimizer | Adam |
| Batch size | 64 |

Task-specific learning schedules and dataset-dependent settings are specified in the corresponding configuration files under:

```text
configs/
```

For example:

```text
configs/
├── vqa/
│   ├── vqa2_tdauf.yaml
│   └── gqa_tdauf.yaml
├── grounding/
│   ├── refcoco_tdauf.yaml
│   ├── refcocop_tdauf.yaml
│   └── refcocog_tdauf.yaml
└── retrieval/
    └── flickr30k_tdauf.yaml
```

---

## 3. Random Seeds and Deterministic Settings

To improve experimental reproducibility, all results reported as **mean ± standard deviation** are obtained from three independent runs using fixed random seeds.

### 3.1 Random Seeds

The exact random seeds used in the reported experiments are:

```text
<SEED_1>
<SEED_2>
<SEED_3>
```

Before release, replace the placeholders above with the actual random seeds used to produce the reported results.

For example, if the final experiments use:

```text
42
3407
2026
```

the same seeds should be used consistently when reproducing the corresponding multi-run results.

### 3.2 Controlled Random Sources

For each independent run, the random state is controlled for:

- Python;
- NumPy;
- PyTorch CPU;
- PyTorch CUDA;
- all available CUDA devices;
- DataLoader workers.

A reference implementation is shown below:

```python
import os
import random

import numpy as np
import torch


def set_random_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducible experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
```

### 3.3 DataLoader Reproducibility

For multi-worker data loading, worker-level random states are initialized explicitly:

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator
```

The DataLoader can then be constructed as:

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=build_generator(seed),
    pin_memory=True,
)
```

### 3.4 Deterministic Training

When deterministic execution is enabled, the following settings are used:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Where supported by the installed PyTorch/CUDA environment, deterministic algorithms can additionally be enabled through:

```python
torch.use_deterministic_algorithms(True)
```

### 3.5 Running Different Seeds

A single experiment can be executed using:

```bash
python run.py \
    --config configs/vqa/vqa2_tdauf.yaml \
    --seed <SEED> \
    --deterministic \
    --mode train
```

The three reported runs should be executed independently:

```bash
python run.py \
    --config configs/vqa/vqa2_tdauf.yaml \
    --seed <SEED_1> \
    --deterministic \
    --mode train

python run.py \
    --config configs/vqa/vqa2_tdauf.yaml \
    --seed <SEED_2> \
    --deterministic \
    --mode train

python run.py \
    --config configs/vqa/vqa2_tdauf.yaml \
    --seed <SEED_3> \
    --deterministic \
    --mode train
```

### 3.6 Result Organization

For traceability, each run is stored independently:

```text
results/
└── vqa2/
    ├── seed_<SEED_1>/
    ├── seed_<SEED_2>/
    └── seed_<SEED_3>/
```

Each run directory should contain the corresponding configuration, seed, evaluation metrics, and checkpoint information.

> **Reproducibility note:** The random seeds and deterministic settings documented here should exactly match those used to obtain the results reported in the manuscript.


