<div align="center">

# TDAUF

### Two-stage Dynamic Alignment with Uncertainty-aware Fusion

**Dynamic Multimodal Alignment for Vision--Language Understanding**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![GPU](https://img.shields.io/badge/GPU-NVIDIA%20L40-76B900)

**VQA-v2 · GQA · RefCOCO · RefCOCO+ · RefCOCOg · Flickr30K**

</div>

---

## Overview

This repository provides the official implementation of:

> **TDAUF: Two-stage Dynamic Alignment with Uncertainty-aware Fusion**

TDAUF is a unified vision--language framework designed to improve multimodal
alignment through progressive visual calibration, structured language
refinement, and reliability-aware multimodal fusion.

The framework contains three key components:

- **PGDC — Prompt-Guided Dynamic Calibration:** dynamically calibrates
  visual evidence according to linguistic guidance.

- **DCLA — Dynamic Capsule Language Alignment:** organizes language-token
  representations into latent semantic units through dynamic capsule routing
  and refines them using visual grounding signals.

- **UADF — Uncertainty-Aware Dynamic Fusion:** adaptively combines multimodal
  representations according to confidence and reliability.

TDAUF is evaluated on three representative vision--language tasks:

| Task | Benchmarks |
|:---|:---|
| Visual Question Answering | VQA-v2, GQA |
| Visual Grounding | RefCOCO, RefCOCO+, RefCOCOg |
| Image--Text Retrieval | Flickr30K |

---

## Highlights

- **Two-stage dynamic alignment:** progressively calibrates visual evidence
  and refines linguistic semantics.

- **Prompt-guided visual calibration:** combines FiLM modulation, Top-k
  evidence selection, and LoRA-based adaptation.

- **Language-only capsule routing:** constructs latent semantic capsules
  exclusively from language-token votes.

- **Visual-guided semantic refinement:** visual information acts as an
  external grounding and reliability signal for language refinement.

- **Uncertainty-aware fusion:** integrates modality confidence and KL-based
  reliability for adaptive multimodal fusion.

- **Multi-task evaluation:** evaluated on VQA, visual grounding, and
  image--text retrieval benchmarks.

- **Reproducible experiments:** supports deterministic execution,
  independent multi-seed training, checkpoint tracking, and statistical
  significance analysis.

---

## 1. Project Structure

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
│   │       ├── dcla.py
│   │       ├── uadf.py
│   │       └── modules.py
│   │
│   ├── engine/
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── checkpoint.py
│       ├── logger.py
│       └── metrics.py
│
├── data/
├── scripts/
├── checkpoints/
├── results/
├── requirements.txt
├── requirements-lock.txt
├── run.py
└── README.md
```

### Paper-to-Code Mapping

| Component | Implementation |
|:---|:---|
| TDAUF framework | `tdauf/models/tdauf/net.py` |
| PGDC | `tdauf/models/tdauf/pgdc.py` |
| DCLA | `tdauf/models/tdauf/dcla.py` |
| UADF | `tdauf/models/tdauf/uadf.py` |
| Shared modules | `tdauf/models/tdauf/modules.py` |
| Seed control | `tdauf/core/seed.py` |
| Training | `tdauf/engine/train.py` |
| Evaluation | `tdauf/engine/evaluate.py` |

---

## 2. Model Configuration

### Transformer Backbone

| Setting | Value |
|:---|---:|
| Encoder layers | 6 |
| Decoder layers | 6 |
| Attention heads | 8 |
| Dimension per head | 64 |
| Hidden dimension | 512 |
| Multimodal representation | 1024 |

### Visual and Language Representations

| Representation | Configuration |
|:---|:---|
| Grid-level visual features | CLIP-RN50x16 |
| Region-level visual features | VinVL-X152-C4 |
| Text encoder | CLIP-RN50x16 text encoder |
| Text encoder training | Frozen |

Grid-level and region-level visual features are projected into the common
hidden space before multimodal reasoning.

### PGDC

| Hyperparameter | Value |
|:---|---:|
| Top-k ratio | 0.3 |
| LoRA rank | 8 |
| Prompt tokens | 4 |

PGDC performs language-guided visual calibration using FiLM modulation,
Top-k evidence selection, and LoRA-based task adaptation.

### DCLA

| Hyperparameter | Value |
|:---|---:|
| Number of capsules | 10 |
| Routing iterations | 4 |
| Geometric decay coefficient | 0.6 |

> **Important:** DCLA performs capsule routing exclusively over
> **language-token votes**. Visual tokens are not directly routed into
> capsules. Instead, visual information is summarized and used as an
> external grounding and reliability signal during language refinement.

### UADF

UADF performs adaptive multimodal fusion using:

- modality-level confidence;
- KL-based reliability estimation;
- adaptive multimodal weighting.

The repository also supports the fusion variants used in the ablation study:
`Equal-weight`, `Confidence-only`, and `KL-reliability-only`.

---

## 3. Datasets

The experiments cover six widely used vision--language benchmarks.

| Task | Dataset | Evaluation |
|:---|:---|:---|
| VQA | VQA-v2 | Yes/No, Number, Other, All |
| VQA | GQA | All, Binary, Open, Validity, Plausibility, Consistency, Distribution |
| Grounding | RefCOCO | val / testA / testB |
| Grounding | RefCOCO+ | val / testA / testB |
| Grounding | RefCOCOg | val / test |
| Retrieval | Flickr30K | R@1, R@5, R@10, Rsum |

Datasets are not redistributed with this repository. Please download them
from their official sources and follow the corresponding licenses.

Recommended directory structure:

```text
data/
├── vqa/
├── gqa/
├── grounding/
│   ├── refcoco/
│   ├── refcocop/
│   └── refcocog/
└── flickr30k/
```

Task-specific preprocessing and dataset paths are defined in the corresponding
configuration files under `configs/`.

---

## 4. Environment

### Hardware

The experiments are conducted using **NVIDIA L40 GPUs**.

| Hardware | Configuration |
|:---|:---|
| GPU | NVIDIA L40 |
| GPU memory | 48 GB |
| CPU | Multi-core x86-64 |
| System memory | >= 64 GB recommended |
| Storage | SSD recommended |

### Recommended Software Environment

| Software | Version |
|:---|:---|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10 |
| PyTorch | 2.1.x |
| CUDA | 12.1 |
| cuDNN | 8.9.x |
| NumPy | 1.26.x |

> The versions above provide the recommended reproduction environment.
> For exact dependency versions, please refer to `requirements-lock.txt`.

### Installation

```bash
git clone <TDAUF_REPOSITORY_URL>
cd TDAUF

conda create -n tdauf python=3.10 -y
conda activate tdauf

pip install -r requirements-lock.txt
```

Verify the GPU environment:

```bash
python -c "import torch; print(torch.__version__); \
print(torch.version.cuda); \
print(torch.cuda.get_device_name(0))"
```

Expected GPU:

```text
NVIDIA L40
```

---

## 5. Training

The general training command is:

```bash
python run.py \
    --config <CONFIG_FILE> \
    --seed <SEED> \
    --deterministic \
    --mode train
```

### Example: VQA-v2

```bash
python run.py \
    --config configs/vqa/vqa2_tdauf.yaml \
    --seed 42 \
    --deterministic \
    --mode train
```

Other tasks can be reproduced by changing the configuration file:

```text
VQA-v2     configs/vqa/vqa2_tdauf.yaml
GQA        configs/vqa/gqa_tdauf.yaml
RefCOCO    configs/grounding/refcoco_tdauf.yaml
RefCOCO+   configs/grounding/refcocop_tdauf.yaml
RefCOCOg   configs/grounding/refcocog_tdauf.yaml
Flickr30K  configs/retrieval/flickr30k_tdauf.yaml
```

The common optimization settings are:

| Setting | Value |
|:---|---:|
| Optimizer | Adam |
| Batch size | 64 |
| Independent runs | 3 |
| Model selection | Validation performance |

Task-specific learning rates, schedules, epochs, warm-up settings, and other
optimization parameters are stored in the corresponding YAML files.

---

## 6. Evaluation

Evaluate a trained model using:

```bash
python run.py \
    --config <CONFIG_FILE> \
    --seed <SEED> \
    --mode evaluate \
    --checkpoint <CHECKPOINT_PATH>
```

The same preprocessing pipeline and official task metrics are used for both
validation and final evaluation.

Checkpoints are selected **only according to validation performance**.
Test data are not used for checkpoint selection or hyperparameter tuning.

Example result organization:

```text
results/
├── vqa2/
│   ├── seed_42/
│   │   ├── config.yaml
│   │   ├── metrics.json
│   │   ├── run_info.json
│   │   └── train.log
│   ├── seed_<SEED_2>/
│   └── seed_<SEED_3>/
├── gqa/
├── refcoco/
├── refcocop/
├── refcocog/
└── flickr30k/
```

---

## 7. Reproducibility

### Random Seeds

All results reported as **mean ± standard deviation** are obtained from
three independent runs.

```text
Seed 1: <SEED_1>
Seed 2: <SEED_2>
Seed 3: <SEED_3>
```

The implementation controls the random state of:

- Python;
- NumPy;
- PyTorch CPU/CUDA;
- CUDA devices;
- DataLoader workers.

For deterministic execution:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

### Multi-Seed Experiments

The three runs are executed independently:

```bash
for SEED in <SEED_1> <SEED_2> <SEED_3>
do
    python run.py \
        --config configs/vqa/vqa2_tdauf.yaml \
        --seed ${SEED} \
        --deterministic \
        --mode train
done
```

Each run is stored separately, and the final result is reported as:

```text
mean ± standard deviation
```

Exact bitwise equivalence across different hardware/software environments is
not guaranteed. Reproduction should therefore focus on consistency with the
reported multi-run metric range.

---

## 8. Statistical Significance

Statistical significance analyses reported in the paper use a
**two-sided Welch's t-test**.

The evaluation protocol is:

| Setting | Value |
|:---|:---|
| Test | Two-sided Welch's t-test |
| Significance level | 0.05 |
| Internal TDAUF runs | 3 vs. 3 |
| Reported statistics | mean ± standard deviation |

A result is considered statistically significant when:

```text
p < 0.05
```

For comparisons with external methods, tests based on published
mean/standard-deviation statistics are treated as **auxiliary statistical
evidence**, rather than paired or fully backbone-controlled tests.

This follows the statistical evaluation protocol reported in the manuscript.

---

## 9. Reproduction Checklist

Before comparing reproduced results with the paper, please verify:

- [ ] Correct dataset splits and preprocessing are used.
- [ ] CLIP-RN50x16 and VinVL-X152-C4 visual representations are used.
- [ ] The pretrained CLIP text encoder is frozen.
- [ ] PGDC, DCLA, and UADF configurations match the released YAML files.
- [ ] DCLA performs capsule routing over language-token votes only.
- [ ] The same random seeds and deterministic settings are used.
- [ ] Checkpoints are selected using validation performance only.
- [ ] Evaluation uses the released metric implementation.
- [ ] Three independent runs are reported as mean ± standard deviation.

---

### TDAUF

**Two-stage Dynamic Alignment with Uncertainty-aware Fusion**

*Dynamic alignment · Uncertainty-aware fusion · Vision--language learning*

</div>
