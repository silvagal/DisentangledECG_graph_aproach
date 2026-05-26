# Disentangling Morphology and Context in Arrhythmia Classification: A Graph Neural Network Approach

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code for the paper submitted to the **Journal of the Brazilian Computer Society (JBCS)**.

> **Abstract.** We investigate the use of Visibility Graphs (VGs) as a signal-to-graph transformation for ECG beat classification using Graph Neural Networks. ECG beats are represented as graphs where nodes correspond to signal samples and edges encode geometric visibility relationships. We evaluate multiple GNN architectures (GCN, GIN, GINE), graph constructors (VG, HVG, kVG), node/edge feature sets, readout strategies, and positional encodings. We further compare against CNN-based and sequential deep learning baselines. Experiments follow the inter-patient protocol of the MIT-BIH Arrhythmia Database targeting three classes: Normal (N), Supraventricular (S), and Ventricular (V).

---

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Reproducing the Paper Results](#reproducing-the-paper-results)
  - [Running a single experiment](#running-a-single-experiment)
  - [Running a full experiment suite](#running-a-full-experiment-suite)
  - [Feature-only baselines](#feature-only-baselines)
  - [Interpretability analysis](#interpretability-analysis)
- [Configuration Reference](#configuration-reference)
- [Output Format](#output-format)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Requirements

Python 3.9 or later is recommended. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Version |
|---|---|
| torch | 2.2.0 |
| torch_geometric | 2.6.1 |
| ts2vg | 1.2.3 |
| dgl | — |
| scipy | 1.12.0 |
| scikit-learn | 1.4.1 |
| numpy | 1.26.4 |

> **GPU.** Experiments were run on an NVIDIA H200 GPU. A CUDA-capable GPU is strongly recommended; training on CPU is supported but slow.

---

## Dataset

All experiments use the **MIT-BIH Arrhythmia Database** available from PhysioNet:

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. _IEEE Eng Med Biol_ 20(3):45–50, 2001. doi:[10.1109/51.932724](https://doi.org/10.1109/51.932724)

Download the database and organise it following the inter-patient split of De Chazal et al. (2004):

- **DS1** (training): records 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230
- **DS2** (test): records 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

Place the `.mat` signal files in the following layout:

```
Data/
  Train/   ← DS1 .mat files
  Test/    ← DS2 .mat files
```

Beat labels are mapped to three AAMI classes: **N** (Normal), **S** (Supraventricular), **V** (Ventricular).

---

## Repository Structure

```
.
├── src/
│   ├── main.py             # entry point: config → train → evaluate
│   ├── trainer.py          # Trainer.fit() and evaluate_with_report()
│   ├── data_loader.py      # ECGGraphDataset and sequential variants
│   ├── models.py           # all architectures (GCN, GIN, GINE, CNN, Seq2Seq, …)
│   ├── graph_conversion.py # ECG signal → visibility graph
│   ├── losses.py           # focal loss and helpers
│   └── utils.py            # config loading, seeding, positional encodings
│
├── config/                 # one YAML file per experiment
│   ├── Part1_Exp1_1_*.yaml # GNN architecture comparison
│   ├── Part1_Exp1_2_*.yaml # graph constructor comparison (VG/HVG/kVG)
│   ├── Part1_Exp1_3_*.yaml # node and edge feature ablations
│   ├── Part2_Exp2_1_*.yaml # CNN/sequential baselines
│   ├── Part2_GINESeq2Seq*.yaml
│   └── Part3_Exp3_1_*.yaml # readout and positional encoding ablations
│
├── scripts/
│   ├── run_feature_only_baselines.py   # MLP/DNN on handcrafted features
│   ├── run_gnn_explainer_analysis.py   # GNNExplainer interpretability
│   ├── extract_foundation_embeddings.py
│   └── evaluate_foundation_embeddings.py
│
├── suites.py               # suite definitions (lists of configs)
├── run_suite.py            # multi-seed experiment runner
├── aggregate_suite_results.py  # aggregate metrics across seeds
└── requirements.txt
```

---

## Reproducing the Paper Results

### Running a single experiment

```bash
python src/main.py --config config/Part1_Exp1_1_GINE.yaml
```

Results are written to `output/<experiment_name>/`:

```
output/Part1_Exp1_1_GINE/
  metrics.json
  confusion_matrix.csv
  classification_report.txt
  model_state.pt
```

### Running a full experiment suite

`run_suite.py` executes every config in a named suite across multiple random seeds and stores all runs under `output_suite_runs/`.

```bash
# List available suites
python run_suite.py --list

# Reproduce Part 1 (architecture + constructor + feature ablations)
python run_suite.py part1 --runs 5 --seeds 42,153,264,375,486

# Reproduce Part 2 (baselines)
python run_suite.py part2 --runs 5 --seeds 42,153,264,375,486

# Reproduce Part 3 (readout and PE ablations)
python run_suite.py part3 --runs 5 --seeds 42,153,264,375,486

# Full paper pipeline (all parts in sequence)
python run_suite.py paper --runs 5 --seeds 42,153,264,375,486
```

After the runs finish, aggregate statistics across seeds:

```bash
python aggregate_suite_results.py part1
python aggregate_suite_results.py part2
python aggregate_suite_results.py part3
```

### Feature-only baselines

Train and evaluate MLP and DNN classifiers on handcrafted rhythm and statistical descriptors (RR intervals, morphological statistics) without any graph representation:

```bash
python scripts/run_feature_only_baselines.py
```

### Interpretability analysis

Run GNNExplainer on the best-performing GINE+VG+RR model to produce per-class node importance maps:

```bash
python scripts/run_gnn_explainer_analysis.py \
  --config config/Part1_Exp1_3_NodeFeat_RR.yaml \
  --model-path output/Part1_Exp1_3_NodeFeat_RR/model_state.pt
```

---

## Configuration Reference

All experiments are fully specified by YAML files. The critical fields are:

```yaml
seed: 42
device: cuda
experiment_name: MyExperiment

dataset_params:
  path: Data                       # root directory of the dataset
  train_split: Train               # DS1
  test_split: Test                 # DS2
  beat_size_before: 100            # samples before R-peak
  beat_size_after: 180             # samples after R-peak  (280-point window)
  graph_constructor: vg            # "vg" | "hvg" | "kvg"
  feature_set: raw                 # "raw" | "RR" | "Stats"
  edge_features: [delta_t, slope, length]
  undersampling_ratio: 0.1         # fraction of N class kept (legacy protocol)
  undersampling_method: stride     # or "random"
  undersample_test: true           # apply undersampling to test set (legacy)
  paradigm: inter                  # inter-patient split

model_params:
  model_name: gine                 # gcn7 | gcn60 | gin | gine | …
  in_channels: 1
  hidden_channels: 64
  num_layers: 5
  num_classes: 3
  edge_dim: 3
  readout: mean                    # mean | set2set | sortpool

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 150
```

> **Note on undersampling.** `undersample_test: true` applies stride-based undersampling to the test split to maintain the same class-ratio protocol used across all compared models. This is a legacy choice retained for internal consistency; the absolute number of test beats is therefore smaller than the full DS2.

---

## Output Format

`metrics.json` follows this schema:

```json
{
  "N":  {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "fpr": 0.0},
  "S":  {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "fpr": 0.0},
  "V":  {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "fpr": 0.0},
  "weighted avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0},
  "accuracy": 0.0
}
```

`confusion_matrix.csv` uses rows = true labels, columns = predicted labels, ordered N → S → V.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{luz2025arrhythmia,
  title   = {Disentangling Morphology and Context in Arrhythmia Classification:
             A Graph Neural Network Approach},
  author  = {Luz, Eduardo J. S. and others},
  journal = {Journal of the Brazilian Computer Society},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

This work was supported by **FAPEMIG** (grant APQ-01518-21) and by the Federal University of Ouro Preto (UFOP). The MIT-BIH Arrhythmia Database is courtesy of PhysioNet.
