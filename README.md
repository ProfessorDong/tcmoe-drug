# tcmoe-drug

**Target-Conditioned Multi-View Mixture-of-Experts for Few-Shot Drug Discovery**

Reference implementation for the IEEE JBHI manuscript *"Target-Conditioned
Multi-View Mixture-of-Experts for Few-Shot Drug Discovery"* (Dong et al.).
Hosted at [github.com/ProfessorDong/tcmoe-drug](https://github.com/ProfessorDong/tcmoe-drug).

The framework couples a dual-representation encoder (graph attention network +
bidirectional GRU + pre-pooling cross-attention) with a multi-view mixture-of-experts
predictor that performs target-conditioned sparse top-`K` routing over four
molecular views (Morgan ECFP4, dual-encoder embeddings, MACCS keys, and
standardised RDKit descriptors). A stack-augmented recurrent generator trained
with REINFORCE produces target-specific candidates that are scored through the
same MoE predictor.

## Repository layout

```
.
├── data/                       # bioactivity datasets (SCD-1, NK1R, DRD2, FADS) + ChEMBL pretrain
├── models/                     # encoder, MoE, generator, legacy predictor
├── reinforcement/              # REINFORCE policy gradient + reward functions
├── meta_learning/              # MAML / first-order MAML baseline
├── interpretability/           # attention viz, IG attribution
├── utils/                      # chemistry utils, training/evaluation helpers
├── baselines/                  # baseline configs
├── outputs/moe_v5/             # canonical result JSONs reproducing manuscript tables
├── run_moe_predictor.py        # main MoE experiments (ablation, fewshot, selectivity, top-K)
├── run_meta_learning.py        # MAML baseline
├── run_moe_figures.py          # Figs 2 (fewshot) and 5 (router utilization)
├── generate_paper_figures.py   # Fig 3 (property distributions)
├── run_interpretability.py     # Fig 4 / interpretability analysis
├── run_example.py              # end-to-end RL training example
├── evaluate_v3.py              # generation evaluation (validity, novelty, etc.)
├── evaluate_reinvent4.py       # REINVENT 4 baseline evaluation
└── verify_tables.py            # cross-check reported numbers against JSONs
```

## Datasets

The four bioactivity datasets in `data/` were curated from ChEMBL.
- `scd1_binding.csv` — Stearoyl-CoA Desaturase 1 (762 compounds)
- `nk1r_combined.csv` — Neurokinin 1 Receptor (3,056)
- `drd2_bioactivity.csv` — D2 Dopamine Receptor (9,966)
- `fatty_acid_desaturase_bioactivity.csv` — FADS1/2 (1,187)
- `chembl_pretrain_smiles.csv` — 250,000 drug-like SMILES used for generator pretraining

All datasets are split 80/10/10 using Bemis–Murcko scaffold splitting
implemented via RDKit (`utils/chemistry.py`).

## Installation

Tested with Python 3.12, PyTorch 2.10, RDKit 2025.09 on Linux with an
NVIDIA GPU (≥ 8 GB) or CPU.

```bash
# create environment
conda create -n moe python=3.12 -y
conda activate moe

# core deps
pip install -r requirements.txt
```

If you do not have a CUDA-compatible GPU, replace the PyTorch install in
`requirements.txt` with the CPU wheel and pass `--device cpu` to the run
scripts (default device is `cuda:0`, with automatic CPU fallback if CUDA is
unavailable).

## Reproducing the paper

The canonical result JSONs that back every numeric claim in the manuscript
ship with the repository at `outputs/moe_v5/*.json` (~80 KB total). To
verify the reported tables against these JSONs without re-running training:

```bash
python verify_tables.py
```

To regenerate everything from scratch:

```bash
# Within-target ablation (Table II)
python run_moe_predictor.py --experiment ablation

# Few-shot leave-one-target-out for each held-out target (Table III)
for held in scd1 fads drd2 nk1r; do
    python run_moe_predictor.py --experiment fewshot --held_out $held
done

# Top-K and target-conditioning ablation (Table V)
python run_moe_predictor.py --experiment topk

# Cross-target selectivity (Table VI)
python run_moe_predictor.py --experiment selectivity

# MAML negative-result baseline (Table III(a)-(b) MAML rows)
python run_meta_learning.py --held_out scd1
python run_meta_learning.py --held_out fads2_expanded

# Figures
python run_moe_figures.py        # Fig 2 + Fig 5
python generate_paper_figures.py # Fig 3
python run_interpretability.py   # Fig 4 source data
```

Default seeds: 5 random seeds per experiment, deterministic under PyTorch.
Each leave-one-target-out fewshot run takes ~5–10 min on a single RTX 4090,
or several hours on CPU.

## Citation

If you use this code or the result protocols, please cite:

```bibtex
@article{dong2026target,
  title   = {Target-Conditioned Multi-View Mixture-of-Experts for Few-Shot Drug Discovery},
  author  = {Dong, Liang and Ding, Tianqi and Gonzalez, Paulina and {\"O}z, Orhan K. and Sun, Xiankai},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the MIT License (see `LICENSE`).

## Acknowledgments

This work was supported by the National Cancer Institute (NCI) of the National
Institutes of Health (NIH) under award R01CA309499.
