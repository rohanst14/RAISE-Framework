# RAISE Framework

Research implementation combining continuous stirred-tank reactor (CSTR) simulation, gated recurrent units (GRU), sparse identification of nonlinear dynamics (SINDy), and probabilistic process-risk analysis.

## Research workflow

```mermaid
flowchart LR
  A[CSTR simulation] --> B[Training data]
  B --> C[GRU training]
  C --> D[SINDy inference]
  D --> E[Adaptive validation]
  D --> F[Probabilistic risk analysis]
```

## Repository structure

| Script | Purpose |
| --- | --- |
| [cstr_data.py](scripts/cstr_data.py) | Numerical simulation of CSTR dynamics |
| [train_gru.py](scripts/train_gru.py) | Multihorizon GRU training for SINDy coefficients |
| [gru_sindy_inference.py](scripts/gru_sindy_inference.py) | Trained model inference |
| [adaptive_validation.py](scripts/adaptive_validation.py) | Validation using GRU and Keras models |
| [raise_analysis.py](scripts/raise_analysis.py) | Fault probability, severity and risk analysis |

## Installation

```bash
git clone https://github.com/rohanst14/RAISE-Framework.git
cd RAISE-Framework
python -m venv .venv
# Activate the virtual environment for your operating system.
python -m pip install -r requirements.txt
```

## Reproducibility

Run scripts from the repository root. Start with `python scripts/cstr_data.py` to simulate reactor dynamics. Before training, inference or analysis, obtain the required datasets, checkpoints and fitted parameters listed in the [reproducibility guide](docs/reproducibility.md). The published scripts do not provide a complete end-to-end dataset-generation pipeline.

Dependencies are inferred from imports; versions and runtime compatibility are not yet validated. This research implementation is not a validated operational safety system.

## Citation

Reference the repository URL and exact commit used. See [CITATION.cff](CITATION.cff). No software license or publication DOI is asserted.
