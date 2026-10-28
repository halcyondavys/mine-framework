# MINE Framework (Python)

**MINE: A Framework for Dynamic Regressor Selection**

This repository contains the Python implementation of the MINE framework, originally proposed by Moura et al. (2021) and implemented in MATLAB. MINE performs dynamic selection and combination of regressors from a homogeneous ensemble using competence-based measures. This Python port was developed as part of a doctoral thesis in Computer Engineering at PPGEc-UPE / CIn-UFPE (Recife, Brazil), where MINE serves as the main baseline for comparison with the DREAM framework.

The original MATLAB implementation is available at: https://github.com/tmoura/mine

---

## Framework Overview

MINE operates in two main computational phases:

**Phase 2 — Homogeneous Ensemble Generation**

A pool of 100 homogeneous regressors (Bagging) is trained per dataset and stored for reuse in the dynamic selection phase. The ensemble is generated using the same cross-validation protocol adopted by DREAM (20 executions × 10-fold), ensuring a fair experimental comparison. The resulting trained models are saved as `.npz` files organized by dataset and execution.

**Phases 3–4 — Dynamic Regressor Selection and Combination**

For each test instance, MINE estimates the competence of each ensemble member using eight local competence measures computed over the neighborhood of the instance (KNN-based). A Genetic Algorithm (GA) is then applied — per fold, per ensemble size — to optimize the weights assigned to these measures over the training set. Three dynamic combination strategies are evaluated:

- **MINE-S (simple_DS):** selects the single most competent regressor for each instance using the measure with highest optimized weight.
- **MINE-W (simple_DW):** weighted combination of all regressors, with weights proportional to their optimized competence scores.
- **MINE-WS (simple_DWS):** selects a subset of regressors above a competence threshold and combines them with optimized weights.

Additionally, MINE evaluates six reference techniques: literature DS (measure 18), literature DW, literature DWS, individual best model (IND), ensemble mean (MEAN), ensemble median (MEDIAN), and oracle lower bound (ORACLE) — matching the original MATLAB protocol.

---

## Requirements

- Python >= 3.9

Main dependencies:

```
numpy
pandas
scikit-learn
scipy
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Datasets

MINE uses the same 30 benchmark regression datasets adopted in the DREAM framework. The datasets are not distributed with this repository. Place the downloaded files in `data/`. See [`data/README.md`](data/README.md) for the complete list, sources, and download instructions.

All datasets follow the same format: CSV with no header, comma-separated, with the **target variable in the first column**.

---

## How to Run

### Option 1 — Automatic (recommended)

Runs the complete pipeline in a single command:

```bash
python run_all.py
```

### Option 2 — Manual (step by step)

**Phase 2 — Homogeneous Ensemble Generation**
```bash
python phase2_ensemble/run_phase2.py
```

**Phases 3–4 — Dynamic Regressor Selection**
```bash
python phase3_4_dynamic/run_phase3_4.py
python phase3_4_dynamic/results_to_csv.py
```

---

## Repository Structure

```
mine-framework/
├── phase2_ensemble/
│   ├── BaggingHomogeneous100.py       # Bagging ensemble training
│   ├── runGenerateEnsemble100.py      # Ensemble generation per dataset
│   └── run_phase2.py                  # Phase 2 orchestrator
│
├── phase3_4_dynamic/
│   ├── mine_core.py                   # Core algorithms: DS, DW, DWS, GA, KNN
│   ├── run_mine.py                    # run_MINE function (GA + strategies)
│   ├── run_phase3_4.py                # Phases 3–4 orchestrator
│   ├── results_to_csv.py              # Results extraction to CSV
│   └── test_mine_core.py              # Unit tests for mine_core
│
├── main/
│   ├── configs.py                     # Global configurations
│   ├── elm.py                         # Extreme Learning Machine
│   ├── random_layer.py                # ELM random layer
│   ├── best_params.py                 # Best hyperparameters per dataset
│   ├── testDataTesting.py             # Model inference utilities
│   └── trainDataTraining.py           # Base model training functions
│
├── data/
│   └── README.md                      # Dataset list and sources
│
├── run_all.py                         # Full pipeline orchestrator
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## Relationship to DREAM

MINE is the primary baseline used in the experimental evaluation of the DREAM framework. The key methodological differences are:

| Aspect | MINE | DREAM |
|---|---|---|
| Ensemble type | Homogeneous (Bagging) | Heterogeneous (diverse model types) |
| Pool size | Fixed (10 or 100 models) | Adaptive (3–9 models, CSS-based) |
| Phase 1 selection | None (all models used) | Multicriteria global selection (CSS + Bayesian optimization) |
| Competence measures | 8 measures, GA-optimized weights | IDW-KNN local error |
| Combination strategies | MINE-S, MINE-W, MINE-WS | DREAM-DS, DREAM-DW, DREAM-DWS |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this implementation in your research, please consider citing:

**Original MINE paper**

Thiago José Marques Moura, Adriano Lorena Inácio de Oliveira, Rodrigo Mello, Bruno José Torres Fernandes. *MINE: A Framework for Dynamic Regressor Selection*. Expert Systems with Applications, vol. 176, p. 114917, 2021.

```bibtex
@article{moura2021mine,
  author  = {Moura, Thiago José Marques and
             de Oliveira, Adriano Lorena Inácio and
             Mello, Rodrigo and
             Fernandes, Bruno José Torres},
  title   = {MINE: A Framework for Dynamic Regressor Selection},
  journal = {Expert Systems with Applications},
  volume  = {176},
  pages   = {114917},
  year    = {2021},
  doi     = {10.1016/j.eswa.2021.114917}
}
```

**Python port — Doctoral Thesis**

Halcyon Davys P. de Carvalho. *DREAM: A Multi-Criteria Global Selection and Local Competence Framework for Dynamic Combination in Heterogeneous Regression Ensembles*. Doctoral thesis, PPGEc-UPE / CIn-UFPE, Recife, Brazil, 2025.

```bibtex
@phdthesis{carvalho2025dream,
  author  = {Halcyon Davys Pimentel de Carvalho},
  title   = {DREAM: A Multi-Criteria Global Selection and Local Competence
             Framework for Dynamic Combination in Heterogeneous Regression Ensembles},
  school  = {Universidade de Pernambuco / Universidade Federal de Pernambuco},
  year    = {2025},
  address = {Recife, Brazil},
  url     = {https://github.com/halcyondavys/mine-framework}
}
```

---

**Halcyon Davys Pimentel de Carvalho**
ORCID: [0000-0001-8933-5912](https://orcid.org/0000-0001-8933-5912)
PPGEc-UPE / CIn-UFPE — Recife, Brazil
