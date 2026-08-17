# Country Data Fingerprint
### Multidimensional Macroeconomic Proximity, Country Classification & Multi-Output Time-Series Forecasting (2010–2024)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/status-complete--reproducible-brightgreen.svg)](docs/GETTING_STARTED.md)

---

## Executive Summary

The **Country Data Fingerprint** project models the macroeconomic and socio-economic state of **153 countries** across **15 years ($2010–2024$)** as 20-dimensional time-series vectors (**Country Fingerprints**). 

By leveraging balanced panel data from the World Bank and Our World in Data ($2,295$ total observations), this repository implements an end-to-end, reproducible MLOps framework that addresses four core research questions:
1. **Cross-Sectional Similarity (Exp 1)**: How similar are countries within the same year across 20D feature space?
2. **Country Identity (Exp 2)**: Does a 20D fingerprint contain enough structural information to uniquely identify its country of origin? (**97.55% Test Accuracy, 100% Top-5 Accuracy**)
3. **Temporal Forecastability (Exp 3)**: Can a country's future 20D fingerprint ($t+1$) be predicted from its current fingerprint ($t$)? (**83.36% Test Mean $R^2$**)
4. **Identity Preservation (Closed-Loop Pipeline)**: Does a forecasted future fingerprint preserve national structural identity? (**97.71% Test Accuracy, 100% Top-5 Accuracy**)

---

## Key Empirical Benchmark Results

All evaluations were conducted out-of-sample on the **Test Set ($2021–2024$)**, representing post-COVID-19 recovery shocks across $612$ test observations:

| Experiment | Task | Winning Model | Test $R^2$ / Accuracy | Test Top-5 Acc | Key Research Takeaway |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Exp 1** | Same-Year Similarity | Distance Matrices | N/A | N/A | D3.js Hierarchical Edge Bundling graphs successfully map global macro-topology. |
| **Exp 2** | Country Classification | **Extra Trees** 🏆 | **97.55% Acc** | **100.00%** | Structural scale & urbanization define unique country identity signatures. |
| **Exp 3** | Multi-Output Forecasting | **Linear Regression** 🏆 | **83.36% $R^2$** | N/A | Log-linear macroeconomic level trends follow stable, predictable trajectories. |
| **Closed Loop** | Forecast $\rightarrow$ Classify | **LinearReg $\rightarrow$ ExtraTrees** 🏆 | **97.71% Acc** | **100.00%** | Forecasted future fingerprints preserve national identity with zero information decay. |

---

## System Architecture & Pipeline

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   1. DATA PIPELINE                                     │
│  World Bank & OWID Raw Data (194 Countries × 2010–2024)                                │
│    ├── Time-Series Imputation: Linear Interpolation + Forward Fill + Backward Fill     │
│    └── Quality Control: Drop countries with max_missing_allowed > 7 years              │
│    └── Output: Balanced Panel Data (153 Countries × 15 Years × 20 Features = 2,295 obs)│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
┌───────────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────────────┐
│     EXPERIMENT 1      │   │         EXPERIMENT 2          │   │         EXPERIMENT 3          │
│ Cross-sectional       │   │ Country Identity              │   │ Multi-Output                  │
│ Similarity            │   │ Classification                │   │ Forecasting                   │
│                       │   │                               │   │                               │
│ • Distance Matrices   │   │ • Target: country_code_3      │   │ • Target: 20D Vector at t+1   │
│   (Euclidean & Cosine)│   │ • Train: ≤2018 | Val: 2019–20 │   │ • Train: ≤2017 | Val: 2018–19 │
│ • Interactive D3.js   │   │   Test: 2021–2024             │   │   Test: 2020–2023 (Target t+1)│
│   Edge Bundling (HEB) │   │ • Best: Extra Trees (97.55%)  │   │ • Best: Linear Reg (R² 83.36%)│
└───────────────────────┘   └───────────────┬───────────────┘   └───────────────┬───────────────┘
                                            │                                   │
                                            └─────────────────┬─────────────────┘
                                                              ▼
                                            ┌───────────────────────────────────┐
                                            │       CLOSED-LOOP PIPELINE        │
                                            │   Forecasted 20D Vector (Exp 3)   │
                                            │                 │                 │
                                            │                 ▼                 │
                                            │   Extra Trees Classifier (Exp 2)  │
                                            │                 │                 │
                                            │                 ▼                 │
                                            │   Test Accuracy: 97.71%           │
                                            │   Top-5 Accuracy: 100.00%         │
                                            └───────────────────────────────────┘
```

---

## Quick Start & Reproducibility

### 1. Installation
Clone the repository and install dependencies in Python 3.10+:

```bash
git clone https://github.com/ilovecaffeine/country-data-fingerprint.git
cd country-data-fingerprint

# Create and activate environment
conda create -n fingerprint python=3.10 -y
conda activate fingerprint

# Install requirements
pip install -r requirements.txt
```

### 2. Execution (One-Command Master Pipeline)
Run the entire end-to-end pipeline from data ingestion to model benchmarking, HTML visualization, and CSV report exports:

```bash
# Execute complete pipeline from A to Z
python -m src.script.run_pipeline --step all --use_imputed True --drop_missing True
```

### 3. Run Individual Pipeline Stages
```bash
python -m src.script.run_pipeline --step data         # Ingestion, imputation & panel creation
python -m src.script.run_pipeline --step exp1         # Distance matrices & D3.js HEB plots
python -m src.script.run_pipeline --step exp2         # Country classification & error analysis
python -m src.script.run_pipeline --step exp3         # Multi-output forecasting & stability maps
python -m src.script.run_pipeline --step closed_loop  # Closed-loop hybrid pipeline evaluation
```

---

## Repository Directory Structure

```text
country-data-fingerprint/
├── README.md                           # Main repository documentation
├── requirements.txt                     # Python dependencies
├── config/
│   ├── paths.py                        # Centralized directory paths
│   ├── features.yaml                   # 20 feature definitions & transformation rules
│   └── sources.yaml                    # Dataset provenance & Series Code mapping
│
├── docs/                               # Comprehensive project documentation
│   ├── METHODOLOGY.md                  # System architecture, formulas & experiment design
│   ├── DATA_DICTIONARY.md              # Codebook for all 20 indicators
│   ├── RESULTS.md                      # Complete empirical tables & scientific findings
│   ├── GETTING_STARTED.md              # Installation & CLI execution guide
│   └── LIMITATIONS.md                  # Methodological caveats & future extensions
│
├── data/
│   ├── raw/                            # Original World Bank & OWID raw CSV datasets
│   └── processed/                      # Preprocessed, scaled, and split panel datasets
│
├── src/
│   ├── data/                           # Data loading, imputation & transformations
│   ├── distance_matrices/              # Euclidean & Cosine distance matrix computation
│   ├── hierarchical_edge_bundling/     # D3.js interactive HEB HTML generator
│   ├── models/                         # Classifiers, Forecasters & Closed-Loop Pipeline
│   └── script/
│       └── run_pipeline.py             # Master CLI pipeline runner
│
└── results/                            # Auto-generated CSV reports, 300 DPI PNGs, HTMLs
    ├── experiment_1/                   # Distance CSVs & Interactive HEB HTMLs
    ├── experiment_2/                   # Classification CSVs & Feature Importance PNGs
    ├── experiment_3/                   # Forecast CSVs, 4-Quadrant Plotly Bubble Chart
    └── experiment_4/                    # Closed-loop results & misclassification comparison
```

---

## Complete Documentation Index

For in-depth technical details, consult the documentation under `docs/`:

* 🔬 **[METHODOLOGY.md](docs/METHODOLOGY.md)**: Mathematical formulations, data leakage controls, imputation rules, and experiment design.
* 📖 **[DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)**: Full codebook for all 20 macro indicators, World Bank Series Codes, and units.
* 🏆 **[RESULTS.md](docs/RESULTS.md)**: Full benchmark tables, 20-feature importance breakdown, misclassification error analysis, and stability charts.
* 🚀 **[GETTING_STARTED.md](docs/GETTING_STARTED.md)**: Step-by-step installation, virtual environment setup, and CLI reference.
* ⚠️ **[LIMITATIONS.md](docs/LIMITATIONS.md)**: Discussion on imputation smoothing, post-COVID shocks, and future research directions.

---

## Data Sources & Citations

1. **World Development Indicators (WDI)** — World Bank Group  
   * **Source:** [World Bank Open Data](https://databank.worldbank.org/source/world-development-indicators)  
   * **Description:** Primary source for macroeconomic, demographic, and developmental indicators across national panels.

2. **Our World in Data (OWID)** — Global Change Data Lab  
   * **Source:** [Our World in Data GitHub Repository](https://github.com/owid/owid-datasets)  
   * **Description:** Secondary source for supplemental environmental, energy, and social structural metrics.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.