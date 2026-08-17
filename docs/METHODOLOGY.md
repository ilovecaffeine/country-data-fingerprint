#  Methodology & System Architecture

This document provides a comprehensive, mathematically rigorous overview of the data pipeline, feature engineering, experimental design, and validation framework behind the **Country Data Fingerprint** project.

---

##  1. Research Questions & Core Logic

The project models country socio-economic states as **20-dimensional time-series vectors (Country Fingerprints)** across **153 countries** from **2010 to 2024**. It addresses four primary research questions:

| ID | Research Question | Corresponding Experiment |
| :--- | :--- | :--- |
| **RQ1** | How similar are countries to one another within the same year based on their multidimensional fingerprints? | **Exp 1: Same-Year Macro Similarity** |
| **RQ2** | Does a 20D fingerprint contain sufficient information to identify its country of origin? | **Exp 2: Country Classification** |
| **RQ3** | Can a country's future fingerprint ($t+1$) be forecasted from its historical fingerprint ($t$)? | **Exp 3: Multi-Output Forecasting** |
| **RQ4** | Does a forecasted future fingerprint preserve enough structural identity to remain recognizable? | **Closed-Loop Hybrid Pipeline** |

```text
                               COUNTRY DATA (153 Countries × 15 Years)
                                                  │
                                                  ▼
                                      20D COUNTRY FINGERPRINT
                                                  │
                ┌─────────────────────────────────┼─────────────────────────────────┐
                ▼                                 ▼                                 ▼
      EXPERIMENT 1                      EXPERIMENT 2                      EXPERIMENT 3
   Same-Year Similarity                Classification                     Forecasting
(Euclidean / Cosine / HEB)           (153 Country Classes)            (20D Vector at t+1)
                │                                 │                                 │
                ▼                                 ▼                                 ▼
    Cross-sectional Proximity               Country Identity               Forecasted Fingerprint
                                                  │                                 │
                                                  └────────────────┬────────────────┘
                                                                   ▼
                                                         CLOSED-LOOP PIPELINE
                                                     (Identity Preservation Test)
```

---

##  2. System Architecture & End-to-End Pipeline

The system is engineered as an automated, reproducible MLOps pipeline (`src/script/run_pipeline.py`) structured into four processing stages:

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

##  3. Data Ingestion, Imputation & Integrity Validation

### 3.1 Panel Assembly & Imputation
Raw economic indicators from the World Bank and Our World in Data (OWID) contain missing observations across country-year pairs. To assemble a **balanced panel dataset** without introducing selection bias (e.g., dropping developing nations), missing values are imputed sequentially:
1. **Linear Interpolation**: Applied to interior gaps within an observed time series.
2. **Forward Fill (`ffill`)**: Applied to trailing gaps following the last observed year.
3. **Backward Fill (`bfill`)**: Applied to leading gaps prior to the first observed year.

### 3.2 Quality Control Rule
To prevent over-imputation on sparse series, a strict threshold is enforced:
$$\text{max missing allowed} = 7 \text{ years (out of 15)}$$

A country is retained if and only if it possesses at least $15 - 7 = 8$ years of empirical observations. Consequently, every retained country contains at least:
$$\frac{8}{15} = 53.3\% \text{ empirical data}$$
while the maximum allowable imputed proportion is $46.7\%$.

### 3.3 Data Integrity Validation
Before model execution, `validate_panel_data_integrity()` verifies that:
* Exactly $0$ missing/`NaN` values remain.
* All $153$ countries contain complete 15-year sequences (2010–2024).
* Each country-year pair appears exactly once.

---

## ️ 4. Feature Engineering & Preprocessing

The 20 macroeconomic indicators vary significantly in scale and distribution skewness. Mathematical transformations are applied to normalize feature spaces.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 20 FEATURE INDICATORS                                  │
└──────┬─────────────────────┬─────────────────────┬──────────────────────┬──────────────┘
       │                     │                     │                      │
       ▼                     ▼                     ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐       ┌──────────────┐
│  Log (ln x)  │      │    Log1p     │      │ Yeo-Johnson  │       │ Reflect-Log  │
│              │      │  ln(1 + x)   │      │ (Deflation)  │       │ ln(max+1-x)  │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘       └──────┬───────┘
       │                     │                     │                      │
       └─────────────────────┼─────────────────────┴──────────────────────┘
                             │
                             ▼
               ┌──────────────────────────┐
               │ Z-Score Standardization  │
               │ z = (x - μ_train) / σ    │
               └──────────────────────────┘
```

### 4.1 Feature Transformation Rules

#### 1. Log Transformation
For non-negative, heavily right-skewed level variables:
$$y = \ln(x)$$
* **Applied to**: `gdp_per_capita_ppp`, `population_total`.

#### 2. Log1p Transformation
For non-negative percentage/rate variables that may contain zero values:
$$y = \ln(1 + x)$$
* **Applied to**: `co2_emissions_per_capita`, `unemployment_rate_pct`, `trade_pct_gdp`, `industry_pct_gdp`, `agriculture_pct_gdp`, `fertility_rate_births_per_woman`, `under_5_mortality_rate_per_1000`, `general_government_final_consumption_expenditure_pct_gdp`.

#### 3. Yeo-Johnson Transformation
For indicators containing zero or negative values (e.g., deflation):
$$\psi(\lambda, x) = \begin{cases} \ln(x + 1) & \text{if } \lambda = 0, x \ge 0 \\ \frac{(x + 1)^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0, x \ge 0 \end{cases}$$
* **Applied to**: `inflation_gdp_deflator_annual_pct`.

#### 4. Reflect + Log Transformation
For strongly left-skewed, bounded percentage variables ($0 \le x \le 100$):
$$y = \ln\Big(\big(\max(x_{\text{train}}) + 1\big) - x\Big)$$
* **Applied to**: `access_to_electricity_pct`.

### 4.2 Standardization & Data Leakage Prevention
Following feature transformations, $Z$-score standardization is applied:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

> ️ **Data Leakage Prevention Guarantee**: All transformation parameters ($\mu_{\text{train}}$, $\sigma_{\text{train}}$, $\max(x_{\text{train}})$, and Yeo-Johnson $\lambda$) are fitted **strictly on the Training set** and subsequently applied to Validation and Test sets.

---

##  5. Experimental Setup & Results

### 5.1 Experiment 1: Cross-Sectional Macro-Similarity

#### Methodology
Measures pairwise country proximity **within the same observation year** across the 20D feature space. For two country vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^{20}$ in year $t$:

* **Euclidean Distance**:
  $$d E(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{20} (x i - y i)^2}$$

- **Cosine Similarity**:

$$ \text{Similarity}(x, y) = \frac{x \cdot y}{\|x\|_2 \|y\|_2} = \frac{\sum_{i=1}^{20} x_i y_i}{\sqrt{\sum_{i=1}^{20} x_i^2} \sqrt{\sum_{i=1}^{20} y_i^2}} $$

#### Outputs & Network Topology
* Produces $15$ annual distance matrices (2010–2024) for both metrics.
* Generates interactive **D3.js Hierarchical Edge Bundling (HEB)** circular graphs grouping countries by continent on the outer ring with undirected B-spline connection curves.

---

### 5.2 Experiment 2: Country Fingerprint Classification

#### Objective
Predict country identity ($y \in \{1, \dots, 153\}$) given a 20D feature vector $\mathbf{x}_t$.

#### Chronological Data Split
* **Train Set**: $2010 \le t \le 2018$ ($1,377$ samples)
* **Validation Set**: $2019 \le t \le 2020$ ($306$ samples)
* **Test Set**: $2021 \le t \le 2024$ ($612$ samples)

#### Benchmark & Out-of-Sample Results
Evaluated across 9 classifiers. **Extra Trees Classifier** achieved optimal performance:

| Model | Val Accuracy | Val Top-5 Acc | Test Accuracy | Test Top-5 Acc | Test Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extra Trees**  | **100.00%** | **100.00%** | **97.55%** | **100.00%** | **97.41%** |
| Random Forest | 99.35% | 100.00% | 94.12% | 100.00% | 93.51% |
| CatBoost | 97.39% | 100.00% | 94.77% | 99.84% | 94.14% |
| SVC (Linear) | 96.41% | 99.02% | — | — | — |
| MLP Neural Net | 95.42% | 99.67% | — | — | — |
| Logistic Regression | 92.16% | 97.71% | — | — | — |
| XGBoost | 83.99% | 93.79% | — | — | — |
| LightGBM | 67.32% | 87.25% | — | — | — |

#### Error Analysis
All $15$ test misclassifications ($2.45\%$ error rate) occurred exclusively between **geographically adjacent and structurally identical neighbor pairs** (e.g., Lithuania $\rightarrow$ Estonia, Serbia $\rightarrow$ Bulgaria, Uganda $\rightarrow$ Tanzania).

---

### 5.3 Experiment 3: Multi-Output Time-Series Forecasting

#### Objective
Predict the complete 20D feature vector at year $t+1$ ($\hat{\mathbf{y}}_{t+1} \in \mathbb{R}^{20}$) given the 20D feature vector at year $t$ ($\mathbf{x}_t \in \mathbb{R}^{20}$).

#### Sequence Alignment & Target Split
Sequences are paired per country ($t \to t+1$). The split is based on the **target year ($t+1$)**:
* **Train Set**: Input $t \le 2017 \implies$ Target $t+1 \in [2011, 2018]$
* **Validation Set**: Input $t \in [2018, 2019] \implies$ Target $t+1 \in [2019, 2020]$
* **Test Set**: Input $t \in [2020, 2023] \implies$ Target $t+1 \in [2021, 2024]$

#### Evaluation Metrics
Metrics are averaged across all 20 predicted dimensions ($K=20$):

$$\text{Mean } R^2 = \frac{1}{K} \sum_{k=1}^{K} \left(1 - \frac{\sum_{i=1}^{n} (y_{i,k} - \hat{y}_{i,k})^2}{\sum_{i=1}^{n} (y_{i,k} - \bar{y}_k)^2}\right)$$

$$\text{Mean MAE} = \frac{1}{n K} \sum_{i=1}^{n} \sum_{k=1}^{K} |y_{i,k} - \hat{y}_{i,k}|, \qquad \text{Mean RMSE} = \sqrt{\frac{1}{n K} \sum_{i=1}^{n} \sum_{k=1}^{K} (y_{i,k} - \hat{y}_{i,k})^2}$$

#### Benchmark Results

| Model | Val Mean $R^2$ | Val Mean MAE | Val Mean RMSE | Test Mean $R^2$ | Test Mean MAE | Test Mean RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression**  | **85.09%** | **0.1712** | **0.6274** | **83.36%** | **0.1727** | **0.6912** |
| MultiOutput + Random Forest | 84.40% | 0.1671 | 0.6794 | — | — | — |
| MultiOutput + Extra Trees | 84.34% | 0.1669 | 0.6768 | — | — | — |
| XGBoost | 83.70% | 0.1700 | 0.6874 | — | — | — |
| LightGBM | 83.02% | 0.1740 | 0.6924 | — | — | — |
| MLP Neural Net | 82.69% | 0.2091 | 0.7015 | — | — | — |

>  **Econometric Insight**: Linear Regression outperforms complex non-linear tree models in forecasting because log-transformed macroeconomic level variables follow autoregressive, log-linear growth trends. Linear models naturally extrapolate trends out-of-sample, whereas decision trees cannot extrapolate beyond training bounding boxes.

---

### 5.4 Closed-Loop Hybrid Pipeline

#### Methodology
Tests whether forecasted future 20D vectors ($\hat{\mathbf{y}}_{t+1}$) retain enough structural identity to be correctly classified by the Experiment 2 model:

$$
\mathbf{x}_t \Rightarrow \begin{bmatrix} \text{Exp 3 Forecast (Linear Reg)} \end{bmatrix} \Rightarrow \hat{\mathbf{y}}_{t+1} \Rightarrow \begin{bmatrix} \text{Exp 2 Classify (Extra Trees)} \end{bmatrix} \Rightarrow \hat{\text{country-code}}
$$

#### Results & Performance Comparison

| Input Vector to Extra Trees | Test Accuracy | Test Top-5 Accuracy | Test Weighted F1 |
| :--- | :---: | :---: | :---: |
| **Actual Test Features (2021–2024)** | 97.55% | 100.00% | 97.41% |
| **Forecasted Test Features ($\hat{\mathbf{y}}_{2021-2024}$)**  | **97.71%** | **100.00%** | **97.63%** |

#### Scientific Findings
1. **Denoising Effect**: Linear Regression acts as a noise filter on short-term post-COVID fluctuations, producing an idealized trend vector that Extra Trees classifies with slightly higher accuracy (+0.16%).
2. **Identity Preservation**: The forecasted 20D vector preserves the unique structural identity of the country over multi-year horizons.

---

## ️ 6. Methodological Rigor & Robustness Checks

1. **Imputation Impact**: The inclusion of imputed observations is documented. Imputation introduces artificial smoothing along time-series paths, which contributes to high autoregressive predictability ($R^2 > 83\%$).
2. **Leakage Control**: All scaler and transformation parameters are fitted exclusively on training sets ($t \le 2018$ for Exp 2; $t \le 2017$ for Exp 3).
3. **No Self-Loops**: Distance matrices and network graphs explicitly filter out self-comparisons ($d(\mathbf{x}_i, \mathbf{x}_i) = 0$).

---

## ️ 7. MLOps Execution & Reproducibility

The pipeline is fully executable via `src/script/run_pipeline.py`.

```bash
# Execute the complete end-to-end pipeline
python -m src.script.run_pipeline --step all --use_imputed True --drop_missing True

# Execute specific pipeline stages
python -m src.script.run_pipeline --step data         # Ingestion, imputation & panel creation
python -m src.script.run_pipeline --step exp1         # Distance matrices & D3.js HEB plots
python -m src.script.run_pipeline --step exp2         # Country classification & error analysis
python -m src.script.run_pipeline --step exp3         # Multi-output forecasting & stability maps
python -m src.script.run_pipeline --step closed_loop  # Closed-loop hybrid pipeline evaluation
```

---

##  8. Project Artifacts & File Structure

```text
results/
├── experiment_1/
│   ├── distance_matrices/              # Annual Euclidean distance CSVs (2010–2024)
│   ├── cosine_matrices/                # Annual Cosine similarity CSVs (2010–2024)
│   ├── distance_heb_plots/             # Interactive D3.js Euclidean HEB HTMLs
│   └── cosine_heb_plots/               # Interactive D3.js Cosine HEB HTMLs
│
├── experiment_2/
│   ├── auto_benchmark_validation_results.csv
│   ├── extra_trees_test_results.csv
│   ├── extra_trees_feature_importances.csv
│   └── extra_trees_misclassification_summary.csv
│
├── experiment_3/
│   ├── forecast_auto_benchmark_val_results.csv
│   ├── forecast_auto_benchmark_val_feature_details.csv
│   ├── linear_regression_test_results.csv
│   ├── linear_regression_test_feature_details.csv
│   ├── linear_regression_country_stability.csv
│   ├── global_yearly_boxplot_cosine_similarity.png
│   ├── global_yearly_boxplot_rmse.png
│   └── country_stability_interactive.html
│
└── experiment_4/
    ├── pipeline_forecast_then_classify_results.csv
    └── misclassification_comparison_merged.csv
```
