\# 🔬 Methodology \& System Architecture



This document provides a comprehensive technical overview of the methodology, data pipeline, feature engineering, and experimental design behind the \*\*Country Data Fingerprint\*\* project.



\---



\## 📐 1. System Architecture \& End-to-End Pipeline



The project is architected as an automated, reproducible MLOps pipeline (`src/script/run\_pipeline.py`) that handles data ingestion, time-series imputation, distance modeling, multi-class classification, multi-output forecasting, and closed-loop system validation.
+-----------------------------------------------------------------------------------+

| 1. DATA PIPELINE |

| World Bank \& OWID Raw Data (194 Countries, 2010-2024) |

| │ |

| ▼ |

| Time-Series Imputation (Linear + ffill + bfill, max missing <= 7 yrs) |

| │ |

| ▼ |

| Balanced Panel Data Creation (153 Countries x 15 Years x 20 Features) |

+-----------------------------------------┬-----------------------------------------+

│

┌─────────────────────────────────────┼────────────────────────────────────┐

│ │ │

▼ ▼ ▼

+-----------------------+ +---------------------------+ +------------------+

| EXPERIMENT 1 | | EXPERIMENT 2 | | EXPERIMENT 3 |

| Macro-Similarity \& | | Country Classification | | Multi-Output |

| Distance Analysis | | (Fingerprint Identity) | | Forecasting |

| | | | | (1-Year Ahead) |

| • Euclidean \& Cosine | | • Log + StandardScaler | | • Shift(-1) Lag |

| • Wide Summary Tables | | • Time-based Split | | • Multi-target |

| • D3.js Interactive | | • ExtraTrees (Acc 97.55%) | | LR (R² 83.36%) |

| Hierarchical Edge | | • Feature Importances | | • 4-Quadrant |

| Bundling (HEB) | | • Error Analysis | | Stability Map |

+-----------------------+ +-------------┬-------------+ +--------┬---------+

│ │

└────────────┬─────────────┘

│

▼

+------------------------------------+

| CLOSED-LOOP PIPELINE |

| Exp 3 Forecast ──> Exp 2 Classify |

| |

| Evaluates if forecasted 20D future |

| fingerprints preserve identity |

| (Test Acc: 97.71%, Top-5: 100%) |

+------------------------------------+
## 🧹 2. Data Ingestion, Imputation \& Integrity Checks



\### 2.1 Raw Data \& Panel Assembly

\* \*\*Scope\*\*: 20 macro socio-economic indicators across \*\*153 countries\*\* from \*\*2010 to 2024\*\* (15 annual observations per country).

\* \*\*Missing Data Strategy\*\*:

&#x20; \* Raw World Bank data contains gaps. To ensure a \*\*Balanced Panel Data\*\* without removing developing nations (preventing Selection Bias), an imputation strategy is applied:

&#x20;   1. \*\*Linear Interpolation\*\* (`method='linear'`) for interior gaps.

&#x20;   2. \*\*Forward Fill (`ffill`) \& Backward Fill (`bfill`)\*\* for boundary gaps.

&#x20; \* \*\*Quality Control Rule (`max\_missing\_allowed = 7`)\*\*: Any country with $>7$ missing years out of 15 ($>46.7\\%$) is completely removed. This guarantees that \*\*at least $53.3\\%$ of the time-series data per country consists of genuine empirical observations\*\*.



\### 2.2 Data Integrity Validation

Before passing data to experiments, `validate\_panel\_data\_integrity()` verifies:

\* 0 missing/NaN values remaining in the panel.

\* Exactly 15 consecutive years (2010–2024) present for all 153 countries.



\---



\## ⚙️ 3. Feature Engineering \& Preprocessing



To handle extreme skewness and different units across global macro indicators, non-linear transformations are applied:



1\. \*\*Log Transformation (`np.log`)\*\*: Applied to highly skewed level variables (`gdp\_per\_capita\_ppp`, `population\_total`).

2\. \*\*Log1p Transformation (`np.log1p`)\*\*: Applied to percentage indicators with small/zero values (`co2\_emissions\_per\_capita`, `unemployment\_rate\_pct`, `trade\_pct\_gdp`, etc.).

3\. \*\*Yeo-Johnson Transformation\*\*: Applied to `inflation\_gdp\_deflator\_annual\_pct` (supports negative values during deflation).

4\. \*\*Reflect + Log\*\*: Applied to left-skewed bounded indicators (`access\_to\_electricity\_pct`).

5\. \*\*Z-Score Normalization (`StandardScaler`)\*\*: Normalizes all 20 features to $\\mu=0, \\sigma=1$.

&#x20;  > ⚠️ \*\*Data Leakage Prevention\*\*: `StandardScaler` and `PowerTransformer` are \*\*fitted strictly on the Training Set\*\* and then used to transform Validation and Test sets.



\---



\## 🧪 4. Experimental Design \& Methodology



\### 4.1 Temporal Splitting Strategy

To respect the temporal arrow of time and prevent Look-ahead Bias:

\* \*\*Train Set\*\*: Years $2010 – 2017/2018$ (Historical training)

\* \*\*Validation Set\*\*: Years $2018 – 2019/2020$ (Model selection \& hyperparameter benchmarking)

\* \*\*Test Set\*\*: Years $2020 – 2023 / 2021 – 2024$ (Out-of-sample future evaluation during COVID-19 recovery shocks)



\---



\### 4.2 Experiment 1: Macro-Similarity \& Network Topology

\* \*\*Objective\*\*: Measure multi-dimensional socio-economic proximity between countries.

\* \*\*Metrics\*\*: Euclidean Distance ($d\_E$) and Cosine Similarity ($S\_C$).

\* \*\*Visualization\*\*: Interactive \*\*D3.js Hierarchical Edge Bundling (HEB)\*\* circular graphs grouping countries by continent on the outer ring, connected by undirected B-spline curves.



\---



\### 4.3 Experiment 2: Country Fingerprint Classification

\* \*\*Objective\*\*: Multi-class classification predicting country identity (`country\_code\_3`, 153 classes) from a 20D socio-economic vector.

\* \*\*Benchmark\*\*: Evaluated 9 classifiers (Logistic Regression, Random Forest, Extra Trees, SVC, MLP, XGBoost, LightGBM, CatBoost).

\* \*\*Key Finding\*\*: \*\*Extra Trees Classifier\*\* achieved top performance:

&#x20; \* \*\*Test Accuracy\*\*: \*\*97.55%\*\*

&#x20; \* \*\*Test Top-5 Accuracy\*\*: \*\*100.00%\*\*

&#x20; \* \*\*Test F1 Weighted\*\*: \*\*97.41%\*\*

\* \*\*Error Analysis\*\*: Misclassifications (2.45% error rate) occur exclusively between \*\*geographically adjacent and structurally identical neighbor pairs\*\* (e.g., Lithuania $\\rightarrow$ Estonia, Uganda $\\rightarrow$ Tanzania).



\---



\### 4.4 Experiment 3: Multi-Output Time-Series Forecasting

\* \*\*Objective\*\*: Predict all 20 socio-economic features at year $t+1$ given all 20 features at year $t$ across all panel countries.

\* \*\*Benchmark\*\*: Evaluated 9 multi-output regressors (Linear Regression, Native RF/ET, MultiOutput RF/ET, XGBoost, LightGBM, CatBoost MultiRMSE, MLP).

\* \*\*Key Finding\*\*: \*\*Linear Regression\*\* won the benchmark:

&#x20; \* \*\*Test Mean $R^2$\*\*: \*\*83.36%\*\* ($R^2 > 99\\%$ for structural indicators like Population, GDP, Education, CO2).

&#x20; \* \*\*Test Mean MAE\*\*: \*\*0.1727\*\*

&#x20; \* \*\*Test Mean RMSE\*\*: \*\*0.6912\*\*

\* \*\*Econometric Justification\*\*: Macro indicators follow log-linear autoregressive trends. Linear Regression extrapolates linear growth curves out-of-sample much better than step-function Decision Trees.

\* \*\*Stability Evaluation\*\*: Evaluated country forecast stability over time (2021–2024) using a \*\*4-Quadrant Interactive Plotly Bubble Chart\*\* ($\\text{RMSE}\_{\\text{Mean}}$ vs $\\text{RMSE}\_{\\text{Std}}$).



\---



\### 4.5 Closed-Loop Hybrid Pipeline (Exp 3 Forecast ──> Exp 2 Classify)

\* \*\*Objective\*\*: End-to-end system validation testing if predicted future 20D fingerprints retain country identity.

\* \*\*Workflow\*\*:

&#x20; 1. `LinearRegression` (Exp 3) forecasts 20D future features $\\hat{Y}\_{t+1}$ for 2021–2024.

&#x20; 2. Predicted $\\hat{Y}\_{t+1}$ vectors are fed into pre-trained `ExtraTreesClassifier` (Exp 2) to re-identify the country.

\* \*\*Results\*\*:

&#x20; \* \*\*Pipeline Test Accuracy\*\*: \*\*97.71%\*\* (+0.16% over actual raw test features)

&#x20; \* \*\*Pipeline Top-5 Accuracy\*\*: \*\*100.00%\*\*

\* \*\*Scientific Insight\*\*: Linear Regression acts as a \*\*denoising filter\*\*, smoothing short-term post-COVID noise while preserving the core 20D structural fingerprint.



\---



\## 🛠️ 5. MLOps \& Reproducibility



The entire workflow is fully automated via `run\_pipeline.py`. Anyone can reproduce all results, CSV reports, and high-resolution figures (300 DPI) using CLI commands:



```bash

\# Run complete end-to-end pipeline

python -m src.script.run\_pipeline --step all --use\_imputed True --drop\_missing True



\# Run individual modular steps

python -m src.script.run\_pipeline --step data         # Ingestion \& Panel Data

python -m src.script.run\_pipeline --step exp1         # Distance \& HEB Plots

python -m src.script.run\_pipeline --step exp2         # Classification

python -m src.script.run\_pipeline --step exp3         # Forecasting

python -m src.script.run\_pipeline --step closed\_loop  # Closed-Loop Integration

