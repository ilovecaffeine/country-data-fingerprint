# src/script/run_pipeline.py
# cd country-data-fingerprint
# python -m src.script.run_pipeline

import argparse
from pathlib import Path
import sys
from config import paths
import pandas as pd

from src.data.load import build_and_populate_all_draft_data
from src.data.transform import (
    combine_all_drafts_to_panel_data,
    impute_all_ourworldindata_raw_files,
    impute_all_worldbank_raw_files,
    preprocess_and_export_all_experiment_1_years,
    preprocess_and_split_experiment_2,
    preprocess_and_split_experiment_3,
)
from src.distance_matrices.distance_matrices import (
    export_experiment_1_cosine_matrices,
    export_experiment_1_distance_matrices,
    generate_cosine_neighbors_summary,
    generate_euclidean_neighbors_summary,
)
from src.hierarchical_edge_bundling.export_heb import (
    continent_map,
    export_all_years_cosine_heb_plots,
    export_all_years_distance_heb_plots,
)
from src.models.classifier import (
    evaluate_extra_trees_on_test,
    load_experiment_2_country_classification,
    run_auto_benchmark_experiment_2,
)
from src.models.forecast import (
    evaluate_linear_regression_on_test,
    load_experiment_3,
    plot_country_stability_scatter_chart,
    plot_global_yearly_boxplot,
    run_auto_benchmark_experiment_3,
)
from src.models.forecast_then_classify import (
    plot_misclassification_comparison,
    run_forecast_then_classify_pipeline,
)

PIPELINE_HELPER_TEXT = """
================================================================================
💡 COMMAND-LINE FLAG COMBINATIONS (--use_imputed & --drop_missing):
================================================================================
1. [--use_imputed True  --drop_missing True]  (RECOMMENDED / DEFAULT)
   👉 Performs data imputation and complete cleanup.
      Executes the full pipeline end-to-end without missing values.

2. [--use_imputed <True|False>  --drop_missing False]
   👉 Exports 'draft_panel_2010_2024.csv' with missing values (NaNs) intact,
      then terminates the pipeline execution.

3. [--use_imputed False  --drop_missing True]
   👉 Generates an EMPTY 'draft_panel_2010_2024.csv' (0 rows) due to 100% missing
      entries in the raw dataset.
================================================================================
"""


def validate_panel_data_integrity(panel_file_path: Path) -> bool:
    """Validates the integrity of the draft_panel_2010_2024.csv table:
    1. File must not be empty.
    2. Must not contain any NaN / missing values.
    3. All countries must have data for all years (2010-2024).
    """
    if not panel_file_path.exists():
        print(f"❌ [ERROR] Panel Data file not found: {panel_file_path}")
        return False

    df_panel = pd.read_csv(panel_file_path)

    # 1. Check for empty dataframe
    if df_panel.empty:
        print("❌ [ERROR] Panel Data table is empty (0 countries, 0 rows)!")
        return False

    # 2. Check for Missing Values (NaNs)
    total_nans = df_panel.isna().sum().sum()
    if total_nans > 0:
        print(
            f"❌ [ERROR] Detected {total_nans} missing cells (NaN) in Panel"
            " Data!"
        )
        nan_cols = df_panel.isna().sum()
        print("Missing column details:\n", nan_cols[nan_cols > 0])
        return False

    # 3. Check full year coverage for each country
    expected_years = set(paths.YEARS)
    expected_year_count = len(expected_years)

    years_per_country = df_panel.groupby("country_code_3")["year"].apply(set)
    incomplete_countries = {
        code: expected_years - yrs
        for code, yrs in years_per_country.items()
        if yrs != expected_years
    }

    if incomplete_countries:
        print(
            "❌ [ERROR] Detected"
            f" {len(incomplete_countries)} countries with missing years!"
        )
        for code, missing_yrs in list(incomplete_countries.items())[:5]:
            print(
                f"   - Country {code}: Missing years"
                f" {sorted(list(missing_yrs))}"
            )
        return False

    num_countries = df_panel["country_code_3"].nunique()
    print(
        f"✅ DATA INTEGRITY CHECK PASSED: {num_countries} countries with 100%"
        f" clean data across all {expected_year_count} years (2010-2024), 0 NaN"
        " cells."
    )
    return True


def step_1_data_processing(
    use_imputed: bool = True, drop_countries_with_missing: bool = True
):
    print("\n=======================================================")
    print("📍 STAGE 1: LOADING AND CLEANING PANEL DATA")
    print("=======================================================")

    # 1. Impute raw data if use_imputed flag is True
    if use_imputed:
        print(
            "🔄 Performing imputation on raw data files (World Bank & OWID)..."
        )
        impute_all_worldbank_raw_files()
        impute_all_ourworldindata_raw_files()

    # 2. Load and populate 20 indicators into draft tables
    build_and_populate_all_draft_data(
        use_imputed=use_imputed, clear_existing=True
    )

    # 3. Combine into Panel Data dataframe
    combine_all_drafts_to_panel_data(
        drop_countries=False,
        drop_countries_with_missing=drop_countries_with_missing,
    )

    # 🌟 DATA INTEGRITY VALIDATION CHECK
    panel_csv_path = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"
    print("\n🔍 Validating integrity of draft_panel_2010_2024.csv...")

    if not validate_panel_data_integrity(panel_csv_path):
        print("\n⛔ [STOP PIPELINE] Panel Data did not pass validation!")
        print(f"👉 Saved Panel Data at: {panel_csv_path}")
        sys.exit(1)  # Terminate execution immediately

    # 4. Preprocess and export data for 3 Experiments (Runs only when data is 100% clean)
    print("\n🔄 Preprocessing and splitting data for 3 Experiments...")
    preprocess_and_export_all_experiment_1_years()
    preprocess_and_split_experiment_2()
    preprocess_and_split_experiment_3()


def step_2_experiment_1():
    print("\n=======================================================")
    print("📍 STAGE 2: EXPERIMENT 1 - DISTANCE MATRICES & HEB PLOTS")
    print("=======================================================")

    # 1. Calculate and export Distance Matrices
    export_experiment_1_distance_matrices()
    export_experiment_1_cosine_matrices()

    # 2. Export Neighbor Summary files (Wide format)
    generate_euclidean_neighbors_summary()
    generate_cosine_neighbors_summary()

    # 3. Export all HEB HTML plots (Euclidean & Cosine)
    export_all_years_distance_heb_plots(continent_map=continent_map)
    export_all_years_cosine_heb_plots(continent_map=continent_map)


def step_3_experiment_2():
    print("\n=======================================================")
    print("📍 STAGE 3: EXPERIMENT 2 - COUNTRY CLASSIFICATION")
    print("=======================================================")

    # 1. Load Experiment 2 data
    (
        X_train_exp2,
        y_train_exp2,
        X_val_exp2,
        y_val_exp2,
        X_test_exp2,
        y_test_exp2,
        le,
    ) = load_experiment_2_country_classification()

    # 2. Auto-Benchmark model comparison on Validation set
    run_auto_benchmark_experiment_2(
        X_train=X_train_exp2,
        y_train=y_train_exp2,
        X_val=X_val_exp2,
        y_val=y_val_exp2,
        plot_results=True,
    )

    # 3. Evaluate winning Extra Trees model on TEST set
    evaluate_extra_trees_on_test(
        X_train=X_train_exp2,
        y_train=y_train_exp2,
        X_test=X_test_exp2,
        y_test=y_test_exp2,
        label_encoder=le,
    )


def step_4_experiment_3():
    print("\n=======================================================")
    print("📍 STAGE 4: EXPERIMENT 3 - MULTI-OUTPUT FORECASTING")
    print("=======================================================")

    # 1. Load Experiment 3 data
    (
        X_train_exp3,
        y_train_exp3,
        X_val_exp3,
        y_val_exp3,
        X_test_exp3,
        y_test_exp3,
    ) = load_experiment_3()

    # 2. Auto-Benchmark regression model comparison on Validation set
    run_auto_benchmark_experiment_3(
        X_train=X_train_exp3,
        Y_train=y_train_exp3,
        X_val=X_val_exp3,
        Y_val=y_val_exp3,
        plot_results=True,
    )

    # 3. Evaluate winning Linear Regression model on TEST set
    test_results_df, test_feature_details_df, country_year_df = (
        evaluate_linear_regression_on_test(
            X_train=X_train_exp3,
            Y_train=y_train_exp3,
            X_test=X_test_exp3,
            Y_test=y_test_exp3,
            plot_results=True,
        )
    )

    # 4. Plot global forecast stability drift analysis boxplots by year
    plot_global_yearly_boxplot(country_year_df, metric="Cosine_similarity")
    plot_global_yearly_boxplot(country_year_df, metric="RMSE")

    # 5. Plot 4-quadrant country stability scatter chart
    plot_country_stability_scatter_chart(country_year_df)


def step_5_closed_loop():
    print("\n=======================================================")
    print(
        "📍 STAGE 5: CLOSED-LOOP PIPELINE (EXP 3 FORECAST -> EXP 2 CLASSIFY)"
    )
    print("=======================================================")

    # 1. Load data for both Exp 2 and Exp 3
    (
        X_train_exp2,
        y_train_exp2,
        X_val_exp2,
        y_val_exp2,
        X_test_exp2,
        y_test_exp2,
        le,
    ) = load_experiment_2_country_classification()

    (
        X_train_exp3,
        y_train_exp3,
        X_val_exp3,
        y_val_exp3,
        X_test_exp3,
        y_test_exp3,
    ) = load_experiment_3()

    # 2. Execute closed-loop chained pipeline
    run_forecast_then_classify_pipeline(
        X_train_exp3=X_train_exp3,
        Y_train_exp3=y_train_exp3,
        X_test_exp3=X_test_exp3,
        X_train_exp2=X_train_exp2,
        y_train_exp2=y_train_exp2,
        y_test_exp2=y_test_exp2,
        label_encoder=le,
    )

    plot_misclassification_comparison(show_fig=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            "Boolean value expected (True/False or 1/0)."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Master Execution Pipeline for the Country Data Fingerprint Project"
        ),
        epilog=PIPELINE_HELPER_TEXT,  # Display helper instructions on --help
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "data", "exp1", "exp2", "exp3", "closed_loop"],
        help=(
            "Select stage to run: 'all', 'data', 'exp1', 'exp2', 'exp3',"
            " 'closed_loop' (Default: 'all')"
        ),
    )

    parser.add_argument(
        "--use_imputed",
        type=str2bool,
        default=True,
        help="Use imputed data (True) or unimputed data (False). Default: True",
    )

    parser.add_argument(
        "--drop_missing",
        type=str2bool,
        default=True,
        help=(
            "Filter out countries containing missing values (True/False."
            " Default: True)"
        ),
    )

    args = parser.parse_args()

    print("=======================================================")
    print("🌟 COUNTRY DATA FINGERPRINT - MASTER PIPELINE 🌟")
    print("=======================================================")

    if args.step in ["all", "data"]:
        step_1_data_processing(
            use_imputed=args.use_imputed,
            drop_countries_with_missing=args.drop_missing,
        )

    if args.step in ["all", "exp1"]:
        step_2_experiment_1()

    if args.step in ["all", "exp2"]:
        step_3_experiment_2()

    if args.step in ["all", "exp3"]:
        step_4_experiment_3()

    if args.step in ["all", "closed_loop"]:
        step_5_closed_loop()

    print("\n=======================================================")
    print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    main()
    """
# 1. Run full Pipeline end-to-end
python -m src.script.run_pipeline --step all --use_imputed True
python -m src.script.run_pipeline --step all --use_imputed False

# 2. Run individual pipeline stages:
python -m src.script.run_pipeline --step data         # Data processing only
python -m src.script.run_pipeline --step exp1         # Run Exp 1 only (Distance & HEB)
python -m src.script.run_pipeline --step exp2         # Run Exp 2 only (Classification)
python -m src.script.run_pipeline --step exp3         # Run Exp 3 only (Forecasting)
python -m src.script.run_pipeline --step closed_loop  # Run Closed-Loop only (Forecast -> Classify)
    """