# src/models/forecast_then_classify.py
# cd country-data-fingerprint
# python -m src.models.forecast_then_classify
from pathlib import Path
from config import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from src.models.classifier import load_experiment_2_country_classification
from src.models.forecast import load_experiment_3


def run_forecast_then_classify_pipeline(
    X_train_exp3: pd.DataFrame,
    Y_train_exp3: pd.DataFrame,
    X_test_exp3: pd.DataFrame,
    X_train_exp2: pd.DataFrame,
    y_train_exp2: np.ndarray,
    y_test_exp2: np.ndarray,
    label_encoder,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exp 3 -> Exp 2 Chained Pipeline:

    1. Use Linear Regression (Exp 3) to forecast 20 future macroeconomic indicators (2021-2024).
    2. Use ExtraTreesClassifier (Exp 2) to classify countries based on the forecasted indicators.
    3. Evaluate whether the model correctly identifies countries from their forecasted fingerprints.
    4. Export misclassification CSV pairs, error Pivot matrix, and save Heatmap image.
    """
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT4_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print(
        "🚀 STARTING CHAINED PIPELINE: FORECAST (EXP 3) -> CLASSIFY (EXP 2)"
    )
    print("=======================================================\n")

    # 1. STEP 1: TRAIN LINEAR REGRESSION AND FORECAST FUTURE MACRO FINGERPRINTS (2021-2024)
    print("🔄 1. Training Linear Regression (Exp 3)...")
    forecaster = LinearRegression()
    forecaster.fit(X_train_exp3, Y_train_exp3)

    # Forecast 20 future indicators for the Test set (2021-2024)
    Y_pred_future_array = forecaster.predict(X_test_exp3)
    Y_pred_future = pd.DataFrame(
        Y_pred_future_array,
        columns=X_train_exp2.columns,
        index=X_test_exp3.index,
    )
    print(
        f"   👉 Successfully generated {Y_pred_future.shape[0]} forecasted macro fingerprints for 2021-2024."
    )

    # 2. STEP 2: TRAIN EXTRATREES CLASSIFIER (EXP 2)
    print("\n🔄 2. Training ExtraTreesClassifier (Exp 2)...")
    classifier = ExtraTreesClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    classifier.fit(X_train_exp2, y_train_exp2)

    # 3. STEP 3: PASS FUTURE FINGERPRINTS TO CLASSIFIER FOR COUNTRY IDENTIFICATION
    print(
        "\n🔄 3. Feeding forecasted fingerprints into Classifier for country identification..."
    )
    classes_list = np.unique(y_train_exp2)

    y_pipeline_pred = classifier.predict(Y_pred_future)
    y_pipeline_proba = classifier.predict_proba(Y_pred_future)

    # 4. STEP 4: EVALUATE ACCURACY
    acc = accuracy_score(y_test_exp2, y_pipeline_pred)
    top5_acc = top_k_accuracy_score(
        y_test_exp2, y_pipeline_proba, k=5, labels=classes_list
    )
    f1 = f1_score(y_test_exp2, y_pipeline_pred, average="weighted")

    pipeline_results_df = pd.DataFrame(
        [
            {
                "Pipeline": (
                    "LinearRegression (Exp3 Forecast) -> ExtraTrees (Exp2"
                    " Classify)"
                ),
                "Test_Accuracy": acc,
                "Test_Top5_Acc": top5_acc,
                "Test_F1_Weighted": f1,
            }
        ]
    )

    print("=======================================================")
    print(
        "🏆 COUNTRY IDENTIFICATION RESULTS FROM FORECASTED FINGERPRINTS"
        " (2021-2024):"
    )
    print("=======================================================")
    print(pipeline_results_df.to_string(index=False))

    output_csv = output_dir / "pipeline_forecast_then_classify_results.csv"
    pipeline_results_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n📁 Saved Pipeline results file: {output_csv.name}")

    # 5. ANALYZE AND EXPORT MISCLASSIFIED COUNTRIES REPORT
    y_test_true_codes = label_encoder.inverse_transform(y_test_exp2)
    y_pipeline_pred_codes = label_encoder.inverse_transform(y_pipeline_pred)

    errors_mask = y_test_true_codes != y_pipeline_pred_codes
    error_df = pd.DataFrame(
        {
            "Real_Country": y_test_true_codes[errors_mask],
            "Predicted_Country": y_pipeline_pred_codes[errors_mask],
        }
    )

    print(
        f"\n⚠️ Number of misclassified samples: {len(error_df)} /"
        f" {len(y_test_exp2)}"
    )

    if not error_df.empty:
        # a) Save CSV report listing misclassified pairs with frequency counts
        error_summary = error_df.value_counts().reset_index(name="Count")
        error_summary_csv = output_dir / "pipeline_misclassifications.csv"
        error_summary.to_csv(error_summary_csv, index=False, encoding="utf-8")
        print(f"📁 Saved misclassified pairs report: {error_summary_csv.name}")

        # b) Create and save Error Pivot Matrix to CSV
        error_pivot = pd.crosstab(
            error_df["Real_Country"],
            error_df["Predicted_Country"],
            margins=False,
        )
        error_pivot_csv = output_dir / "pipeline_error_pivot_matrix.csv"
        error_pivot.to_csv(error_pivot_csv, encoding="utf-8")
        print(f"📁 Saved error pivot matrix: {error_pivot_csv.name}")

        # c) Plot and save Error Heatmap Chart
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            error_pivot,
            annot=True,
            fmt="d",
            cmap="Reds",
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(
            "Misclassification Heatmap (Real vs Predicted Country)",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Predicted Country", fontsize=11)
        plt.ylabel("Real Country", fontsize=11)
        plt.tight_layout()

        heatmap_path = output_dir / "pipeline_error_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"🖼️ Saved error heatmap image: {heatmap_path.name}")
    else:
        print("🎉 No classification errors occurred!")

    return pipeline_results_df, error_df


def plot_misclassification_comparison(
    extra_trees_csv: Path | str | None = None,
    pipeline_csv: Path | str | None = None,
    output_dir: Path | None = None,
    show_fig: bool = True,
) -> pd.DataFrame:
    """Read two CSV files containing misclassification statistics (Standalone Extra Trees & Closed-loop Pipeline),

    merge them, and plot a grouped bar chart comparing misclassification pairs (Actual -> Predicted).
    """
    if extra_trees_csv is None:
        extra_trees_csv = (
            paths.RESULTS_EXPERIMENT2_DIR
            / "extra_trees_misclassification_summary.csv"
        )
    else:
        extra_trees_csv = Path(extra_trees_csv)

    if pipeline_csv is None:
        pipeline_csv = (
            paths.RESULTS_EXPERIMENT4_DIR / "pipeline_misclassifications.csv"
        )
    else:
        pipeline_csv = Path(pipeline_csv)

    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT4_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. VALIDATE INPUT FILES
    if not extra_trees_csv.exists() or not pipeline_csv.exists():
        print(
            "[ERROR] Misclassification CSV file not found:"
            f" {extra_trees_csv} or {pipeline_csv}"
        )
        return pd.DataFrame()

    # 2. LOAD DATA FROM CSV FILES
    df_et = pd.read_csv(extra_trees_csv)
    df_pipe = pd.read_csv(pipeline_csv)

    # 3. MERGE DATASETS
    merged = pd.merge(
        df_et,
        df_pipe,
        on=["Real_Country", "Predicted_Country"],
        how="outer",
        suffixes=("_Standalone", "_Pipeline"),
    ).fillna(0)

    # Ensure count columns are integers
    merged["Count_Standalone"] = merged["Count_Standalone"].astype(int)
    merged["Count_Pipeline"] = merged["Count_Pipeline"].astype(int)

    # Sort by total misclassification count in descending order
    merged["Total_Count"] = (
        merged["Count_Standalone"] + merged["Count_Pipeline"]
    )
    merged = merged.sort_values("Total_Count", ascending=False).reset_index(
        drop=True
    )

    # Create category labels "Actual -> Predicted"
    labels = merged["Real_Country"] + " -> " + merged["Predicted_Country"]

    # 4. PLOT GROUPED BAR CHART
    plt.figure(figsize=(12, 6))

    x_positions = np.arange(len(merged))
    bar_width = 0.38

    bars1 = plt.bar(
        x_positions - bar_width / 2,
        merged["Count_Standalone"],
        width=bar_width,
        label="Extra Trees on Test Set",
        color="#2b5c8f",
    )

    bars2 = plt.bar(
        x_positions + bar_width / 2,
        merged["Count_Pipeline"],
        width=bar_width,
        label="Extra Trees on Forecasted Set",
        color="#d95f02",
    )

    plt.xticks(x_positions, labels, rotation=45, ha="right", fontsize=9)
    plt.xlabel(
        "Misclassification Pair (Actual -> Predicted)",
        fontsize=11,
        fontweight="bold",
    )
    plt.ylabel("Count", fontsize=11, fontweight="bold")
    plt.title(
        "Misclassification Comparison: Standalone Extra Trees vs. Closed-Loop"
        " Pipeline",
        fontsize=12,
        fontweight="bold",
    )
    plt.legend(frameon=True, loc="upper right")
    plt.grid(False)

    # Display value annotations on top of each bar
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    # 🖼️ SAVE PLOT AND EXPORT MERGED CSV
    img_path = (
        output_dir
        / "misclassification_comparison_standalone_vs_pipeline.png"
    )
    plt.savefig(img_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ Misclassification comparison plot saved to: {img_path.name}")

    merged_csv_path = output_dir / "misclassification_comparison_merged.csv"
    merged.to_csv(merged_csv_path, index=False, encoding="utf-8")
    print(
        f"📁 Merged misclassification summary saved to: {merged_csv_path.name}"
    )

    if show_fig:
        plt.show()

    return merged


if __name__ == "__main__":
    # 1. Load Experiment 2 data (for Classifier)
    (
        X_train_exp2,
        y_train_exp2,
        X_val_exp2,
        y_val_exp2,
        X_test_exp2,
        y_test_exp2,
        le,
    ) = load_experiment_2_country_classification(encode_target=True)

    # 2. Load Experiment 3 data (for Forecaster)
    (
        X_train_exp3,
        y_train_exp3,
        X_val_exp3,
        y_val_exp3,
        X_test_exp3,
        y_test_exp3,
    ) = load_experiment_3()

    # 3. Run Chained Pipeline (Exp 3 -> Exp 2)
    pipeline_results_df, pipeline_error_df = (
        run_forecast_then_classify_pipeline(
            X_train_exp3=X_train_exp3,
            Y_train_exp3=y_train_exp3,
            X_test_exp3=X_test_exp3,
            X_train_exp2=X_train_exp2,
            y_train_exp2=y_train_exp2,
            y_test_exp2=y_test_exp2,
            label_encoder=le,
        )
    )

    plot_misclassification_comparison(show_fig=True)