# src/models/classifier.py
# cd country-data-fingerprint
# python -m src.models.classifier

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from config import paths
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier


def load_experiment_2_country_classification(
    data_dir: Path | None = None, encode_target: bool = True
):
    """Loads data for Experiment 2 with Target = 'country_code_3'.

    Target (y)   : country_code_3 (Country code)
    Features (X) : All socio-economic indicators (excluding country_code_3,
    country_name, year)
    """
    if data_dir is None:
        data_dir = paths.PROCESSED_DATA_EXPERIMENT2_DIR

    # 1. READ THE 3 CSV FILES
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # 2. DEFINE TARGET AND COLUMNS TO EXCLUDE FROM X
    target_col = "country_code_3"

    # ⚠️ IMPORTANT: Exclude country_name and year from X to prevent Data Leakage
    cols_to_exclude = ["country_code_3", "country_name", "year"]

    # 3. LIST FEATURE COLUMNS FOR X
    feature_cols = [
        col for col in train_df.columns if col not in cols_to_exclude
    ]

    # 4. SEPARATE X AND y
    X_train_exp2 = train_df[feature_cols]
    y_train_exp2 = train_df[target_col]

    X_val_exp2 = val_df[feature_cols]
    y_val_exp2 = val_df[target_col]

    X_test_exp2 = test_df[feature_cols]
    y_test_exp2 = test_df[target_col]

    # 5. ENCODE LABELS IN Y TO INTEGERS (0, 1, 2, ..., 193)
    label_encoder = None
    if encode_target:
        label_encoder = LabelEncoder()

        # Fit label encoder on Train set
        y_train_exp2 = label_encoder.fit_transform(y_train_exp2)
        y_val_exp2 = label_encoder.transform(y_val_exp2)
        y_test_exp2 = label_encoder.transform(y_test_exp2)

    # 6. PRINT CHECK INFORMATION
    print("=======================================================")
    print("EXPERIMENT 2: COUNTRY CLASSIFICATION (TARGET = country_code_3)")
    print("=======================================================")
    print(
        f"X_train_exp2 : {X_train_exp2.shape}  |  y_train_exp2 :"
        f" {y_train_exp2.shape}"
    )
    print(
        f"X_val_exp2   : {X_val_exp2.shape}  |  y_val_exp2   :"
        f" {y_val_exp2.shape}"
    )
    print(
        f"X_test_exp2  : {X_test_exp2.shape}  |  y_test_exp2  :"
        f" {y_test_exp2.shape}"
    )
    print(
        "Number of Classification Classes (Countries):"
        f" {len(train_df[target_col].unique())}"
    )
    print(f"Number of Features (X)                  : {len(feature_cols)}")

    return (
        X_train_exp2,
        y_train_exp2,
        X_val_exp2,
        y_val_exp2,
        X_test_exp2,
        y_test_exp2,
        label_encoder,
    )


def run_auto_benchmark_experiment_2(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    plot_results: bool = True,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Automatically trains and benchmarks multiple classifiers on the
    Validation set, saving results to a CSV file and generating a comparison
    chart.
    """
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT2_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    classes_list = np.unique(y_train)

    # 1. INITIALIZE ALL MODELS
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "SVC (Linear)": CalibratedClassifierCV(
            SVC(kernel="linear", random_state=42), cv=3
        ),
        "SVC (RBF)": CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=42), cv=3
        ),
        "MLP (Neural Net)": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        ),
        # 🌟 LIGHTGBM PARAMETERS
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=2,  # Allows leaf nodes to contain at least 2 samples
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        # 🌟 XGBOOST PARAMETERS
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=1,  # Allows smaller nodes to split
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
        # CatBoost
        "CatBoost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=4,
            random_state=42,
            verbose=0,
        ),
    }

    results = []

    print("=======================================================")
    print(f"🚀 STARTING AUTO-BENCHMARK ({len(models)} MODELS)")
    print(
        f"   Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples |"
        f" Classes: {len(classes_list)}"
    )
    print("=======================================================\n")

    # 2. TRAINING AND EVALUATION LOOP
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)

        # Predict labels (Top-1)
        y_val_pred = model.predict(X_val)

        # Predict probabilities (for Top-5 Accuracy)
        try:
            y_val_proba = model.predict_proba(X_val)
            top5_acc = top_k_accuracy_score(
                y_val, y_val_proba, k=5, labels=classes_list
            )
        except Exception:
            top5_acc = np.nan

        # Calculate metrics
        acc = accuracy_score(y_val, y_val_pred)
        f1_weighted = f1_score(y_val, y_val_pred, average="weighted")

        results.append(
            {
                "Model": name,
                "Val_Accuracy": acc,
                "Val_Top5_Acc": top5_acc,
                "Val_F1_Weighted": f1_weighted,
            }
        )

        print(
            f"✔️ {name:20s} | Acc: {acc:.4f} | Top-5 Acc: {top5_acc:.4f} | F1:"
            f" {f1_weighted:.4f}"
        )

    # 3. SORT RESULTS DATAFRAME BY VAL ACCURACY
    results_df = (
        pd.DataFrame(results)
        .sort_values("Val_Accuracy", ascending=False)
        .reset_index(drop=True)
    )

    print(
        f"\n{'='*25} 🏆 MODEL RANKING {'='*25}\n",
        results_df.to_string(index=False),
    )

    # 💾 1. SAVE RESULTS TABLE TO CSV
    csv_path = output_dir / "auto_benchmark_validation_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n📁 Saved CSV results file: {csv_path.name}")

    # 4. PLOT AND SAVE CHART
    if plot_results:
        ax = results_df.plot(
            x="Model",
            y=["Val_Accuracy", "Val_Top5_Acc", "Val_F1_Weighted"],
            kind="barh",
            figsize=(10, 7),
            color=["#3498db", "#2ecc71", "#e74c3c"],
            width=0.7,
        )

        # Format plot
        ax.invert_yaxis()
        ax.set(
            xlim=(0, 1.15),
            xlabel="Score",
            title="Model Performance Comparison (Validation Set)",
        )

        # Place legend below chart horizontally with 3 labels
        ax.legend(
            [
                "Val Accuracy (Top-1)",
                "Val Top-5 Accuracy",
                "Val F1 (Weighted)",
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            frameon=False,
        )

        # Automatically add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

        plt.tight_layout()

        # 🖼️ 2. SAVE BENCHMARK CHART IMAGE
        img_path = output_dir / "auto_benchmark_validation_comparison.png"
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Saved plot image: {img_path.name}")

        plt.show()

    print("\n=======================================================")
    print(f"🎉 ALL CSV FILES AND BENCHMARK IMAGES SAVED TO: {output_dir}")
    print("=======================================================")

    return results_df


def evaluate_extra_trees_on_test(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    output_dir: Path | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Conducts an in-depth evaluation of the Extra Trees model on the Test set:

    1. Computes Test_Accuracy, Test_Top5_Acc, and Test_F1_Weighted metrics.
    2. Extracts, exports to CSV, and saves feature importance plots.
    3. Performs detailed error analysis on misclassified countries, exporting CSVs
       and saving a Heatmap visualization.
    """
    # Handle output directory for CSVs and plots
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT2_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    classes_list = np.unique(y_train)

    # =============================================================
    # 1. TRAIN AND EVALUATE ON TEST SET
    # =============================================================
    print("=======================================================")
    print("🎯 EVALUATING EXTRA TREES MODEL ON TEST SET (2021-2024)")
    print("=======================================================\n")

    name = "Extra Trees"
    model = ExtraTreesClassifier(
        n_estimators=100, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict labels and probabilities
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_test_pred)
    top5_acc = top_k_accuracy_score(
        y_test, y_test_proba, k=5, labels=classes_list
    )
    f1 = f1_score(y_test, y_test_pred, average="weighted")

    test_results = [{
        "Model": name,
        "Test_Accuracy": acc,
        "Test_Top5_Acc": top5_acc,
        "Test_F1_Weighted": f1,
    }]

    results_df = pd.DataFrame(test_results)

    # 💾 1. SAVE TEST EVALUATION RESULTS TO CSV
    results_csv_path = output_dir / "extra_trees_test_results.csv"
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8")

    print("📊 TEST SET EVALUATION TABLE:")
    print(results_df.to_string(index=False))
    print(f"📁 Saved CSV: {results_csv_path.name}")

    # =============================================================
    # 2. EXTRACT, SAVE CSV, AND PLOT FEATURE IMPORTANCES
    # =============================================================
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    # 💾 2. SAVE FEATURE IMPORTANCES TO CSV
    importances_df = importances.reset_index()
    importances_df.columns = ["Feature", "Importance"]
    importances_csv_path = output_dir / "extra_trees_feature_importances.csv"
    importances_df.to_csv(importances_csv_path, index=False, encoding="utf-8")

    print(
        f"\n🔥 ALL {len(importances)} KEY INDICATORS FOR COUNTRY"
        " IDENTIFICATION:"
    )
    print(importances.to_string())
    print(f"📁 Saved CSV: {importances_csv_path.name}")

    # Calculate plot height dynamically based on feature count
    fig_height = max(6, len(importances) * 0.35)

    # Plot all feature importances
    plt.figure(figsize=(10, fig_height))
    ax = importances.plot(kind="barh", color="#2ecc71")

    # Display precise values at the end of each bar
    ax.bar_label(ax.containers[0], fmt="%.4f", padding=3, fontsize=9)
    plt.title(
        "All Feature Importances (Extra Trees Model)",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Importance Score", fontsize=11)
    plt.xlim(0, importances.max() * 1.12)
    plt.gca().invert_yaxis()  # Put most important feature at the top
    plt.tight_layout()

    # framed 🖼️ SAVE FEATURE IMPORTANCE CHART IMAGE
    importances_img_path = output_dir / "extra_trees_feature_importances.png"
    plt.savefig(importances_img_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ Saved plot image: {importances_img_path.name}")
    plt.show()

    # =============================================================
    # 3. ERROR ANALYSIS
    # =============================================================
    y_test_true_codes = label_encoder.inverse_transform(y_test)
    y_test_pred_codes = label_encoder.inverse_transform(y_test_pred)

    # 1. Create mask to filter misclassified samples
    errors_mask = y_test_true_codes != y_test_pred_codes

    # 2. Create DataFrame containing misclassified cases
    error_df = pd.DataFrame({
        "Real_Country": y_test_true_codes[errors_mask],
        "Predicted_Country": y_test_pred_codes[errors_mask],
    })

    print(
        f"\n⚠️ TOTAL MISCLASSIFIED SAMPLES: {len(error_df)} /"
        f" {len(y_test_true_codes)}"
    )

    if not error_df.empty:
        # Group and count error pairs
        summary_df = error_df.value_counts().reset_index(name="Count")

        # 💾 3. SAVE MISCLASSIFICATION PAIRS REPORT TO CSV
        error_summary_csv_path = (
            output_dir / "extra_trees_misclassification_summary.csv"
        )
        summary_df.to_csv(
            error_summary_csv_path, index=False, encoding="utf-8"
        )

        # 3. Create cross-tabulation count matrix
        pivot_df = pd.crosstab(
            index=error_df["Real_Country"],
            columns=error_df["Predicted_Country"],
        )

        # 💾 4. SAVE ERROR MATRIX TO CSV
        error_matrix_csv_path = (
            output_dir / "extra_trees_misclassification_matrix.csv"
        )
        pivot_df.to_csv(error_matrix_csv_path, encoding="utf-8")

        # 4. Plot Heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            cbar=False,
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(
            "Misclassification Heatmap (2021 - 2024)",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Predicted Country", fontsize=10)
        plt.ylabel("Real Country", fontsize=10)
        plt.tight_layout()

        # 🖼️ SAVE ERROR HEATMAP IMAGE
        heatmap_img_path = (
            output_dir / "extra_trees_misclassification_heatmap.png"
        )
        plt.savefig(heatmap_img_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Saved plot image: {heatmap_img_path.name}")
        plt.show()

        print("\nMisclassified country details:")
        print(summary_df.to_string(index=False))
        print(f"📁 Saved CSV: {error_summary_csv_path.name}")
        print(f"📁 Saved CSV: {error_matrix_csv_path.name}")
    else:
        summary_df = pd.DataFrame(
            columns=["Real_Country", "Predicted_Country", "Count"]
        )
        summary_df.to_csv(
            output_dir / "extra_trees_misclassification_summary.csv",
            index=False,
            encoding="utf-8",
        )
        print("🎉 Great! No misclassified samples on the Test set.")

    print("\n=======================================================")
    print(f"🎉 ALL CSV FILES AND CHART IMAGES SAVED TO: {output_dir}")
    print("=======================================================")

    return results_df, importances, error_df


if __name__ == "__main__":
    (
        X_train_exp2,
        y_train_exp2,
        X_val_exp2,
        y_val_exp2,
        X_test_exp2,
        y_test_exp2,
        le,
    ) = load_experiment_2_country_classification(encode_target=True)

    benchmark_df = run_auto_benchmark_experiment_2(
        X_train=X_train_exp2,
        y_train=y_train_exp2,
        X_val=X_val_exp2,
        y_val=y_val_exp2,
        plot_results=True,
    )

    results_df, features, error_df = evaluate_extra_trees_on_test(
        X_train=X_train_exp2,
        y_train=y_train_exp2,
        X_test=X_test_exp2,
        y_test=y_test_exp2,
        label_encoder=le,
    )