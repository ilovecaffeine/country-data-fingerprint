# src/models/forecast.py
# cd country-data-fingerprint
# python -m src.models.forecast

from pathlib import Path
from catboost import CatBoostRegressor
from config import paths
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def load_experiment_3(
    data_dir: Path | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Reads 3 CSV files (train.csv, validation.csv, test.csv) from
    PROCESSED_DATA_EXPERIMENT3_DIR and separates the Feature (X) and Target (Y)
    sets for the 20-indicator forecasting task.

    Returns:
        X_train_exp3, y_train_exp3,
        X_val_exp3, y_val_exp3,
        X_test_exp3, y_test_exp3
    """
    if data_dir is None:
        data_dir = paths.PROCESSED_DATA_EXPERIMENT3_DIR

    # 1. READ 3 CSV FILES
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"[ERROR] Exp 3 data file not found: {p}"
            )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 2. AUTOMATICALLY FILTER ID COLUMNS, FEATURES (X), AND TARGETS (Y)
    id_cols = ["country_code_3", "country_name", "year"]

    # Target columns (Y) are those ending with '_target_next_year'
    target_cols = [
        col for col in train_df.columns if col.endswith("_target_next_year")
    ]

    # Feature columns (X) are indicator columns for year t (excluding ID and Target columns)
    feature_cols = [
        col
        for col in train_df.columns
        if col not in id_cols and col not in target_cols
    ]

    # 3. SEPARATE X AND Y FOR EACH SPLIT
    X_train_exp3 = train_df[feature_cols]
    y_train_exp3 = train_df[target_cols]

    X_val_exp3 = val_df[feature_cols]
    y_val_exp3 = val_df[target_cols]

    X_test_exp3 = test_df[feature_cols]
    y_test_exp3 = test_df[target_cols]

    # 4. PRINT VERIFICATION INFORMATION
    print("=======================================================")
    print("EXPERIMENT 3 DATA LOAD AND SPLIT RESULTS (FORECASTING)")
    print("=======================================================")
    print(
        f"X_train_exp3 : {X_train_exp3.shape}  |  y_train_exp3 :"
        f" {y_train_exp3.shape}"
    )
    print(
        f"X_val_exp3   : {X_val_exp3.shape}  |  y_val_exp3   :"
        f" {y_val_exp3.shape}"
    )
    print(
        f"X_test_exp3  : {X_test_exp3.shape}  |  y_test_exp3  :"
        f" {y_test_exp3.shape}"
    )
    print(f"Number of Input Columns (Features X): {len(feature_cols)}")
    print(f"Number of Output Columns (Targets Y) : {len(target_cols)}")

    return (
        X_train_exp3,
        y_train_exp3,
        X_val_exp3,
        y_val_exp3,
        X_test_exp3,
        y_test_exp3,
    )


def run_auto_benchmark_experiment_3(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    plot_results: bool = True,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Automatically trains and benchmarks 9 Multi-Output Regression methods
    (Multi-Output Forecast for 20 indicators) on the Validation set. Exports an
    overall ranking CSV, per-feature detailed metric CSV, and saves 2 chart
    images (Comparison + R2 Heatmap).
    """
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT3_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. INITIALIZE ALL 9 MODELS
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest (Native)": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Extra Trees (Native)": ExtraTreesRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "MultiOutput + Random Forest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        "MultiOutput + Extra Trees": MultiOutputRegressor(
            ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "LightGBM": MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                min_child_samples=2,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
        ),
        "CatBoost": CatBoostRegressor(
            loss_function="MultiRMSE",
            iterations=300,
            learning_rate=0.05,
            depth=4,
            random_state=42,
            verbose=0,
        ),
        "MLP (Neural Net)": MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        ),
    }

    results = []
    feature_details = []

    print("=======================================================")
    print(
        "🚀 STARTING EXP 3 AUTO-BENCHMARK ("
        f"{len(models)} REGRESSION METHODS FOR 20 INDICATORS)"
    )
    print(
        f"   Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples |"
        f" Target Outputs: {Y_train.shape[1]}"
    )
    print("=======================================================\n")

    # 2. TRAINING AND EVALUATION LOOP ON VALIDATION SET
    for name, model in models.items():
        # Fit model
        model.fit(X_train, Y_train)

        # Predict Y on Validation set
        Y_val_pred = model.predict(X_val)

        # A. CALCULATE OVERALL MEAN METRICS
        mean_r2 = r2_score(Y_val, Y_val_pred, multioutput="uniform_average")
        mean_mae = mean_absolute_error(Y_val, Y_val_pred)
        mean_rmse = np.sqrt(mean_squared_error(Y_val, Y_val_pred))

        results.append(
            {
                "Model": name,
                "Val_Mean_R2": mean_r2,
                "Val_Mean_MAE": mean_mae,
                "Val_Mean_RMSE": mean_rmse,
            }
        )

        # B. CALCULATE DETAILED R2, MAE, RMSE FOR EACH FEATURE
        r2_raw = r2_score(Y_val, Y_val_pred, multioutput="raw_values")
        mae_raw = mean_absolute_error(
            Y_val, Y_val_pred, multioutput="raw_values"
        )
        rmse_raw = np.sqrt(
            mean_squared_error(Y_val, Y_val_pred, multioutput="raw_values")
        )

        for col_name, r2_v, mae_v, rmse_v in zip(
            Y_val.columns, r2_raw, mae_raw, rmse_raw
        ):
            # Clean feature name (strip _target_next_year suffix)
            clean_feature_name = col_name.replace("_target_next_year", "")
            feature_details.append(
                {
                    "Model": name,
                    "Feature": clean_feature_name,
                    "Val_R2": r2_v,
                    "Val_MAE": mae_v,
                    "Val_RMSE": rmse_v,
                }
            )

        print(
            f"✔️ {name:30s} | R2: {mean_r2:.4f} | MAE: {mean_mae:.4f} | RMSE:"
            f" {mean_rmse:.4f}"
        )

    # 3. OVERALL & DETAILED DATAFRAMES
    results_df = (
        pd.DataFrame(results)
        .sort_values("Val_Mean_R2", ascending=False)
        .reset_index(drop=True)
    )

    feature_details_df = pd.DataFrame(feature_details)

    print(
        f"\n{'='*25} 🏆 EXP 3 MODEL RANKING (VALIDATION) {'='*25}\n",
        results_df.to_string(index=False),
    )

    # 💾 1. SAVE OVERALL BENCHMARK RESULTS CSV FILE
    csv_path = output_dir / "forecast_auto_benchmark_val_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n📁 Saved overall summary file: {csv_path.name}")

    # 💾 2. SAVE PER-FEATURE METRIC DETAILS CSV FILE
    feature_csv_path = (
        output_dir / "forecast_auto_benchmark_val_feature_details.csv"
    )
    feature_details_df.to_csv(feature_csv_path, index=False, encoding="utf-8")
    print(f"📁 Saved per-feature details file: {feature_csv_path.name}")

    # 4. PLOT & SAVE CHARTS
    if plot_results:
        # --- CHART 1: HORIZONTAL BAR COMPARISON OF OVERALL R2 ---
        fig, ax = plt.subplots(figsize=(10, 6))
        results_df.plot(
            x="Model",
            y="Val_Mean_R2",
            kind="barh",
            ax=ax,
            color="#3498db",
            width=0.6,
            legend=False,
        )

        ax.invert_yaxis()
        ax.set(
            xlim=(0, max(1.05, results_df["Val_Mean_R2"].max() * 1.1)),
            xlabel="Mean R2 Score",
            ylabel="",
            title="Multi-Output Regression Performance Comparison (Validation Set)",
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f", padding=3, fontsize=9)

        plt.tight_layout()

        # 🖼️ SAVE HORIZONTAL BAR CHART IMAGE
        img_path = output_dir / "forecast_auto_benchmark_val_comparison.png"
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Saved comparison plot image: {img_path.name}")
        plt.show()

        # --- CHART 2: R2 SCORE MATRIX HEATMAP (FEATURES VS MODELS) ---
        pivot_r2 = feature_details_df.pivot(
            index="Feature", columns="Model", values="Val_R2"
        )
        # Sort features by mean R2 score descending
        pivot_r2["Mean_R2"] = pivot_r2.mean(axis=1)
        pivot_r2 = pivot_r2.sort_values("Mean_R2", ascending=False).drop(
            columns=["Mean_R2"]
        )

        plt.figure(figsize=(12, 9))
        sns.heatmap(
            pivot_r2,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            cbar_kws={"label": "Val R² Score"},
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(
            "Val R² Score Heatmap (Features vs Models)",
            fontsize=13,
            fontweight="bold",
        )
        plt.xlabel("Model", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # 🖼️ SAVE R2 HEATMAP IMAGE
        heatmap_img_path = (
            output_dir / "forecast_auto_benchmark_val_r2_heatmap.png"
        )
        plt.savefig(heatmap_img_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Saved R² Heatmap image: {heatmap_img_path.name}")
        plt.show()

    print("\n=======================================================")
    print(f"🎉 ALL CSV FILES AND CHART IMAGES SAVED TO: {output_dir}")
    print("=======================================================")

    return results_df, feature_details_df


def evaluate_linear_regression_on_test(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    output_dir: Path | None = None,
    test_csv_file: Path | None = None,
    plot_results: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """In-depth evaluation of the Linear Regression model on the TEST set
    (2021-2024):

    1. Overall results table: Model, Test_Mean_R2, Test_Mean_MAE,
    Test_Mean_RMSE.
    2. Per-feature details table: Feature, Test_R2, Test_MAE, Test_RMSE.
    3. Country-Year error metrics table: country_code_3, country_name, year,
    MAE, RMSE, Euclidean distance, Cosine similarity.
    4. Exports 3 CSV files and saves the R2 plot image.
    """
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT3_DIR

    if test_csv_file is None:
        test_csv_file = paths.PROCESSED_DATA_EXPERIMENT3_DIR / "test.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print(
        "🎯 EVALUATING LINEAR REGRESSION MODEL ON TEST SET (2021-2024)"
    )
    print("=======================================================\n")

    # 1. TRAIN LINEAR REGRESSION MODEL
    model_name = "Linear Regression"
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # 2. PREDICT ON TEST SET
    Y_test_pred = model.predict(X_test)

    # 3. OVERALL METRICS
    mean_r2 = r2_score(Y_test, Y_test_pred, multioutput="uniform_average")
    mean_mae = mean_absolute_error(Y_test, Y_test_pred)
    mean_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))

    test_results_df = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Test_Mean_R2": mean_r2,
                "Test_Mean_MAE": mean_mae,
                "Test_Mean_RMSE": mean_rmse,
            }
        ]
    )

    print("📊 1. OVERALL TEST SET PERFORMANCE SUMMARY:")
    print(test_results_df.to_string(index=False))

    # 💾 SAVE OVERALL SUMMARY TABLE TO CSV
    test_results_csv = output_dir / "linear_regression_test_results.csv"
    test_results_df.to_csv(test_results_csv, index=False, encoding="utf-8")
    print(f"\n📁 Saved overall summary CSV: {test_results_csv.name}")

    # 4. PER-FEATURE METRICS
    r2_raw = r2_score(Y_test, Y_test_pred, multioutput="raw_values")
    mae_raw = mean_absolute_error(
        Y_test, Y_test_pred, multioutput="raw_values"
    )
    rmse_raw = np.sqrt(
        mean_squared_error(Y_test, Y_test_pred, multioutput="raw_values")
    )

    feature_results = []
    for col_name, r2_v, mae_v, rmse_v in zip(
        Y_test.columns, r2_raw, mae_raw, rmse_raw
    ):
        clean_name = col_name.replace("_target_next_year", "")
        feature_results.append(
            {
                "Feature": clean_name,
                "Test_R2": r2_v,
                "Test_MAE": mae_v,
                "Test_RMSE": rmse_v,
            }
        )

    test_feature_details_df = (
        pd.DataFrame(feature_results)
        .sort_values("Test_R2", ascending=False)
        .reset_index(drop=True)
    )

    print("\n🔥 2. PER-FEATURE METRIC DETAILS ON TEST SET:")
    print(test_feature_details_df.to_string(index=False))

    # 💾 SAVE PER-FEATURE DETAILS TO CSV
    feature_csv = output_dir / "linear_regression_test_feature_details.csv"
    test_feature_details_df.to_csv(feature_csv, index=False, encoding="utf-8")
    print(f"\n📁 Saved per-feature details CSV: {feature_csv.name}")

    # 5. CALCULATE COUNTRY-YEAR ERROR METRICS
    country_year_df = pd.DataFrame()
    if test_csv_file.exists():
        test_meta = pd.read_csv(test_csv_file)[
            ["country_code_3", "country_name", "year"]
        ]

        Y_true_vals = Y_test.values
        Y_pred_vals = Y_test_pred

        # Row-wise computation (each country-year sample across 20 features)
        # a) Row MAE
        row_mae = np.mean(np.abs(Y_true_vals - Y_pred_vals), axis=1)

        # b) Row RMSE
        row_rmse = np.sqrt(np.mean((Y_true_vals - Y_pred_vals) ** 2, axis=1))

        # c) Row Euclidean distance between actual and predicted
        row_euclidean = np.sqrt(
            np.sum((Y_true_vals - Y_pred_vals) ** 2, axis=1)
        )

        # d) Row Cosine similarity between actual and predicted
        dot_product = np.sum(Y_true_vals * Y_pred_vals, axis=1)
        norm_true = np.linalg.norm(Y_true_vals, axis=1)
        norm_pred = np.linalg.norm(Y_pred_vals, axis=1)
        row_cosine_sim = dot_product / (norm_true * norm_pred + 1e-12)

        country_year_df = pd.DataFrame(
            {
                "country_code_3": test_meta["country_code_3"],
                "country_name": test_meta["country_name"],
                "year": test_meta["year"] + 1,  # Year t + 1 = 2021 to 2024
                "MAE": row_mae,
                "RMSE": row_rmse,
                "Euclidean_distance": row_euclidean,
                "Cosine_similarity": row_cosine_sim,
            }
        ).sort_values(["country_code_3", "year"]).reset_index(drop=True)

        # 💾 SAVE COUNTRY-YEAR ERROR METRICS TO CSV
        country_year_csv = (
            output_dir / "linear_regression_test_country_year_errors.csv"
        )
        country_year_df.to_csv(
            country_year_csv, index=False, encoding="utf-8"
        )

        print(
            "\n📍 3. DETAILED COUNTRY-YEAR ERROR METRICS 2021-2024 (HEAD 10):"
        )
        print(country_year_df.head(10).to_string(index=False))
        print(
            f"\n📁 Saved Country-Year error metrics CSV: {country_year_csv.name}"
        )

    # 6. PLOT AND SAVE TEST R2 SCORE CHART
    if plot_results:
        plt.figure(figsize=(10, 8))
        bars = plt.barh(
            test_feature_details_df["Feature"][::-1],
            test_feature_details_df["Test_R2"][::-1],
            color="#2ecc71",
            height=0.6,
        )

        plt.title(
            "Linear Regression - Test R² Score across 20 Indicators"
            " (2021-2024)",
            fontsize=12,
            fontweight="bold",
        )
        plt.xlabel("Test R² Score", fontsize=11)
        plt.xlim(
            min(0, test_feature_details_df["Test_R2"].min() - 0.05), 1.08
        )

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()

        # 🖼️ SAVE CHART IMAGE
        img_path = output_dir / "linear_regression_test_feature_r2.png"
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Saved Test R² plot image: {img_path.name}")

        plt.show()

    print("\n=======================================================")
    print(
        "🎉 ALL LINEAR REGRESSION TEST RESULTS SAVED TO: "
        f"{output_dir}"
    )
    print("=======================================================")

    return test_results_df, test_feature_details_df, country_year_df


def plot_global_yearly_boxplot(
    country_year_df: pd.DataFrame,
    metric: str = "Cosine_similarity",
    save_path: Path | str | None = None,
):
    """Plots a boxplot showing the distribution of scores across all countries by
    year.

    Auto-saves to paths.RESULTS_EXPERIMENT3_DIR if save_path is None.
    """
    # 1. Automatically determine default save path if save_path is None
    if save_path is None:
        metric_lower = metric.lower().replace(" ", "_")
        if "cosine_similarity" in metric_lower:
            save_path = (
                paths.RESULTS_EXPERIMENT3_DIR
                / "global_yearly_boxplot_cosine_similarity.png"
            )
        elif "rmse" in metric_lower:
            save_path = (
                paths.RESULTS_EXPERIMENT3_DIR
                / "global_yearly_boxplot_rmse.png"
            )
        else:
            save_path = (
                paths.RESULTS_EXPERIMENT3_DIR
                / f"global_yearly_boxplot_{metric_lower}.png"
            )
    else:
        save_path = Path(save_path)

    # Automatically create directory if it does not exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Plot Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=country_year_df,
        x="year",
        y=metric,
        hue="year",  # Assign x to hue to avoid deprecation warning
        palette="Set2",
        legend=False,  # Turn off redundant legend
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "red",
            "markeredgecolor": "red",
        },
    )

    plt.title(
        f"Global Forecast Drift Evaluation Across Years ({metric})",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Forecast Year", fontsize=11)
    plt.ylabel(metric, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # 3. Save image and display
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ Boxplot image saved to: {save_path.name}")

    plt.show()


def plot_country_stability_scatter_chart(
    country_year_df: pd.DataFrame,
    output_path: Path | str | None = None,
    csv_output_path: Path | str | None = None,
    show_fig: bool = True,
    marker_size: int = 10,
    x_max: float | None = 2.5,  # X-axis limit to exclude outlier ZWE
    y_max: float | None = 1.5,  # Y-axis limit
) -> pd.DataFrame:
    """Calculates country_stability_df, saves stability statistics CSV, and

    generates/saves an interactive Plotly Scatter Plot by country.

    Parameters
    ----------
    country_year_df : pd.DataFrame
        DataFrame containing columns: country_code_3, country_name, year,
        RMSE, Cosine_similarity (or Cosine similarity).
    output_path : Path | str, optional
        Path to save interactive HTML file. Default:
        country_stability_interactive.html
    csv_output_path : Path | str, optional
        Path to save stability statistics CSV file. Default:
        linear_regression_country_stability.csv
    show_fig : bool, default=True
        Whether to display the figure directly in Jupyter/Colab.
    marker_size : int, default=10
        Fixed size of marker points.
    x_max : float | None, default=2.5
        Upper limit for X-axis.
    y_max : float | None, default=1.5
        Upper limit for Y-axis.

    Returns
    -------
    pd.DataFrame
        Returns grouped and calculated country_stability_df.
    """
    # 1. Automatically determine save paths for HTML and CSV files
    if output_path is None:
        output_path = (
            paths.RESULTS_EXPERIMENT3_DIR
            / "country_stability_interactive.html"
        )
    else:
        output_path = Path(output_path)

    if csv_output_path is None:
        csv_output_path = (
            paths.RESULTS_EXPERIMENT3_DIR
            / "linear_regression_country_stability.csv"
        )
    else:
        csv_output_path = Path(csv_output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Automatically detect similarity column name (with underscore or space)
    sim_col = (
        "Cosine_similarity"
        if "Cosine_similarity" in country_year_df.columns
        else "Cosine similarity"
    )

    # 3. CALCULATE country_stability_df FROM country_year_df
    country_stability_df = (
        country_year_df.groupby(["country_code_3", "country_name"])
        .agg(
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std"),
            Similarity_Mean=(sim_col, "mean"),
            Similarity_Std=(sim_col, "std"),
        )
        .reset_index()
    )

    # 💾 4. SAVE STABILITY STATISTICS TO CSV
    country_stability_df.to_csv(csv_output_path, index=False, encoding="utf-8")
    print(f"📁 Saved stability statistics file: {csv_output_path.name}")

    # 5. CREATE INTERACTIVE SCATTER PLOT WITH PLOTLY
    fig = px.scatter(
        country_stability_df,
        x="RMSE_Mean",
        y="RMSE_Std",
        hover_name="country_name",
        hover_data={
            "country_code_3": True,
            "RMSE_Mean": ":.4f",
            "RMSE_Std": ":.4f",
            "Similarity_Mean": ":.4f",
        },
        title="<b>Forecast Accuracy & Stability Analysis by Country</b>",
        labels={
            "RMSE_Mean": "Mean RMSE (Lower is more accurate)",
            "RMSE_Std": "RMSE Std Dev (Lower is more stable)",
            "Similarity_Mean": "Mean Similarity",
            "country_code_3": "Country Code",
        },
        template="plotly_white",
    )

    # 6. Customize markers
    fig.update_traces(
        marker=dict(
            size=marker_size,
            opacity=0.7,
            line=dict(width=1, color="black"),
        )
    )

    # Add 2 median lines dividing into 4 quadrants
    median_x = country_stability_df["RMSE_Mean"].median()
    median_y = country_stability_df["RMSE_Std"].median()

    fig.add_vline(
        x=median_x, line_dash="dash", line_color="red", opacity=0.6
    )  # Split by Mean
    fig.add_hline(
        y=median_y, line_dash="dash", line_color="green", opacity=0.6
    )  # Split by Std Dev

    # 7. Zoom into dense data region
    if x_max is not None:
        fig.update_xaxes(range=[0, x_max])
    if y_max is not None:
        fig.update_yaxes(range=[0, y_max])

    # 💾 8. SAVE INTERACTIVE HTML FILE
    fig.write_html(str(output_path))
    print(f"📁 Interactive HTML chart saved to: {output_path.name}")

    if show_fig:
        fig.show()

    return country_stability_df


if __name__ == "__main__":
    # 1. Load dataset splits for Experiment 3 (20-indicator forecast)
    (
        X_train_exp3,
        y_train_exp3,
        X_val_exp3,
        y_val_exp3,
        X_test_exp3,
        y_test_exp3,
    ) = load_experiment_3()

    # 2. Run Auto-Benchmark comparing 9 Multi-Output Regression models on the Validation set
    print("\n🚀 Running Auto-Benchmark for Experiment 3...")
    benchmark_results_df = run_auto_benchmark_experiment_3(
        X_train=X_train_exp3,
        Y_train=y_train_exp3,
        X_val=X_val_exp3,
        Y_val=y_val_exp3,
        plot_results=True,
    )

    # 3. Evaluate Linear Regression model on Test set
    test_results_df, test_feature_details_df, country_year_df = (
        evaluate_linear_regression_on_test(
            X_train=X_train_exp3,
            Y_train=y_train_exp3,
            X_test=X_test_exp3,
            Y_test=y_test_exp3,
            plot_results=True,
        )
    )

    # 1. Save Boxplot chart for Cosine Similarity by year
    plot_global_yearly_boxplot(country_year_df, metric="Cosine_similarity")

    # 2. Save Boxplot chart for RMSE by year
    plot_global_yearly_boxplot(country_year_df, metric="RMSE")

    country_stability_df = plot_country_stability_scatter_chart(
        country_year_df
    )