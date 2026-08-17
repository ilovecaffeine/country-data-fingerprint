# src/models/forecast.py
# cd "C:\Users\admin\Documents\Code_for_fun\country-data-fingerprint"
# python -m src.models.forecast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from config import paths

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def load_experiment_3(
    data_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Đọc 3 file CSV (train.csv, validation.csv, test.csv) từ PROCESSED_DATA_EXPERIMENT3_DIR
    và tách ra các bộ Feature (X) và Target (Y) cho bài toán Forecast 20 chỉ số.

    Returns:
        X_train_exp3, y_train_exp3,
        X_val_exp3, y_val_exp3,
        X_test_exp3, y_test_exp3
    """
    if data_dir is None:
        data_dir = paths.PROCESSED_DATA_EXPERIMENT3_DIR

    # 1. ĐỌC 3 FILE CSV
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"[ERROR] Không tìm thấy file dữ liệu Exp 3: {p}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # 2. TỰ ĐỘNG LỌC CỘT ID, FEATURES (X) VÀ TARGETS (Y)
    id_cols = ["country_code_3", "country_name", "year"]
    
    # Cột Target (Y) là những cột có đuôi '_target_next_year'
    target_cols = [col for col in train_df.columns if col.endswith("_target_next_year")]
    
    # Cột Features (X) là những cột chỉ số năm t (loại trừ các cột ID và cột Target)
    feature_cols = [col for col in train_df.columns if col not in id_cols and col not in target_cols]

    # 3. TÁCH X VÀ Y CHO TỪNG TẬP
    X_train_exp3 = train_df[feature_cols]
    y_train_exp3 = train_df[target_cols]

    X_val_exp3 = val_df[feature_cols]
    y_val_exp3 = val_df[target_cols]

    X_test_exp3 = test_df[feature_cols]
    y_test_exp3 = test_df[target_cols]

    # 4. IN THÔNG TIN KIỂM TRA
    print("=======================================================")
    print("KẾT QUẢ TẢI VÀ TÁCH DỮ LIỆU EXPERIMENT 3 (FORECASTING)")
    print("=======================================================")
    print(f"X_train_exp3 : {X_train_exp3.shape}  |  y_train_exp3 : {y_train_exp3.shape}")
    print(f"X_val_exp3   : {X_val_exp3.shape}  |  y_val_exp3   : {y_val_exp3.shape}")
    print(f"X_test_exp3  : {X_test_exp3.shape}  |  y_test_exp3  : {y_test_exp3.shape}")
    print(f"Số lượng Cột Đầu vào (Features X): {len(feature_cols)}")
    print(f"Số lượng Cột Đầu ra (Targets Y)  : {len(target_cols)}")

    return (
        X_train_exp3, y_train_exp3,
        X_val_exp3, y_val_exp3,
        X_test_exp3, y_test_exp3
    )



def run_auto_benchmark_experiment_3(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    plot_results: bool = True,
    output_dir: Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tự động huấn luyện và so sánh 9 phương pháp Hồi quy Đa đầu ra (Multi-Output Forecast 20 chỉ số)
    trên tập Validation. Xuất file CSV xếp hạng chung, file chi tiết từng Feature và lưu 2 ảnh biểu đồ (Comparison + Heatmap R2).
    """
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT3_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. KHỞI TẠO TẤT CẢ 9 MÔ HÌNH
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
    print(f"🚀 BẮT ĐẦU AUTO-BENCHMARK EXP 3 ({len(models)} PHƯƠNG PHÁP HỒI QUY 20 CHỈ SỐ)")
    print(
        f"   Train: {X_train.shape[0]} mẫu | Val: {X_val.shape[0]} mẫu | Số Output Target: {Y_train.shape[1]}"
    )
    print("=======================================================\n")

    # 2. VÒNG LẶP HUẤN LUYỆN VÀ ĐÁNH GIÁ TRÊN TẬP VALIDATION
    for name, model in models.items():
        # Fit mô hình
        model.fit(X_train, Y_train)

        # Predict Y trên tập Validation
        Y_val_pred = model.predict(X_val)

        # A. TÍNH CHỈ SỐ TRUNG BÌNH TỔNG THỂ (Overall Means)
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

        # B. TÍNH R2, MAE, RMSE CHI TIẾT CHO TỪNG FEATURE (Per-feature Metrics)
        r2_raw = r2_score(Y_val, Y_val_pred, multioutput="raw_values")
        mae_raw = mean_absolute_error(Y_val, Y_val_pred, multioutput="raw_values")
        rmse_raw = np.sqrt(mean_squared_error(Y_val, Y_val_pred, multioutput="raw_values"))

        for col_name, r2_v, mae_v, rmse_v in zip(Y_val.columns, r2_raw, mae_raw, rmse_raw):
            # Làm sạch tên feature (bỏ đuôi _target_next_year)
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
            f"✔️ {name:30s} | R2: {mean_r2:.4f} | MAE: {mean_mae:.4f} | RMSE: {mean_rmse:.4f}"
        )

    # 3. DATAFRAME TỔNG HỢP & CHI TIẾT
    results_df = pd.DataFrame(results).sort_values(
        "Val_Mean_R2", ascending=False
    ).reset_index(drop=True)

    feature_details_df = pd.DataFrame(feature_details)

    print(
        f"\n{'='*25} 🏆 BẢNG XẾP HẠNG MÔ HÌNH EXP 3 (VALIDATION) {'='*25}\n",
        results_df.to_string(index=False),
    )

    # 💾 1. LƯU FILE CSV KẾT QUẢ BENCHMARK TỔNG HỢP
    csv_path = output_dir / "forecast_auto_benchmark_val_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n📁 Đã lưu file kết quả tổng hợp: {csv_path.name}")

    # 💾 2. LƯU FILE CSV CHI TIẾT ĐIỂM SỐ TỪNG FEATURE
    feature_csv_path = output_dir / "forecast_auto_benchmark_val_feature_details.csv"
    feature_details_df.to_csv(feature_csv_path, index=False, encoding="utf-8")
    print(f"📁 Đã lưu file chi tiết từng Feature: {feature_csv_path.name}")

    # 4. PLOT & SAVE CHARTS
    if plot_results:
        # --- BIỂU ĐỒ 1: CỘT NGANG SO SÁNH R2 TỔNG THỂ ---
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

        # 🖼️ LƯU ẢNH BIỂU ĐỒ CỘT NGANG
        img_path = output_dir / "forecast_auto_benchmark_val_comparison.png"
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Đã lưu ảnh biểu đồ so sánh: {img_path.name}")
        plt.show()

        # --- BIỂU ĐỒ 2: HEATMAP MA TRẬN R2 SCORE (FEATURES VS MODELS) ---
        pivot_r2 = feature_details_df.pivot(
            index="Feature", columns="Model", values="Val_R2"
        )
        # Sắp xếp các Feature theo điểm R2 trung bình từ cao xuống thấp
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

        # 🖼️ LƯU ẢNH BIỂU ĐỒ HEATMAP R2
        heatmap_img_path = output_dir / "forecast_auto_benchmark_val_r2_heatmap.png"
        plt.savefig(heatmap_img_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Đã lưu ảnh biểu đồ Heatmap R²: {heatmap_img_path.name}")
        plt.show()

    print(f"\n=======================================================")
    print(f"🎉 TẤT CẢ FILE CSV VÀ ẢNH BIỂU ĐỒ ĐÃ ĐƯỢC LƯU TẠI: {output_dir}")
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
    """Đánh giá chuyên sâu mô hình Linear Regression trên tập TEST (2021-2024):

    1. Bảng kết quả tổng thể: Model, Test_Mean_R2, Test_Mean_MAE, Test_Mean_RMSE.
    2. Bảng chi tiết từng Feature: Feature, Test_R2, Test_MAE, Test_RMSE.
    3. Bảng sai số từng Quốc gia - từng Năm: country_code_3, country_name, year, MAE, RMSE, Euclidean dis, Cosine similarity.
    4. Xuất 3 file CSV và lưu ảnh biểu đồ R2.
    """
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT3_DIR

    if test_csv_file is None:
        test_csv_file = paths.PROCESSED_DATA_EXPERIMENT3_DIR / "test.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("🎯 ĐÁNH GIÁ MÔ HÌNH LINEAR REGRESSION TRÊN TẬP TEST (2021-2024)")
    print("=======================================================\n")

    # 1. HUẤN LUYỆN MÔ HÌNH LINEAR REGRESSION
    model_name = "Linear Regression"
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # 2. DỰ BÁO TRÊN TẬP TEST
    Y_test_pred = model.predict(X_test)

    # 3. CHỈ SỐ TỔNG THỂ (OVERALL METRICS)
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

    print("📊 1. BẢNG TỔNG HỢP HIỆU NĂNG TẬP TEST:")
    print(test_results_df.to_string(index=False))

    # 💾 LƯU BẢNG TỔNG HỢP RA CSV
    test_results_csv = output_dir / "linear_regression_test_results.csv"
    test_results_df.to_csv(test_results_csv, index=False, encoding="utf-8")
    print(f"\n📁 Đã lưu file CSV tổng hợp: {test_results_csv.name}")

    # 4. CHỈ SỐ CHI TIẾT THEO TỪNG FEATURE (PER-FEATURE METRICS)
    r2_raw = r2_score(Y_test, Y_test_pred, multioutput="raw_values")
    mae_raw = mean_absolute_error(Y_test, Y_test_pred, multioutput="raw_values")
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

    print("\n🔥 2. BẢNG CHI TIẾT CÁC CHỈ SỐ TRÊN TẬP TEST CHO TỪNG FEATURE:")
    print(test_feature_details_df.to_string(index=False))

    # 💾 LƯU BẢNG CHI TIẾT TỪNG FEATURE RA CSV
    feature_csv = output_dir / "linear_regression_test_feature_details.csv"
    test_feature_details_df.to_csv(feature_csv, index=False, encoding="utf-8")
    print(f"\n📁 Đã lưu file CSV chi tiết từng Feature: {feature_csv.name}")

    # 5. TÍNH BẢNG SAI SỐ THEO TỪNG QUỐC GIA - TỪNG NĂM (COUNTRY-YEAR ERROR METRICS)
    country_year_df = pd.DataFrame()
    if test_csv_file.exists():
        test_meta = pd.read_csv(test_csv_file)[
            ["country_code_3", "country_name", "year"]
        ]

        Y_true_vals = Y_test.values
        Y_pred_vals = Y_test_pred

        # Tính toán theo từng dòng (từng mẫu quốc gia - năm trên 20 đặc trưng)
        # a) MAE hàng
        row_mae = np.mean(np.abs(Y_true_vals - Y_pred_vals), axis=1)

        # b) RMSE hàng
        row_rmse = np.sqrt(np.mean((Y_true_vals - Y_pred_vals) ** 2, axis=1))

        # c) Khoảng cách Euclidean thực tế và dự báo
        row_euclidean = np.sqrt(np.sum((Y_true_vals - Y_pred_vals) ** 2, axis=1))

        # d) Độ tương đồng Cosine thực tế và dự báo
        dot_product = np.sum(Y_true_vals * Y_pred_vals, axis=1)
        norm_true = np.linalg.norm(Y_true_vals, axis=1)
        norm_pred = np.linalg.norm(Y_pred_vals, axis=1)
        row_cosine_sim = dot_product / (norm_true * norm_pred + 1e-12)

        country_year_df = pd.DataFrame(
            {
                "country_code_3": test_meta["country_code_3"],
                "country_name": test_meta["country_name"],
                "year": test_meta["year"] + 1,  # Năm t + 1 = 2021 đến 2024
                "MAE": row_mae,
                "RMSE": row_rmse,
                "Euclidean_distance": row_euclidean,
                "Cosine_similarity": row_cosine_sim,
            }
        ).sort_values(["country_code_3", "year"]).reset_index(drop=True)

        # 💾 LƯU BẢNG SAI SỐ THEO QUỐC GIA - NĂM RA CSV
        country_year_csv = output_dir / "linear_regression_test_country_year_errors.csv"
        country_year_df.to_csv(country_year_csv, index=False, encoding="utf-8")

        print("\n📍 3. BẢNG SAI SỐ CHI TIẾT THEO QUỐC GIA - NĂM 2021-2024 (HEAD 10):")
        print(country_year_df.head(10).to_string(index=False))
        print(f"\n📁 Đã lưu file CSV sai số theo Quốc gia - Năm: {country_year_csv.name}")

    # 6. VẼ VÀ LƯU BIỂU ĐỒ R2 SCORE TRÊN TẬP TEST
    if plot_results:
        plt.figure(figsize=(10, 8))
        bars = plt.barh(
            test_feature_details_df["Feature"][::-1],
            test_feature_details_df["Test_R2"][::-1],
            color="#2ecc71",
            height=0.6,
        )

        plt.title(
            "Linear Regression - Test R² Score across 20 Indicators (2021-2024)",
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

        # 🖼️ LƯU ẢNH BIỂU ĐỒ
        img_path = output_dir / "linear_regression_test_feature_r2.png"
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Đã lưu ảnh biểu đồ Test R²: {img_path.name}")

        plt.show()

    print(f"\n=======================================================")
    print(
        f"🎉 TẤT CẢ FILE KẾT QUẢ TEST LINEAR REGRESSION ĐÃ ĐƯỢC LƯU TẠI: {output_dir}"
    )
    print("=======================================================")

    return test_results_df, test_feature_details_df, country_year_df

def plot_global_yearly_boxplot(
    country_year_df: pd.DataFrame,
    metric: str = "Cosine_similarity",
    save_path: Path | str | None = None,
):
    """Plots a boxplot showing the distribution of scores across all countries by year.

    Auto-saves to paths.RESULTS_EXPERIMENT3_DIR if save_path is None.
    """
    # 1. Tự động xác định đường dẫn lưu mặc định nếu save_path là None
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

    # Tự động tạo thư mục nếu chưa có
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Vẽ biểu đồ Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=country_year_df,
        x="year",
        y=metric,
        hue="year",  # Gán biến x vào hue để tránh cảnh báo deprecation
        palette="Set2",
        legend=False,  # Tắt chú thích legend thừa
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

    # 3. Lưu hình ảnh và hiển thị
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ Boxplot image saved to: {save_path.name}")

    plt.show()


def plot_country_stability_scatter_chart(
    country_year_df: pd.DataFrame,
    output_path: Path | str | None = None,
    csv_output_path: Path | str | None = None,
    show_fig: bool = True,
    marker_size: int = 10,
    x_max: float | None = 2.5,  # Giới hạn trục X để loại bỏ ngoại lệ ZWE
    y_max: float | None = 1.5,  # Giới hạn trục Y
) -> pd.DataFrame:
    """Tự động tính toán country_stability_df, lưu file CSV thống kê độ ổn định

    và tạo/lưu biểu đồ Plotly Scatter Plot tương tác theo Quốc gia.

    Parameters
    ----------
    country_year_df : pd.DataFrame
        DataFrame chứa các cột: country_code_3, country_name, year, RMSE, Cosine_similarity (hoặc Cosine similarity).
    output_path : Path | str, optional
        Đường dẫn lưu file HTML tương tác. Mặc định: country_stability_interactive.html
    csv_output_path : Path | str, optional
        Đường dẫn lưu file CSV thống kê độ ổn định. Mặc định: linear_regression_country_stability.csv
    show_fig : bool, default=True
        Có hiển thị biểu đồ trực tiếp trên Jupyter/Colab hay không.
    marker_size : int, default=10
        Kích thước cố định của các điểm marker.
    x_max : float | None, default=2.5
        Giới hạn trên của trục X.
    y_max : float | None, default=1.5
        Giới hạn trên của trục Y.

    Returns
    -------
    pd.DataFrame
        Trả về country_stability_df đã được gom nhóm và tính toán.
    """
    # 1. Tự động xác định đường dẫn lưu file HTML và CSV
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

    # 2. Tự động nhận diện tên cột Similarity (có gạch dưới hoặc khoảng trắng)
    sim_col = (
        "Cosine_similarity"
        if "Cosine_similarity" in country_year_df.columns
        else "Cosine similarity"
    )

    # 3. TÍNH TOÁN country_stability_df TỪ country_year_df
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

    # 💾 4. LƯU BẢNG THỐNG KÊ ĐỘ ỔN ĐỊNH RA FILE CSV
    country_stability_df.to_csv(
        csv_output_path, index=False, encoding="utf-8"
    )
    print(f"📁 Đã lưu file thống kê độ ổn định: {csv_output_path.name}")

    # 5. TẠO BIỂU ĐỒ SCATTER PLOT TƯƠNG TÁC BẰNG PLOTLY
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

    # 6. Tùy chỉnh markers
    fig.update_traces(
        marker=dict(
            size=marker_size,
            opacity=0.7,
            line=dict(width=1, color="black"),
        )
    )

    # Thêm 2 đường trung vị phân chia 4 góc phần tư
    median_x = country_stability_df["RMSE_Mean"].median()
    median_y = country_stability_df["RMSE_Std"].median()

    fig.add_vline(
        x=median_x, line_dash="dash", line_color="red", opacity=0.6
    )  # Phân chia theo Mean
    fig.add_hline(
        y=median_y, line_dash="dash", line_color="green", opacity=0.6
    )  # Phân chia theo Std Dev

    # 7. Zoom vào vùng dữ liệu tập trung
    if x_max is not None:
        fig.update_xaxes(range=[0, x_max])
    if y_max is not None:
        fig.update_yaxes(range=[0, y_max])

    # 💾 8. LƯU FILE HTML TƯƠNG TÁC
    fig.write_html(str(output_path))
    print(f"📁 Interactive HTML chart saved to: {output_path.name}")

    if show_fig:
        fig.show()

    return country_stability_df

if __name__ == "__main__":
    # 1. Tải dữ liệu các tập cho Experiment 3 (Forecast 20 chỉ số)
    (
        X_train_exp3, y_train_exp3,
        X_val_exp3, y_val_exp3,
        X_test_exp3, y_test_exp3
    ) = load_experiment_3()

    # 2. Khởi chạy Auto-Benchmark so sánh 9 mô hình Hồi quy Đa đầu ra trên tập Validation
    print("\n🚀 Đang chạy Auto-Benchmark cho Experiment 3...")
    benchmark_results_df = run_auto_benchmark_experiment_3(
        X_train=X_train_exp3,
        Y_train=y_train_exp3,
        X_val=X_val_exp3,
        Y_val=y_val_exp3,
        plot_results=True
    )

    # 3. Đánh giá Linear Regression trên tập Test
    test_results_df, test_feature_details_df, country_year_df = evaluate_linear_regression_on_test(
        X_train=X_train_exp3,
        Y_train=y_train_exp3,
        X_test=X_test_exp3,
        Y_test=y_test_exp3,
        plot_results=True
    )
    # 1. Lưu biểu đồ Boxplot cho Cosine Similarity theo năm
    plot_global_yearly_boxplot(
        country_year_df, 
        metric="Cosine_similarity"
    )
    # 2. Lưu biểu đồ Boxplot cho RMSE theo năm
    plot_global_yearly_boxplot(
        country_year_df,
        metric="RMSE"
    )

    country_stability_df = plot_country_stability_scatter_chart(country_year_df)
