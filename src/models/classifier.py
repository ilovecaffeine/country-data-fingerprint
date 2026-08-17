# src/models/classifier.py
# cd "C:\Users\admin\Documents\Code_for_fun\country-data-fingerprint"
# python -m src.models.classifier

from config import paths
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

# Trực tiếp import các thư viện Boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def load_experiment_2_country_classification(
    data_dir: Path | None = None,
    encode_target: bool = True
):
    """
    Tải dữ liệu cho Experiment 2 với Target = 'country_code_3'.
    
    Target (y)   : country_code_3 (Mã quốc gia)
    Features (X) : Tất cả các chỉ số kinh tế - xã hội (Đã loại bỏ country_code_3, country_name, year)
    """
    if data_dir is None:
        data_dir = paths.PROCESSED_DATA_EXPERIMENT2_DIR

    # 1. ĐỌC 3 FILE CSV
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    # 2. XÁC ĐỊNH TARGET VÀ CÁC CỘT BẮT BUỘC LOẠI BỎ KHỎI X
    target_col = "country_code_3"
    
    # ⚠️ QUAN TRỌNG: Loại bỏ country_name và year khỏi X để tránh Data Leakage
    cols_to_exclude = ["country_code_3", "country_name", "year"]

    # 3. DANH SÁCH CỘT ĐẶC TRƯNG X
    feature_cols = [col for col in train_df.columns if col not in cols_to_exclude]

    # 4. TÁCH X VÀ y
    X_train_exp2 = train_df[feature_cols]
    y_train_exp2 = train_df[target_col]

    X_val_exp2 = val_df[feature_cols]
    y_val_exp2 = val_df[target_col]

    X_test_exp2 = test_df[feature_cols]
    y_test_exp2 = test_df[target_col]

    # 5. MÃ HÓA NHÃN CỦA Y THÀNH SỐ NGUYÊN (0, 1, 2,..., 193)
    label_encoder = None
    if encode_target:
        label_encoder = LabelEncoder()
        
        # Fit mã hóa trên tập Train
        y_train_exp2 = label_encoder.fit_transform(y_train_exp2)
        y_val_exp2 = label_encoder.transform(y_val_exp2)
        y_test_exp2 = label_encoder.transform(y_test_exp2)

    # 6. IN THÔNG TIN KIỂM TRA
    print("=======================================================")
    print("EXPERIMENT 2: PHÂN LOẠI QUỐC GIA (TARGET = country_code_3)")
    print("=======================================================")
    print(f"X_train_exp2 : {X_train_exp2.shape}  |  y_train_exp2 : {y_train_exp2.shape}")
    print(f"X_val_exp2   : {X_val_exp2.shape}  |  y_val_exp2   : {y_val_exp2.shape}")
    print(f"X_test_exp2  : {X_test_exp2.shape}  |  y_test_exp2  : {y_test_exp2.shape}")
    print(f"Số lượng Lớp phân loại (Số quốc gia): {len(train_df[target_col].unique())}")
    print(f"Số lượng Đặc trưng (Features X)     : {len(feature_cols)}")

    return (
        X_train_exp2, y_train_exp2,
        X_val_exp2, y_val_exp2,
        X_test_exp2, y_test_exp2,
        label_encoder
    )

def run_auto_benchmark_experiment_2(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    plot_results: bool = True,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Tự động huấn luyện và so sánh hiệu năng nhiều Classifiers trên tập
    Validation, lưu kết quả ra CSV và ảnh biểu đồ.
    """
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT2_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    classes_list = np.unique(y_train)

    # 1. KHỞI TẠO DANH SÁCH TẤT CẢ MÔ HÌNH
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
        # 🌟 ĐIỀU CHỈNH THÔNG SỐ LIGHTGBM
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=2,  # Cho phép lá cây chứa từ 2 mẫu
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        # 🌟 ĐIỀU CHỈNH THÔNG SỐ XGBOOST
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=1,  # Cho phép nút nhỏ tách nhánh
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
    print(f"🚀 BẮT ĐẦU AUTO-BENCHMARK ({len(models)} MÔ HÌNH)")
    print(
        f"   Train: {X_train.shape[0]} mẫu | Val: {X_val.shape[0]} mẫu | Số lớp: {len(classes_list)}"
    )
    print("=======================================================\n")

    # 2. VÒNG LẶP HUẤN LUYỆN VÀ ĐÁNH GIÁ
    for name, model in models.items():
        # Fit mô hình
        model.fit(X_train, y_train)

        # Predict nhãn (Top-1)
        y_val_pred = model.predict(X_val)

        # Predict xác suất (để tính Top-5 Accuracy)
        try:
            y_val_proba = model.predict_proba(X_val)
            top5_acc = top_k_accuracy_score(
                y_val, y_val_proba, k=5, labels=classes_list
            )
        except Exception:
            top5_acc = np.nan

        # Tính chỉ số
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
            f"✔️ {name:20s} | Acc: {acc:.4f} | Top-5 Acc: {top5_acc:.4f} | F1: {f1_weighted:.4f}"
        )

    # 3. DATAFRAME KẾT QUẢ SẮP XẾP THEO VAL ACCURACY
    results_df = pd.DataFrame(results).sort_values(
        "Val_Accuracy", ascending=False
    ).reset_index(drop=True)

    print(
        f"\n{'='*25} 🏆 BẢNG XẾP HẠNG MÔ HÌNH {'='*25}\n",
        results_df.to_string(index=False),
    )

    # 💾 1. LƯU BẢNG KẾT QUẢ RA FILE CSV
    csv_path = output_dir / "auto_benchmark_validation_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n📁 Đã lưu file kết quả CSV: {csv_path.name}")

    # 4. VẼ VÀ LƯU BIỂU ĐỒ
    if plot_results:
        ax = results_df.plot(
            x="Model",
            y=["Val_Accuracy", "Val_Top5_Acc", "Val_F1_Weighted"],
            kind="barh",
            figsize=(10, 7),
            color=["#3498db", "#2ecc71", "#e74c3c"],
            width=0.7,
        )

        # Định dạng biểu đồ
        ax.invert_yaxis()
        ax.set(
            xlim=(0, 1.15),
            xlabel="Score",
            title="Model Performance Comparison (Validation Set)",
        )

        # Legend đặt dưới biểu đồ, xếp 3 nhãn nằm ngang
        ax.legend(
            ["Val Accuracy (Top-1)", "Val Top-5 Accuracy", "Val F1 (Weighted)"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            frameon=False,
        )

        # Tự động gán nhãn giá trị lên các cột
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

        plt.tight_layout()

        # 🖼️ 2. LƯU ẢNH BIỂU ĐỒ BENCHMARK
        img_path = output_dir / "auto_benchmark_validation_comparison.png"
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Đã lưu ảnh biểu đồ: {img_path.name}")

        plt.show()

    print(f"\n=======================================================")
    print(f"🎉 TẤT CẢ FILE CSV VÀ ẢNH BENCHMARK ĐÃ ĐƯỢC LƯU TẠI: {output_dir}")
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
    """Đánh giá chuyên sâu mô hình Extra Trees trên tập Test:

    1. Bảng kết quả Test_Accuracy, Test_Top5_Acc, Test_F1_Weighted. 
    2. Trích xuất, xuất CSV và lưu ảnh biểu đồ tất cả đặc trưng quan trọng. 
    3. Phân tích chi tiết danh sách các quốc gia bị đoán nhầm (Error Analysis), xuất CSV và lưu ảnh Heatmap.
    """
    # Xử lý thư mục lưu kết quả CSV và Ảnh
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT2_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    classes_list = np.unique(y_train)

    # =============================================================
    # 1. HUẤN LUYỆN VÀ TÍNH CHỈ SỐ ĐÁNH GIÁ TẬP TEST
    # =============================================================
    print("=======================================================")
    print("🎯 ĐÁNH GIÁ MÔ HÌNH EXTRA TREES TRÊN TẬP TEST (2021-2024)")
    print("=======================================================\n")

    name = "Extra Trees"
    model = ExtraTreesClassifier(
        n_estimators=100, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Dự báo nhãn và xác suất
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    # Tính toán chỉ số
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
    
    # 💾 1. LƯU BẢNG KẾT QUẢ TEST RA CSV
    results_csv_path = output_dir / "extra_trees_test_results.csv"
    results_df.to_csv(results_csv_path, index=False, encoding="utf-8")
    
    print("📊 BẢNG KẾT QUẢ ĐÁNH GIÁ TẬP TEST:")
    print(results_df.to_string(index=False))
    print(f"📁 Đã lưu CSV: {results_csv_path.name}")

    # =============================================================
    # 2. TRÍCH XUẤT, LƯU CSV VÀ LƯU ẢNH FEATURE IMPORTANCES
    # =============================================================
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    # 💾 2. LƯU FEATURE IMPORTANCES RA CSV
    importances_df = importances.reset_index()
    importances_df.columns = ["Feature", "Importance"]
    importances_csv_path = output_dir / "extra_trees_feature_importances.csv"
    importances_df.to_csv(importances_csv_path, index=False, encoding="utf-8")

    print(f"\n🔥 TẤT CẢ {len(importances)} CHỈ SỐ QUAN TRỌNG ĐỂ ĐỊNH DANH QUỐC GIA:")
    print(importances.to_string())
    print(f"📁 Đã lưu CSV: {importances_csv_path.name}")

    # Tự động tính chiều cao biểu đồ
    fig_height = max(6, len(importances) * 0.35)

    # Vẽ biểu đồ tất cả đặc trưng
    plt.figure(figsize=(10, fig_height))
    ax = importances.plot(kind="barh", color="#2ecc71")

    # Hiển thị giá trị cụ thể ở đầu mỗi thanh
    ax.bar_label(ax.containers[0], fmt="%.4f", padding=3, fontsize=9)
    plt.title("All Feature Importances (Extra Trees Model)", fontsize=12, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=11)
    plt.xlim(0, importances.max() * 1.12)
    plt.gca().invert_yaxis()  # Đưa đặc trưng quan trọng nhất lên đầu
    plt.tight_layout()

    # 🖼️ LƯU ẢNH BIỂU ĐỒ FEATURE IMPORTANCE
    importances_img_path = output_dir / "extra_trees_feature_importances.png"
    plt.savefig(importances_img_path, dpi=300, bbox_inches="tight")
    print(f"🖼️ Đã lưu ảnh biểu đồ: {importances_img_path.name}")
    plt.show()

    # =============================================================
    # 3. PHÂN TÍCH LỖI DỰ BÁO (ERROR ANALYSIS)
    # =============================================================
    y_test_true_codes = label_encoder.inverse_transform(y_test)
    y_test_pred_codes = label_encoder.inverse_transform(y_test_pred)

    # 1. Tạo mặt nạ lọc các mẫu bị đoán sai
    errors_mask = y_test_true_codes != y_test_pred_codes

    # 2. Tạo DataFrame chứa danh sách lỗi
    error_df = pd.DataFrame({
        "Real_Country": y_test_true_codes[errors_mask],
        "Predicted_Country": y_test_pred_codes[errors_mask],
    })

    print(f"\n⚠️ TỔNG SỐ MẪU ĐOÁN SAI: {len(error_df)} / {len(y_test_true_codes)}")

    if not error_df.empty:
        # Gom nhóm đếm số lượng từng cặp lỗi
        summary_df = error_df.value_counts().reset_index(name="Count")
        
        # 💾 3. LƯU BÁO CÁO CÁC CẶP BỊ ĐOÁN SAI RA CSV
        error_summary_csv_path = output_dir / "extra_trees_misclassification_summary.csv"
        summary_df.to_csv(error_summary_csv_path, index=False, encoding="utf-8")

        # 3. Tạo ma trận đếm số lượng lỗi bằng pd.crosstab
        pivot_df = pd.crosstab(
            index=error_df["Real_Country"], columns=error_df["Predicted_Country"]
        )

        # 💾 4. LƯU MA TRẬN PIVOT LỖI RA CSV
        error_matrix_csv_path = output_dir / "extra_trees_misclassification_matrix.csv"
        pivot_df.to_csv(error_matrix_csv_path, encoding="utf-8")

        # 4. Vẽ Heatmap
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
        plt.title("Misclassification Heatmap (2021 - 2024)", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Country", fontsize=10)
        plt.ylabel("Real Country", fontsize=10)
        plt.tight_layout()

        # 🖼️ LƯU ẢNH BIỂU ĐỒ HEATMAP LỖI
        heatmap_img_path = output_dir / "extra_trees_misclassification_heatmap.png"
        plt.savefig(heatmap_img_path, dpi=300, bbox_inches="tight")
        print(f"🖼️ Đã lưu ảnh biểu đồ: {heatmap_img_path.name}")
        plt.show()

        print("\nChi tiết các quốc gia bị đoán nhầm:")
        print(summary_df.to_string(index=False))
        print(f"📁 Đã lưu CSV: {error_summary_csv_path.name}")
        print(f"📁 Đã lưu CSV: {error_matrix_csv_path.name}")
    else:
        summary_df = pd.DataFrame(columns=["Real_Country", "Predicted_Country", "Count"])
        summary_df.to_csv(output_dir / "extra_trees_misclassification_summary.csv", index=False, encoding="utf-8")
        print("🎉 Tuyệt vời! Không có mẫu nào bị đoán sai trên tập Test.")

    print(f"\n=======================================================")
    print(f"🎉 TẤT CẢ FILE CSV VÀ ẢNH BIỂU ĐỒ ĐÃ ĐƯỢC LƯU TẠI: {output_dir}")
    print("=======================================================")

    return results_df, importances, error_df


if __name__ == "__main__":
    (
        X_train_exp2, y_train_exp2,
        X_val_exp2, y_val_exp2,
        X_test_exp2, y_test_exp2,
        le,
    ) = load_experiment_2_country_classification(encode_target=True)
    #print(X_train_exp2.columns.tolist())
    
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

'''
    # 2. XEM HEAD CỦA X_TRAIN & Y_TRAIN
    print("\n📌 --- 1. X_train_exp2 (5 dòng đầu) ---")
    print(X_train_exp2.head())

    print("\n📌 --- 2. y_train_exp2 (5 phần tử đầu) ---")
    print("Dạng số nguyên  :", y_train_exp2[:5])
    print("Mã quốc gia thực:", le.inverse_transform(y_train_exp2[:5]))

    # 3. XEM HEAD CỦA X_VAL & Y_VAL
    print("\n📌 --- 3. X_val_exp2 (5 dòng đầu) ---")
    print(X_val_exp2.head())

    print("\n📌 --- 4. y_val_exp2 (5 phần tử đầu) ---")
    print("Dạng số nguyên  :", y_val_exp2[:5])
    print("Mã quốc gia thực:", le.inverse_transform(y_val_exp2[:5]))

    # 4. XEM HEAD CỦA X_TEST & Y_TEST
    print("\n📌 --- 5. X_test_exp2 (5 dòng đầu) ---")
    print(X_test_exp2.head())

    print("\n📌 --- 6. y_test_exp2 (5 phần tử đầu) ---")
    print("Dạng số nguyên  :", y_test_exp2[:5])
    print("Mã quốc gia thực:", le.inverse_transform(y_test_exp2[:5]))
'''
