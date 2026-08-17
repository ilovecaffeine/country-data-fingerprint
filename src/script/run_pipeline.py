# src/script/run_pipeline.py
# cd "C:\Users\admin\Documents\Code_for_fun\country-data-fingerprint"
# python -m src.script.run_pipeline

import argparse
import sys
from pathlib import Path
import pandas as pd

from config import paths
from src.data.load import build_and_populate_all_draft_data
from src.data.transform import (
	impute_all_worldbank_raw_files,
	impute_all_ourworldindata_raw_files,
	combine_all_drafts_to_panel_data,
	preprocess_and_export_all_experiment_1_years,
	preprocess_and_split_experiment_2,
	preprocess_and_split_experiment_3
)
from src.distance_matrices.distance_matrices import (
    export_experiment_1_distance_matrices,
    export_experiment_1_cosine_matrices,
    generate_euclidean_neighbors_summary,
    generate_cosine_neighbors_summary
)
from src.hierarchical_edge_bundling.export_heb import (
    export_all_years_distance_heb_plots,
    export_all_years_cosine_heb_plots,
    continent_map
)
from src.models.classifier import (
    load_experiment_2_country_classification,
    run_auto_benchmark_experiment_2,
    evaluate_extra_trees_on_test
)

from src.models.forecast import (
    load_experiment_3,
    run_auto_benchmark_experiment_3,
    evaluate_linear_regression_on_test,
    plot_global_yearly_boxplot,
    plot_country_stability_scatter_chart
)

from src.models.forecast_then_classify import (
	run_forecast_then_classify_pipeline, plot_misclassification_comparison
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
    """Kiểm tra độ toàn vẹn của bảng draft_panel_2010_2024.csv:
    1. File không được rỗng.
    2. Không chứa bất kỳ giá trị NaN / missing nào.
    3. Tất cả các quốc gia đều phải đủ các năm (2010-2024).
    """
    if not panel_file_path.exists():
        print(f"❌ [ERROR] Không tìm thấy file Panel Data: {panel_file_path}")
        return False

    df_panel = pd.read_csv(panel_file_path)

    # 1. Kiểm tra bảng rỗng
    if df_panel.empty:
        print(f"❌ [ERROR] Bảng Panel Data bị rỗng (0 quốc gia, 0 dòng)!")
        return False

    # 2. Kiểm tra ô Missing Values (NaN)
    total_nans = df_panel.isna().sum().sum()
    if total_nans > 0:
        print(f"❌ [ERROR] Phát hiện {total_nans} ô bị missing (NaN) trong Panel Data!")
        nan_cols = df_panel.isna().sum()
        print("Chi tiết các cột bị missing:\n", nan_cols[nan_cols > 0])
        return False

    # 3. Kiểm tra Đủ Năm cho từng Quốc gia
    expected_years = set(paths.YEARS)
    expected_year_count = len(expected_years)

    years_per_country = df_panel.groupby("country_code_3")["year"].apply(set)
    incomplete_countries = {
        code: expected_years - yrs
        for code, yrs in years_per_country.items()
        if yrs != expected_years
    }

    if incomplete_countries:
        print(f"❌ [ERROR] Phát hiện {len(incomplete_countries)} quốc gia bị thiếu năm!")
        for code, missing_yrs in list(incomplete_countries.items())[:5]:
            print(f"   - Nước {code}: Thiếu các năm {sorted(list(missing_yrs))}")
        return False

    num_countries = df_panel["country_code_3"].nunique()
    print(f"✅ KIỂM TRA DỮ LIỆU THÀNH CÔNG: {num_countries} quốc gia sạch 100% dữ liệu, đủ {expected_year_count} năm (2010-2024), 0 ô NaN.")
    return True

def step_1_data_processing(
        use_imputed: bool = True,
        drop_countries_with_missing: bool = True
):
    print("\n=======================================================")
    print("📍 GIAI ĐOẠN 1: NẠP VÀ LÀM SẠCH DỮ LIỆU PANEL DATA")
    print("=======================================================")

    # 1. Tiến hành Impute dữ liệu thô nếu cờ use_imputed = True
    if use_imputed:
        print("🔄 Đang thực hiện nội suy (Impute) cho các file dữ liệu thô (World Bank & OWID)...")
        impute_all_worldbank_raw_files()
        impute_all_ourworldindata_raw_files()

    # 2. Nạp và ghép 20 chỉ số vào draft
    build_and_populate_all_draft_data(use_imputed=use_imputed, clear_existing=True)

    # 3. Gom thành bảng Panel Data (drop_countries=False, drop_countries_with_missing=True)
    combine_all_drafts_to_panel_data(
        drop_countries=False,
        drop_countries_with_missing=drop_countries_with_missing
    )

    # 🌟 BƯỚC KIỂM TRA ĐỘ TOÀN VẸN (VALIDATION CHECK)
    panel_csv_path = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"
    print("\n🔍 Đang kiểm tra độ toàn vẹn của draft_panel_2010_2024.csv...")
    
    if not validate_panel_data_integrity(panel_csv_path):
        print("\n⛔ [STOP PIPELINE] Dữ liệu Panel Data không đạt yêu cầu!")
        print(f"👉 Đã lưu Panel Data tại: {panel_csv_path}")
        sys.exit(1)  # Dừng chương trình ngay lập tức

    # 4. Tiền xử lý dữ liệu xuất ra cho 3 Experiments (Chỉ chạy khi dữ liệu đã đạt chuẩn 100%)
    print("\n🔄 Tiến hành tiền xử lý và tách dữ liệu cho 3 Experiments...")
    preprocess_and_export_all_experiment_1_years()
    preprocess_and_split_experiment_2()
    preprocess_and_split_experiment_3()

def step_2_experiment_1():
    print("\n=======================================================")
    print("📍 GIAI ĐOẠN 2: EXPERIMENT 1 - DISTANCE MATRICES & HEB PLOTS")
    print("=======================================================")

    # 1. Tính và xuất các Ma trận Khoảng cách
    export_experiment_1_distance_matrices()
    export_experiment_1_cosine_matrices()

    # 2. Xuất file tổng hợp Hàng xóm (Wide format)
    generate_euclidean_neighbors_summary()
    generate_cosine_neighbors_summary()

    # 3. Xuất tất cả biểu đồ HEB HTML (Euclidean & Cosine)
    export_all_years_distance_heb_plots(continent_map=continent_map)
    export_all_years_cosine_heb_plots(continent_map=continent_map)

def step_3_experiment_2():
    print("\n=======================================================")
    print("📍 GIAI ĐOẠN 3: EXPERIMENT 2 - COUNTRY CLASSIFICATION")
    print("=======================================================")

    # 1. Load dữ liệu Exp 2
    (
        X_train_exp2, y_train_exp2,
        X_val_exp2, y_val_exp2,
        X_test_exp2, y_test_exp2,
        le
    ) = load_experiment_2_country_classification()

    # 2. Auto-Benchmark so sánh các mô hình trên tập Validation
    run_auto_benchmark_experiment_2(
        X_train=X_train_exp2,
        y_train=y_train_exp2,
        X_val=X_val_exp2,
        y_val=y_val_exp2,
        plot_results=True
    )

    # 3. Đánh giá mô hình Quán quân Extra Trees trên tập TEST
    evaluate_extra_trees_on_test(
        X_train=X_train_exp2,
        y_train=y_train_exp2,
        X_test=X_test_exp2,
        y_test=y_test_exp2,
        label_encoder=le
    )


def step_4_experiment_3():
    print("\n=======================================================")
    print("📍 GIAI ĐOẠN 4: EXPERIMENT 3 - MULTI-OUTPUT FORECASTING")
    print("=======================================================")

    # 1. Load dữ liệu Exp 3
    (
        X_train_exp3, y_train_exp3,
        X_val_exp3, y_val_exp3,
        X_test_exp3, y_test_exp3
    ) = load_experiment_3()

    # 2. Auto-Benchmark so sánh các mô hình Hồi quy trên tập Validation
    run_auto_benchmark_experiment_3(
        X_train=X_train_exp3,
        Y_train=y_train_exp3,
        X_val=X_val_exp3,
        Y_val=y_val_exp3,
        plot_results=True
    )

    # 3. Đánh giá mô hình Quán quân Linear Regression trên tập TEST
    (
        test_results_df, 
        test_feature_details_df, 
        country_year_df
    ) = evaluate_linear_regression_on_test(
        X_train=X_train_exp3,
        Y_train=y_train_exp3,
        X_test=X_test_exp3,
        Y_test=y_test_exp3,
        plot_results=True
    )

    # 4. Vẽ biểu đồ phân tích độ ổn định dự báo toàn cầu theo năm
    plot_global_yearly_boxplot(country_year_df, metric="Cosine_similarity")
    plot_global_yearly_boxplot(country_year_df, metric="RMSE")

    # 5. Vẽ biểu đồ Scatter Plot 4 góc phần tư về độ ổn định các quốc gia
    plot_country_stability_scatter_chart(country_year_df)


def step_5_closed_loop():
    print("\n=======================================================")
    print("📍 GIAI ĐOẠN 5: CLOSED-LOOP PIPELINE (EXP 3 FORECAST -> EXP 2 CLASSIFY)")
    print("=======================================================")

    # 1. Nạp dữ liệu của cả Exp 2 và Exp 3
    (
        X_train_exp2, y_train_exp2,
        X_val_exp2, y_val_exp2,
        X_test_exp2, y_test_exp2,
        le
    ) = load_experiment_2_country_classification()

    (
        X_train_exp3, y_train_exp3,
        X_val_exp3, y_val_exp3,
        X_test_exp3, y_test_exp3
    ) = load_experiment_3()

    # 2. Chạy Pipeline ghép nối đóng vòng
    run_forecast_then_classify_pipeline(
        X_train_exp3=X_train_exp3,
        Y_train_exp3=y_train_exp3,
        X_test_exp3=X_test_exp3,
        X_train_exp2=X_train_exp2,
        y_train_exp2=y_train_exp2,
        y_test_exp2=y_test_exp2,
        label_encoder=le
    )

    plot_misclassification_comparison(show_fig=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Kỳ vọng giá trị Boolean (True/False hoặc 1/0).')

def main():
    parser = argparse.ArgumentParser(
        description="Master Execution Pipeline cho Dự án Country Data Fingerprint",
        epilog=PIPELINE_HELPER_TEXT, # 👈 Hiển thị hướng dẫn khi gõ --help
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "data", "exp1", "exp2", "exp3", "closed_loop"],
        help="Chọn giai đoạn cần chạy: 'all', 'data', 'exp1', 'exp2', 'exp3', 'closed_loop' (Mặc định: 'all')"
    )
    
    parser.add_argument(
        "--use_imputed",
        type=str2bool,
        default=True,
        help="Sử dụng dữ liệu đã Impute (True) hay chưa Impute (False). Mặc định: True"
    )

    parser.add_argument(
        "--drop_missing",
        type=str2bool,
        default=True,
        help="Lọc bỏ hoàn toàn các quốc gia có chứa dữ liệu missing (True/False. Mặc định: True)"
    )

    args = parser.parse_args()

    print("=======================================================")
    print("🌟 COUNTRY DATA FINGERPRINT - MASTER PIPELINE 🌟")
    print("=======================================================")

    if args.step in ["all", "data"]:
        step_1_data_processing(
            use_imputed=args.use_imputed,
            drop_countries_with_missing=args.drop_missing
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
    print("🎉 TOÀN BỘ PIPELINE ĐÃ HOÀN THÀNH THÀNH CÔNG RỰC RỠ!")
    print("=======================================================")


if __name__ == "__main__":
    main()
    '''
# 1. Chạy toàn bộ Pipeline từ A đến Z
python -m src.script.run_pipeline --step all --use_imputed True
python -m src.script.run_pipeline --step all --use_imputed False

# 2. Chạy riêng từng giai đoạn:
python -m src.script.run_pipeline --step data         # Chỉ xử lý dữ liệu
python -m src.script.run_pipeline --step exp1         # Chỉ chạy Exp 1 (Distance & HEB)
python -m src.script.run_pipeline --step exp2         # Chỉ chạy Exp 2 (Classification)
python -m src.script.run_pipeline --step exp3         # Chỉ chạy Exp 3 (Forecasting)
python -m src.script.run_pipeline --step closed_loop  # Chỉ chạy Closed-Loop (Forecast -> Classify)
    '''