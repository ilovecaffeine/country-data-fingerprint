# src/data/check_correlation.py
# cd "C:\Users\admin\Documents\Code_for_fun\country-data-fingerprint"
# python -m src.data.check_correlation

import pandas as pd
from config import paths


def check_redundant_columns(threshold: float = 0.90) -> None:
    """Đọc từng file draft_{year}.csv, tính ma trận tương quan giữa các cột số

    và in ra các cặp cột bị tương quan cao hơn ngưỡng threshold (mặc định >
    0.90).
    """
    print(f"\n=======================================================")
    print(f" KIỂM TRA CÁC CẶP CỘT BỊ DƯ THỪA (HỆ SỐ TƯƠNG QUAN |r| > {threshold})")
    print(f"=======================================================\n")

    # Lưu lại tổng hợp các cặp bị lặp nhiều nhất qua các năm
    overall_summary = {}

    for year in paths.YEARS:
        draft_file = paths.get_draft_csv_path(year)

        if not draft_file.exists():
            print(f"[SKIP] File không tồn tại: {draft_file.name}")
            continue

        df = pd.read_csv(draft_file)

        # Chỉ lấy các cột dữ liệu kiểu số (bỏ các cột mã nước, tên nước...)
        numeric_df = df.select_dtypes(include=["float64", "int64"])

        # Tính ma trận tương quan Pearson (tự động bỏ qua NaNs theo từng cặp)
        corr_matrix = numeric_df.corr().abs()

        # Tìm các cặp cột có |r| > threshold (chỉ xét tam giác trên của ma trận để tránh lặp)
        redundant_pairs = []
        cols = corr_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r_val = corr_matrix.iloc[i, j]
                if r_val > threshold:
                    col1, col2 = cols[i], cols[j]
                    redundant_pairs.append((col1, col2, r_val))

                    # Đếm tần suất xuất hiện qua các năm
                    pair_key = tuple(sorted([col1, col2]))
                    overall_summary[pair_key] = (
                        overall_summary.get(pair_key, 0) + 1
                    )

        # In kết quả cho năm hiện tại
        print(f"📅 NĂM {year} (File: {draft_file.name}):")
        if not redundant_pairs:
            print("   -> Không có cặp cột nào có tương quan >", threshold)
        else:
            # Sắp xếp theo hệ số tương quan giảm dần
            redundant_pairs.sort(key=lambda x: x[2], reverse=True)
            for col1, col2, r_val in redundant_pairs:
                print(f"   ⚠️  [{col1}] <---> [{col2}]: r = {r_val:.4f}")
        print("-" * 65)

    # In bảng tổng hợp các cặp biến bị lặp xuyên suốt nhiều năm nhất
    print("\n=======================================================")
    print(" TỔNG HỢP CÁC CẶP BIẾN BỊ DƯ THỪA XUYÊN SUỐT NHIỀU NĂM NHẤT")
    print("=======================================================")
    sorted_summary = sorted(
        overall_summary.items(), key=lambda x: x[1], reverse=True
    )
    for (col1, col2), count in sorted_summary:
        print(f" 🔥 Cặp [{col1}] & [{col2}]: Xuất hiện {count}/{len(paths.YEARS)} năm")


if __name__ == "__main__":
    check_redundant_columns(threshold=0.90)