# src/distance_matrices/distance_matrices.py
# cd "C:\Users\admin\Documents\Code_for_fun\country-data-fingerprint"
# python -m src.distance_matrices.distance_matrices
from config import paths
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform


def compute_euclidean_matrix_scipy(
    df: pd.DataFrame, id_cols: list[str] = ["country_code_3"]
) -> pd.DataFrame:
    """Tính Ma trận khoảng cách Euclidean giữa tất cả các hàng trong DataFrame."""
    labels = df[id_cols[0]].values
    feature_cols = [
        c
        for c in df.columns
        if c not in id_cols + ["country_name", "year"]
    ]
    features = df[feature_cols].values

    # Tính ma trận khoảng cách đối xứng (NxN)
    dist_matrix = squareform(pdist(features, metric="euclidean"))

    # Trả về ma trận dạng DataFrame với nhãn dòng/cột là mã quốc gia
    return pd.DataFrame(dist_matrix, index=labels, columns=labels)


def find_closest_countries(
    dist_df: pd.DataFrame, country_code: str = "VNM", top_k: int = 5
) -> pd.Series:
    """Tìm top_k quốc gia có khoảng cách Euclidean gần nhất với một quốc gia cho trước."""
    if country_code not in dist_df.index:
        raise ValueError(
            f"Không tìm thấy quốc gia {country_code} trong ma trận!"
        )

    # Lấy hàng tương ứng, sắp xếp tăng dần, bỏ qua chính nó (vị trí 0)
    return dist_df.loc[country_code].sort_values().iloc[1 : top_k + 1]


def export_experiment_1_distance_matrices(
    exp1_input_dir: Path | None = None,
    output_dir: Path | None = None,
    target_year: int | None = None,
) -> None:
    """Đọc dữ liệu từ PROCESSED_DATA_EXPERIMENT1_DIR, tính ma trận khoảng cách

    Euclidean và xuất ra file CSV tương ứng.

    Parameters
    ----------
    exp1_input_dir : Path, optional
        Thư mục chứa các file CSV của Experiment 1. Mặc định dùng config paths.
    output_dir : Path, optional
        Thư mục lưu ma trận khoảng cách xuất ra.
    target_year : int, optional
        Nếu truyền cụ thể (ví dụ: 2024), chỉ xuất ma trận năm đó.
        Nếu None, sẽ tự động quét và xuất cho TẤT CẢ các năm.
    """
    if exp1_input_dir is None:
        exp1_input_dir = paths.PROCESSED_DATA_EXPERIMENT1_DIR

    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT1_DIR / "distance_matrices"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lấy danh sách các file CSV cần xử lý
    if target_year is not None:
        csv_files = [exp1_input_dir / f"panel_exp1_{target_year}.csv"]
    else:
        csv_files = sorted(exp1_input_dir.glob("panel_exp1_*.csv"))

    if not csv_files:
        print(
            f"[ERROR] Không tìm thấy file dữ liệu Experiment 1 trong: {exp1_input_dir}"
        )
        return

    print("=======================================================")
    print("BẮT ĐẦU TÍNH VÀ XUẤT MA TRẬN KHOẢNG CÁCH EUCLIDEAN")
    print("=======================================================\n")

    for file_path in csv_files:
        if not file_path.exists():
            print(f"[SKIP] File không tồn tại: {file_path.name}")
            continue

        # Trích xuất năm từ tên file (panel_exp1_2024.csv -> 2024)
        year_str = file_path.stem.split("_")[-1]

        # 2. Đọc dữ liệu
        df_year = pd.read_csv(file_path)

        # 3. Tính ma trận khoảng cách
        dist_matrix_df = compute_euclidean_matrix_scipy(
            df_year, id_cols=["country_code_3"]
        )

        # 4. Lưu ma trận ra CSV
        output_matrix_path = output_dir / f"euclidean_matrix_{year_str}.csv"
        dist_matrix_df.to_csv(output_matrix_path, encoding="utf-8")

        print(f"✅ Đã xuất ma trận năm {year_str} -> {output_matrix_path.name}")

        # 5. In thử nghiệm Top 5 gần Việt Nam nhất (nếu có VNM trong bảng)
        if "VNM" in dist_matrix_df.index:
            top_vnm = find_closest_countries(
                dist_matrix_df, country_code="VNM", top_k=5
            )
            print(f"   📍 Top 5 quốc gia gần Vietnam (VNM) nhất năm {year_str}:")
            for code, dist in top_vnm.items():
                print(f"      - {code}: {dist:.4f}")
            print("-" * 55)

    print("\n=======================================================")
    print(f"📁 TẤT CẢ MA TRẬN ĐÃ ĐƯỢC LƯU TẠI: {output_dir}")
    print("=======================================================")


def compute_cosine_similarity_matrix_scipy(
    df: pd.DataFrame, id_cols: list[str] = ["country_code_3"]
) -> pd.DataFrame:
    """Tính Ma trận Cosine Similarity giữa tất cả các hàng trong DataFrame.

    Returns
    -------
    pd.DataFrame
        Ma trận tương đồng có giá trị trong khoảng [-1, 1].
        Giá trị càng gần 1 biểu thị hai quốc gia càng có cấu trúc Fingerprint tương đồng.
    """
    labels = df[id_cols[0]].values
    feature_cols = [
        c
        for c in df.columns
        if c not in id_cols + ["country_name", "year"]
    ]
    features = df[feature_cols].values

    # pdist với metric='cosine' tính Cosine Distance = 1 - Cosine Similarity
    cosine_dist_matrix = squareform(pdist(features, metric="cosine"))

    # Chuyển đổi sang Cosine Similarity
    cosine_sim_matrix = 1.0 - cosine_dist_matrix

    return pd.DataFrame(cosine_sim_matrix, index=labels, columns=labels)


def find_most_similar_countries(
    sim_df: pd.DataFrame, country_code: str = "VNM", top_k: int = 5
) -> pd.Series:
    """Tìm top_k quốc gia có Cosine Similarity cao nhất với một quốc gia cho trước."""
    if country_code not in sim_df.index:
        raise ValueError(
            f"Không tìm thấy quốc gia {country_code} trong ma trận!"
        )

    # Lấy hàng tương ứng, sắp xếp giảm dần, bỏ qua chính nó (vị trí đầu tiên = 1.0)
    return sim_df.loc[country_code].sort_values(ascending=False).iloc[1 : top_k + 1]


def export_experiment_1_cosine_matrices(
    exp1_input_dir: Path | None = None,
    output_dir: Path | None = None,
    target_year: int | None = None,
) -> None:
    """Đọc dữ liệu từ PROCESSED_DATA_EXPERIMENT1_DIR, tính ma trận Cosine Similarity

    và xuất ra file CSV tương ứng (`cosine_similarity_matrix_{year}.csv`).

    Parameters
    ----------
    exp1_input_dir : Path, optional
        Thư mục chứa các file CSV của Experiment 1. Mặc định dùng config paths.
    output_dir : Path, optional
        Thư mục lưu ma trận Cosine Similarity xuất ra.
    target_year : int, optional
        Nếu truyền cụ thể (ví dụ: 2024), chỉ xuất ma trận năm đó.
        Nếu None, sẽ tự động quét và xuất cho TẤT CẢ các năm.
    """
    if exp1_input_dir is None:
        exp1_input_dir = paths.PROCESSED_DATA_EXPERIMENT1_DIR

    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT1_DIR / "cosine_matrices"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Lấy danh sách các file CSV cần xử lý
    if target_year is not None:
        csv_files = [exp1_input_dir / f"panel_exp1_{target_year}.csv"]
    else:
        csv_files = sorted(exp1_input_dir.glob("panel_exp1_*.csv"))

    if not csv_files:
        print(
            f"[ERROR] Không tìm thấy file dữ liệu Experiment 1 trong: {exp1_input_dir}"
        )
        return

    print("=======================================================")
    print("BẮT ĐẦU TÍNH VÀ XUẤT MA TRẬN COSINE SIMILARITY")
    print("=======================================================\n")

    for file_path in csv_files:
        if not file_path.exists():
            print(f"[SKIP] File không tồn tại: {file_path.name}")
            continue

        # Trích xuất năm từ tên file (panel_exp1_2024.csv -> 2024)
        year_str = file_path.stem.split("_")[-1]

        # 2. Đọc dữ liệu
        df_year = pd.read_csv(file_path)

        # 3. Tính ma trận Cosine Similarity
        cosine_sim_df = compute_cosine_similarity_matrix_scipy(
            df_year, id_cols=["country_code_3"]
        )

        # 4. Lưu ma trận ra CSV
        output_matrix_path = output_dir / f"cosine_similarity_matrix_{year_str}.csv"
        cosine_sim_df.to_csv(output_matrix_path, encoding="utf-8")

        print(
            f"✅ Đã xuất ma trận Cosine Similarity năm {year_str} -> {output_matrix_path.name}"
        )

        # 5. In thử nghiệm Top 5 gần Việt Nam nhất (nếu có VNM trong bảng)
        if "VNM" in cosine_sim_df.index:
            top_vnm = find_most_similar_countries(
                cosine_sim_df, country_code="VNM", top_k=5
            )
            print(
                f"   📍 Top 5 quốc gia tương đồng với Vietnam (VNM) nhất năm {year_str}:"
            )
            for code, sim in top_vnm.items():
                print(f"      - {code}: {sim:.4f}")
            print("-" * 55)

    print("\n=======================================================")
    print(f"📁 TẤT CẢ MA TRẬN TƯƠNG ĐỒNG ĐÃ ĐƯỢC LƯU TẠI: {output_dir}")
    print("=======================================================")


def generate_euclidean_neighbors_summary(
    matrices_dir: Path | None = None,
    output_file: Path | None = None,
    raw_panel_file: Path | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Tổng hợp top K hàng xóm gần nhất theo Euclidean Distance từ tất cả các năm (2010-2024)

    thành file duy nhất dạng Wide Format:
      - Cột: country_code_3 | country_name | rank | 2010 | 2011 | ... | 2024
      - Ô: "khoảng_cách (mã_nước_B)" (ví dụ: "2.8729 (KHM)")
    """
    if matrices_dir is None:
        matrices_dir = paths.RESULTS_EXPERIMENT1_DIR / "distance_matrices"

    if output_file is None:
        output_file = paths.RESULTS_EXPERIMENT1_DIR / "euclidean_neighbors.csv"

    if raw_panel_file is None:
        raw_panel_file = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"

    # 1. Đọc ánh xạ mã quốc gia -> tên quốc gia (country_code_3 -> country_name)
    country_name_map = {}
    if raw_panel_file.exists():
        df_raw = pd.read_csv(raw_panel_file)
        country_name_map = (
            df_raw.drop_duplicates(subset=["country_code_3"])
            .set_index("country_code_3")["country_name"]
            .to_dict()
        )

    # 2. Lấy danh sách các file ma trận Euclidean đã xuất
    matrix_files = sorted(matrices_dir.glob("euclidean_matrix_*.csv"))

    if not matrix_files:
        print(f"[ERROR] Không tìm thấy file ma trận nào trong: {matrices_dir}")
        return pd.DataFrame()

    print("=======================================================")
    print("BẮT ĐẦU TẠO FILE TỔNG HỢP EUCLIDEAN NEIGHBORS (WIDE FORMAT)")
    print("=======================================================\n")

    # Dictionary lưu thông tin theo cấu trúc: dict[country][rank][year] = "dist (code)"
    records_dict = {}
    all_years = []

    for file_path in matrix_files:
        year_str = file_path.stem.split("_")[-1]
        all_years.append(year_str)

        # Đọc ma trận khoảng cách
        df_matrix = pd.read_csv(file_path, index_col=0)

        for country_a in df_matrix.index:
            if country_a not in records_dict:
                records_dict[country_a] = {
                    r: {} for r in range(1, top_k + 1)
                }

            # Tìm top K nước gần nhất (loại bỏ chính country_a)
            row = df_matrix.loc[country_a].sort_values(ascending=True)
            top_neighbors = row.iloc[1 : top_k + 1]

            for rank_idx, (country_b, dist_val) in enumerate(
                top_neighbors.items(), start=1
            ):
                formatted_value = f"{dist_val:.4f} ({country_b})"
                records_dict[country_a][rank_idx][year_str] = formatted_value

    all_years = sorted(list(set(all_years)), key=lambda x: int(x))

    # 3. Chuyển đổi dữ liệu Dictionary sang DataFrame dạng Wide
    rows = []
    for country_code in sorted(records_dict.keys()):
        c_name = country_name_map.get(country_code, country_code)

        for r in range(1, top_k + 1):
            row_dict = {
                "country_code_3": country_code,
                "country_name": c_name,
                "rank": r,
            }
            # Điền giá trị cho từng năm
            for y in all_years:
                row_dict[y] = records_dict[country_code][r].get(y, None)

            rows.append(row_dict)

    summary_df = pd.DataFrame(rows)

    # 4. Xuất ra file CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"✅ ĐÃ XUẤT THÀNH CÔNG: {output_file}")
    print(f"📊 Tổng số dòng: {len(summary_df)} ({len(records_dict)} quốc gia x 3 ranks)")
    print("=======================================================\n")


    return summary_df

def generate_cosine_neighbors_summary(
    matrices_dir: Path | None = None,
    output_file: Path | None = None,
    raw_panel_file: Path | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Tổng hợp top K hàng xóm tương đồng nhất theo Cosine Similarity từ tất cả các năm (2010-2024)

    thành file duy nhất dạng Wide Format:
      - Cột: country_code_3 | country_name | rank | 2010 | 2011 | ... | 2024
      - Ô: "độ_tương_đồng (mã_nước_B)" (ví dụ: "0.8072 (KHM)")
    """
    if matrices_dir is None:
        matrices_dir = paths.RESULTS_EXPERIMENT1_DIR / "cosine_matrices"

    if output_file is None:
        output_file = paths.RESULTS_EXPERIMENT1_DIR / "cosine_neighbors.csv"

    if raw_panel_file is None:
        raw_panel_file = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"

    # 1. Đọc ánh xạ mã quốc gia -> tên quốc gia (country_code_3 -> country_name)
    country_name_map = {}
    if raw_panel_file.exists():
        df_raw = pd.read_csv(raw_panel_file)
        country_name_map = (
            df_raw.drop_duplicates(subset=["country_code_3"])
            .set_index("country_code_3")["country_name"]
            .to_dict()
        )

    # 2. Lấy danh sách các file ma trận Cosine Similarity đã xuất
    # Tìm cả các file có tiền tố cosine_similarity_matrix_*.csv hoặc cosine_matrix_*.csv
    matrix_files = sorted(matrices_dir.glob("*cosine*matrix_*.csv"))

    if not matrix_files:
        # Fallback thử tìm tất cả file csv trong thư mục nếu quy cách đặt tên khác
        matrix_files = sorted(matrices_dir.glob("*.csv"))

    if not matrix_files:
        print(f"[ERROR] Không tìm thấy file ma trận nào trong: {matrices_dir}")
        return pd.DataFrame()

    print("=======================================================")
    print("BẮT ĐẦU TẠO FILE TỔNG HỢP COSINE NEIGHBORS (WIDE FORMAT)")
    print("=======================================================\n")

    # Dictionary lưu thông tin theo cấu trúc: dict[country][rank][year] = "sim (code)"
    records_dict = {}
    all_years = []

    for file_path in matrix_files:
        # Trích xuất năm từ tên file (ví dụ: cosine_similarity_matrix_2024.csv -> 2024)
        year_str = file_path.stem.split("_")[-1]
        all_years.append(year_str)

        # Đọc ma trận độ tương đồng Cosine
        df_matrix = pd.read_csv(file_path, index_col=0)

        for country_a in df_matrix.index:
            if country_a not in records_dict:
                records_dict[country_a] = {
                    r: {} for r in range(1, top_k + 1)
                }

            # Tìm top K nước có Cosine Similarity cao nhất
            # Sắp xếp GIẢM DẦN (ascending=False), loại bỏ chính country_a ở vị trí đầu (1.0)
            row = df_matrix.loc[country_a].sort_values(ascending=False)
            top_neighbors = row.iloc[1 : top_k + 1]

            for rank_idx, (country_b, sim_val) in enumerate(
                top_neighbors.items(), start=1
            ):
                formatted_value = f"{sim_val:.4f} ({country_b})"
                records_dict[country_a][rank_idx][year_str] = formatted_value

    all_years = sorted(list(set(all_years)), key=lambda x: int(x))

    # 3. Chuyển đổi dữ liệu Dictionary sang DataFrame dạng Wide
    rows = []
    for country_code in sorted(records_dict.keys()):
        c_name = country_name_map.get(country_code, country_code)

        for r in range(1, top_k + 1):
            row_dict = {
                "country_code_3": country_code,
                "country_name": c_name,
                "rank": r,
            }
            # Điền giá trị cho từng năm
            for y in all_years:
                row_dict[y] = records_dict[country_code][r].get(y, None)

            rows.append(row_dict)

    summary_df = pd.DataFrame(rows)

    # 4. Xuất ra file CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"✅ ĐÃ XUẤT THÀNH CÔNG: {output_file}")
    print(f"📊 Tổng số dòng: {len(summary_df)} ({len(records_dict)} quốc gia x {top_k} ranks)")
    print("=======================================================\n")


    return summary_df


if __name__ == "__main__":
	# 1. Đọc dữ liệu Experiment 1 của năm 2024
	#df_2024 = pd.read_csv(paths.PROCESSED_DATA_EXPERIMENT1_DIR / "panel_exp1_2024.csv")

	# 2. Tính ma trận khoảng cách
	#dist_matrix_df = compute_euclidean_matrix_scipy(df_2024, id_cols=["country_code_3"])

	# 3. Tìm 5 quốc gia có Fingerprint gần Việt Nam (VNM) nhất trong năm 2024
	#top_vnm = find_closest_countries(dist_matrix_df, country_code="VNM", top_k=5)

	#print("Top 5 quốc gia gần Việt Nam nhất về khoảng cách Euclidean năm 2024:")
	#print(top_vnm)

    # Xuất tất cả các năm (2010 -> 2024)
	export_experiment_1_distance_matrices()

    # Hoặc nếu chỉ muốn xuất riêng năm 2024:
    # export_experiment_1_distance_matrices(target_year=2024)

    # Xuất Cosine Similarity Matrix cho tất cả các năm
	export_experiment_1_cosine_matrices()

	generate_euclidean_neighbors_summary()
	generate_cosine_neighbors_summary()