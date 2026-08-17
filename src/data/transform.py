# src/data/transform.py
#cd "C:\Users\admin\Documents\Code_for_fun\country-data-fingerprint"
#python -m src.data.transform
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
from config import paths
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler


# Danh sách mặc định các quốc gia/lãnh thổ bị khuyết dữ liệu quá nhiều
DEFAULT_COUNTRIES_TO_DROP = [
    # 10 quốc gia bị khuyết dữ liệu nặng / vi quốc gia ban đầu
    "PRK",  # North Korea
    "ERI",  # Eritrea
    "SSD",  # South Sudan
    "VEN",  # Venezuela
    "LIE",  # Liechtenstein
    "MCO",  # Monaco
    "TUV",  # Tuvalu
    "NRU",  # Nauru
    "AND",  # Andorra
    "SOM",  # Somalia

    # Các quốc gia bị khuyết 100% chỉ số Govt Expend & Trade (hoặc Unemployment) liên tục 15 năm
    "KNA",  # Saint Kitts and Nevis
    "GRD",  # Grenada
    "DMA",  # Dominica
    "FSM",  # Micronesia
    "GUY",  # Guyana
    "JOR",  # Jordan
    "JAM",  # Jamaica
    "MMR",  # Myanmar
    "SMR",  # San Marino
    "NGA",  # Nigeria
    "VCT",  # Saint Vincent and the Grenadines
    "TTO",  # Trinidad and Tobago
    "LBR",  # Liberia
    "ATG",  # Antigua and Barbuda
    "PNG",  # Papua New Guinea
    "LCA",  # Saint Lucia
    "BRB",  # Barbados
    "SUR",  # Suriname
    "PLW",  # Palau
    "AFG",  # Afghanistan (thiếu 10/15 năm liên tiếp từ 2010-2019)

    # 3. Nhóm 11 quốc gia mới bổ sung (thiếu 100% 1 chỉ số hoặc bị khuyết đứt gãy 2017-2024)
    "KIR",  # Kiribati (thiếu 100% unemployment)
    "SYC",  # Seychelles (thiếu 100% unemployment)
    "MHL",  # Marshall Islands (thiếu 100% unemployment)
    "YEM",  # Yemen (thiếu 100% GDP PPP)
    "CUB",  # Cuba (thiếu 100% GDP PPP)
    "BDI",  # Burundi (thiếu 100% Trade)
    "LAO",  # Laos (thiếu Govt Expend & Trade từ 2017-2024)
    "SDN",  # Sudan (thiếu Internet users)
    "WSM",  # Samoa (thiếu Internet users từ 2015-2024)
    "VUT",  # Vanuatu (thiếu Internet users từ 2016-2024)
    "TKM",  # Turkmenistan (thiếu Internet users từ 2017-2024)
]


def combine_all_drafts_to_panel_data(
    drop_countries: bool = True,
    countries_to_drop: Optional[list[str]] = None,
    drop_countries_with_missing: bool = False,
    ) -> pd.DataFrame:
    """Đọc các file draft_{year}.csv, tự động chèn cột 'year' vào vị trí thứ 3,

    tùy chọn lọc bỏ các quốc gia trong danh sách, tùy chọn lọc bỏ tất cả quốc gia bị missing data,
    và gộp lại thành 1 file Panel Data duy nhất lưu tại data/raw/draft_panel_2010_2024.csv.

    Parameters
    ----------
    drop_countries : bool, optional
        Nếu True, sẽ thực hiện lọc bỏ danh sách quốc gia chỉ định. Mặc định là True.
    countries_to_drop : list[str], optional
        Danh sách các mã quốc gia (ISO3) cần bỏ. Nếu None, sẽ dùng DEFAULT_COUNTRIES_TO_DROP.
    drop_countries_with_missing : bool, optional
        Nếu True, sẽ lọc bỏ HOÀN TOÀN tất cả các quốc gia có ít nhất 1 ô dữ liệu bị missing/NaN
        trong cả giai đoạn 2010-2024 (chỉ giữ lại các nước sạch dữ liệu 100%). Mặc định là False.
    """
    # Nếu không truyền danh sách riêng thì dùng danh sách mặc định
    if countries_to_drop is None:
        countries_to_drop = DEFAULT_COUNTRIES_TO_DROP

    all_dfs = []

    for year in paths.YEARS:
        draft_file = paths.get_draft_csv_path(year)

        if not draft_file.exists():
            print(f"[SKIP] File không tồn tại: {draft_file.name}")
            continue

        df = pd.read_csv(draft_file)

        # Tránh lỗi nếu file đã có sẵn cột 'year'
        if "year" in df.columns:
            df = df.drop(columns=["year"])

        # 1. Chèn cột 'year' vào vị trí thứ 3 (index = 2)
        df.insert(loc=2, column="year", value=int(year))

        all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] Không tìm thấy file draft nào để tổng hợp!")
        return pd.DataFrame()

    # 2. Nối tất cả các bảng lại theo chiều dọc
    panel_df = pd.concat(all_dfs, ignore_index=True)

    # 3. Lọc bỏ quốc gia nếu tùy chọn drop_countries = True
    total_before = panel_df["country_code_3"].nunique()

    if drop_countries and countries_to_drop:
        panel_df = panel_df[~panel_df["country_code_3"].isin(countries_to_drop)].copy()
        total_after = panel_df["country_code_3"].nunique()
        print(
            f"🗑️ Đã lọc bỏ {total_before - total_after} quốc gia trong danh sách chỉ định."
        )

    # 4. LỌC BỎ TẤT CẢ QUỐC GIA BỊ MISSING DATA (NẾU BẬT CỜ drop_countries_with_missing)
    if drop_countries_with_missing:
        before_nan_drop = panel_df["country_code_3"].nunique()
        
        # Tìm danh sách mã quốc gia có ít nhất 1 ô bị missing (NaN)
        nan_countries = panel_df[panel_df.isna().any(axis=1)]["country_code_3"].unique()
        
        # Lọc giữ lại những quốc gia KHÔNG nằm trong danh sách nan_countries
        panel_df = panel_df[~panel_df["country_code_3"].isin(nan_countries)].copy()
        after_nan_drop = panel_df["country_code_3"].nunique()
        
        print(
            f"🧹 [CỜ CẢNH BÁO] Đã lọc bỏ hoàn toàn {before_nan_drop - after_nan_drop} quốc gia có chứa dữ liệu missing "
            f"(Chỉ giữ lại {after_nan_drop} quốc gia sạch dữ liệu 100%)."
        )

    total_final = panel_df["country_code_3"].nunique()

    # 5. Sắp xếp lại dữ liệu theo Quốc gia và Theo Năm liên tục
    panel_df = panel_df.sort_values(
        by=["country_code_3", "year"]
    ).reset_index(drop=True)

    # 6. Lưu thành file draft_panel_2010_2024.csv
    output_path = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(output_path, index=False, encoding="utf-8")

    print(
        f"✅ ĐÃ TẠO THÀNH CÔNG BẢNG PANEL DATA! ({len(panel_df)} dòng x {len(panel_df.columns)} cột | {total_final} quốc gia)"
    )
    print(f"📁 Lưu tại: {output_path}")

    return panel_df


def impute_worldbank_raw_file(
    input_file: Path,
    output_file: Path,
    eval_start: int = 2010,
    eval_end: int = 2024,
    impute_start: int = 1995,
    impute_end: int = 2024,
    max_missing_allowed: int = 7, 
) -> None:
    """Nội suy chuỗi thời gian cho file Raw World Bank (Wide Format).

    - Giữ nguyên 4 dòng metadata đầu tiên trong file xuất ra.
    - Kiểm tra dải [eval_start, eval_end]. Nếu số ô missing <= max_missing_allowed
    - Tiến hành nội suy tuyến tính + ffill + bfill trên dải [impute_start, impute_end].
    - Xuất ra file _imputed.csv cùng thư mục.
    """
    if not input_file.exists():
        print(f"[ERROR] File nguồn không tồn tại: {input_file}")
        return

    # 1. Đọc và lưu lại 4 dòng đầu (metadata) của file gốc
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        header_lines = [f.readline() for _ in range(4)]

    # 2. Đọc dữ liệu bảng World Bank (bỏ 4 dòng metadata)
    df_raw = pd.read_csv(input_file, skiprows=4)

    # 3. Xác định các danh sách cột năm có mặt trong file
    years_eval = [
        str(y) for y in range(eval_start, eval_end + 1) if str(y) in df_raw.columns
    ]
    years_impute = [
        str(y) for y in range(impute_start, impute_end + 1) if str(y) in df_raw.columns
    ]

    if not years_eval or not years_impute:
        print(f"[WARNING] Không tìm thấy đủ các cột năm trong file: {input_file.name}")
        return

    # 4. Đếm số lượng NaN trong khoảng 2010 - 2024 cho từng hàng (quốc gia)
    missing_count = df_raw[years_eval].isna().sum(axis=1)

    # 5. Lọc các hàng thỏa mãn điều kiện missing <= 7
    eligible_mask = missing_count <= max_missing_allowed

    # 6. Tiến hành Impute CHỈ cho các hàng thỏa mãn điều kiện, trên dải 1995 - 2024
    if eligible_mask.any():
        # Trích xuất khối dữ liệu cần impute
        block_to_impute = df_raw.loc[eligible_mask, years_impute].astype(float)

        # Nội suy tuyến tính theo chiều ngang (axis=1) + ffill + bfill
        block_imputed = (
            block_to_impute.interpolate(method="linear", axis=1)
            .ffill(axis=1)
            .bfill(axis=1)
        )

        # Gán lại dữ liệu đã impute vào dataframe gốc
        df_raw.loc[eligible_mask, years_impute] = block_imputed

    # 7. Lưu file _imputed.csv: Ghi 4 dòng header trước, sau đó ghi bảng dữ liệu (mode='a')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.writelines(header_lines)

    df_raw.to_csv(output_file, mode="a", index=False, encoding="utf-8")

    total_rows = len(df_raw)
    imputed_rows = eligible_mask.sum()
    skipped_rows = total_rows - imputed_rows

    print(f"[SUCCESS] Đã xử lý: {input_file.name}")
    print(f"          -> Impute thành công (1995-2024): {imputed_rows}/{total_rows} quốc gia")
    print(f"          -> Bỏ qua (missing > {max_missing_allowed}): {skipped_rows} quốc gia")
    print(f"          -> File xuất ra: {output_file.name}\n")


def impute_all_worldbank_raw_files() -> None:
    """Hàm chạy Impute cho TẤT CẢ các file Raw World Bank có trong config paths."""
    print("=======================================================")
    print("BẮT ĐẦU IMPUTE CÁC FILE RAW WORLD BANK (1995 - 2024)")
    print("=======================================================\n")

    # Danh sách các cặp file Raw gốc và file Imputed tương ứng
    wb_file_pairs = [
        (paths.RAW_GDP_CSV, paths.RAW_GDP_IMPUTED_CSV),
        (paths.RAW_GDP_GROWTH_CSV, paths.RAW_GDP_GROWTH_IMPUTED_CSV),
        (paths.RAW_GOVT_EXPENDITURE_CSV, paths.RAW_GOVT_EXPENDITURE_IMPUTED_CSV),
        (paths.RAW_INFLATION_CSV, paths.RAW_INFLATION_IMPUTED_CSV),
        (paths.RAW_UNEMPLOYMENT_CSV, paths.RAW_UNEMPLOYMENT_IMPUTED_CSV),
        (paths.RAW_POPULATION_TOTAL_CSV, paths.RAW_POPULATION_TOTAL_IMPUTED_CSV),
        (paths.RAW_POPULATION_GROWTH_CSV, paths.RAW_POPULATION_GROWTH_IMPUTED_CSV),
        (paths.RAW_URBAN_POPULATION_PCT_CSV, paths.RAW_URBAN_POPULATION_PCT_IMPUTED_CSV),
        (paths.RAW_URBAN_POPULATION_GROWTH_CSV, paths.RAW_URBAN_POPULATION_GROWTH_IMPUTED_CSV),
        (paths.RAW_LIFE_EXPECTANCY_CSV, paths.RAW_LIFE_EXPECTANCY_IMPUTED_CSV),
        (paths.RAW_FERTILITY_RATE_CSV, paths.RAW_FERTILITY_RATE_IMPUTED_CSV),
        (paths.RAW_UNDER_5_MORTALITY_CSV, paths.RAW_UNDER_5_MORTALITY_IMPUTED_CSV),
        (paths.RAW_ACCESS_TO_ELECTRICITY_CSV, paths.RAW_ACCESS_TO_ELECTRICITY_IMPUTED_CSV),
        (paths.RAW_ACCESS_TO_BASIC_DRINKING_WATER_CSV, paths.RAW_ACCESS_TO_BASIC_DRINKING_WATER_IMPUTED_CSV),
        (paths.RAW_SECONDARY_EDUCATION_CSV, paths.RAW_SECONDARY_EDUCATION_IMPUTED_CSV),
        (paths.RAW_INTERNET_USERS_CSV, paths.RAW_INTERNET_USERS_IMPUTED_CSV),
        (paths.RAW_TRADE_CSV, paths.RAW_TRADE_IMPUTED_CSV),
        (paths.RAW_AGRICULTURE_CSV, paths.RAW_AGRICULTURE_IMPUTED_CSV),
        (paths.RAW_INDUSTRY_CSV, paths.RAW_INDUSTRY_IMPUTED_CSV),
        (paths.RAW_SERVICES_CSV, paths.RAW_SERVICES_IMPUTED_CSV),
    ]

    for raw_path, imputed_path in wb_file_pairs:
        impute_worldbank_raw_file(input_file=raw_path, output_file=imputed_path)

    print("=> TẤT CẢ FILE RAW WORLD BANK ĐÃ ĐƯỢC IMPUTE XONG!")


def impute_ourworldindata_raw_file(
    input_file: Path,
    output_file: Path,
    eval_start: int = 2010,
    eval_end: int = 2024,
    impute_start: int = 1995,
    impute_end: int = 2024,
    max_missing_allowed: int = 7,
) -> None:
    """Nội suy chuỗi thời gian cho file Raw Our World in Data (Long Format).

    - Chuyển tạm sang Wide Format để đếm missing giai đoạn 2010-2024.
    - Nếu missing <= 7: Tiến hành nội suy tuyến tính + ffill + bfill (1995-2024).
    - Chuyển ngược về Long Format và xuất file _imputed.csv.
    """
    if not input_file.exists():
        print(f"[ERROR] File nguồn OWID không tồn tại: {input_file}")
        return

    # 1. Đọc file OWID gốc (dạng Long Format, đọc từ dòng 0)
    df_raw = pd.read_csv(input_file)

    # Identifiers: Entity và Code (nếu có)
    id_cols = [c for c in ["Entity", "Code"] if c in df_raw.columns]
    if not id_cols:
        print(f"[ERROR] Không tìm thấy cột Entity/Code trong: {input_file.name}")
        return

    # Xác định cột chứa giá trị chỉ số (cột còn lại ngoài Entity, Code, Year)
    val_cols = [c for c in df_raw.columns if c not in id_cols + ["Year"]]
    if not val_cols:
        print(
            f"[ERROR] Không tìm thấy cột giá trị chỉ số trong: {input_file.name}"
        )
        return

    val_col = val_cols[0]  # Cột chỉ số chính

    # 2. Pivot bảng từ Long Format sang Wide Format (Hàng: Entity/Code, Cột: Year)
    df_pivot = df_raw.pivot(index=id_cols, columns="Year", values=val_col)

    # Tạo đầy đủ các cột năm trong dải 1995 - 2024 (tránh trường hợp OWID bỏ sót năm)
    all_impute_years = list(range(impute_start, impute_end + 1))
    df_pivot = df_pivot.reindex(
        columns=sorted(set(df_pivot.columns).union(all_impute_years))
    )

    # 3. Lấy danh sách cột năm đánh giá và năm nội suy
    years_eval = [y for y in range(eval_start, eval_end + 1) if y in df_pivot.columns]
    years_impute = [
        y for y in range(impute_start, impute_end + 1) if y in df_pivot.columns
    ]

    # 4. Đếm số ô missing trong khoảng 2010 - 2024
    missing_count = df_pivot[years_eval].isna().sum(axis=1)

    # 5. Lọc các hàng thỏa mãn điều kiện missing <= 7
    eligible_mask = missing_count <= max_missing_allowed

    # 6. Tiến hành Impute cho khối thỏa mãn điều kiện trên dải 1995 - 2024
    if eligible_mask.any():
        block_to_impute = df_pivot.loc[eligible_mask, years_impute].astype(
            float
        )

        block_imputed = (
            block_to_impute.interpolate(method="linear", axis=1)
            .ffill(axis=1)
            .bfill(axis=1)
        )

        df_pivot.loc[eligible_mask, years_impute] = block_imputed

    # 7. Chuyển từ Wide Format trở lại Long Format chuẩn OWID
    df_imputed_long = df_pivot.reset_index().melt(
        id_vars=id_cols, var_name="Year", value_name=val_col
    )

    # Ép lại kiểu số nguyên cho Year và loại bỏ các dòng bị hoàn toàn NaN
    df_imputed_long["Year"] = df_imputed_long["Year"].astype(int)
    df_imputed_long = df_imputed_long.dropna(subset=[val_col]).reset_index(
        drop=True
    )

    # Giữ đúng thứ tự cột như file gốc
    orig_cols = [c for c in df_raw.columns if c in df_imputed_long.columns]
    df_imputed_long = df_imputed_long[orig_cols].sort_values(
        by=id_cols + ["Year"]
    )

    # 8. Lưu ra file _imputed.csv
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_imputed_long.to_csv(output_file, index=False, encoding="utf-8")

    total_entities = len(df_pivot)
    imputed_entities = eligible_mask.sum()
    skipped_entities = total_entities - imputed_entities

    print(f"[SUCCESS OWID] Đã xử lý: {input_file.name}")
    print(
        f"               -> Impute thành công (1995-2024): {imputed_entities}/{total_entities} quốc gia/thực thể"
    )
    print(
        f"               -> Bỏ qua (missing > {max_missing_allowed}): {skipped_entities} quốc gia/thực thể"
    )
    print(f"               -> File xuất ra: {output_file.name}\n")


def impute_all_ourworldindata_raw_files() -> None:
    """Hàm chạy Impute cho TẤT CẢ các file Raw Our World in Data có trong config paths."""
    print("=======================================================")
    print("BẮT ĐẦU IMPUTE CÁC FILE RAW OUR WORLD IN DATA (1995 - 2024)")
    print("=======================================================\n")

    owid_file_pairs = [
        (
            paths.RAW_CO2_EMISSIONS_OWID_CSV,
            paths.RAW_CO2_EMISSIONS_OWID_IMPUTED_CSV,
        ),
        (
            paths.RAW_ADULT_SCHOOLING_OWID_CSV,
            paths.RAW_ADULT_SCHOOLING_OWID_IMPUTED_CSV,
        ),
    ]

    for raw_path, imputed_path in owid_file_pairs:
        impute_ourworldindata_raw_file(
            input_file=raw_path, output_file=imputed_path
        )

    print("=> TẤT CẢ FILE RAW OUR WORLD IN DATA ĐÃ ĐƯỢC IMPUTE XONG!")


def process_experiment_1_for_year(
    df_panel: pd.DataFrame, target_year: int
) -> pd.DataFrame:
    # 1. BƯỚC 1: LỌC DỮ LIỆU CỦA DUY NHẤT NĂM CẦN PHÂN TÍCH
    df_year = df_panel[df_panel["year"] == target_year].copy()

    # 2. BƯỚC 2: BIẾN ĐỔI HÀM XỬ LÝ SKEW (Thực hiện độc lập trên năm này)
    # a) Log transform
    log_cols = ["gdp_per_capita_ppp", "population_total"]
    for col in log_cols:
        df_year[col] = np.log(df_year[col])

    # b) Log1p transform
    log1p_cols = [
        "general_government_final_consumption_expenditure_pct_gdp",
        "co2_emissions_per_capita",
        "under_5_mortality_rate_per_1000",
        "trade_pct_gdp",
        "unemployment_rate_pct",
        "industry_pct_gdp",
        "agriculture_pct_gdp",
        "fertility_rate_births_per_woman",
    ]
    for col in log1p_cols:
        df_year[col] = np.log1p(df_year[col])

    # c) Yeo-Johnson cho Lạm phát
    pt = PowerTransformer(method="yeo-johnson")
    df_year["inflation_gdp_deflator_annual_pct"] = pt.fit_transform(
        df_year[["inflation_gdp_deflator_annual_pct"]]
    )

    # d) Lật + Log cho Điện
    max_elec = df_year["access_to_electricity_pct"].max()
    df_year["access_to_electricity_pct"] = np.log(
        (max_elec + 1) - df_year["access_to_electricity_pct"]
    )

    # 3. BƯỚC 3: SCALING (FIT & TRANSFORM CHỈ TRÊN DỮ LIỆU CỦA NĂM NÀY)
    id_cols = ["country_code_3", "country_name", "year"]
    feature_cols = [c for c in df_year.columns if c not in id_cols]

    scaler = StandardScaler()
    df_year[feature_cols] = scaler.fit_transform(df_year[feature_cols])

    return df_year

def preprocess_and_export_all_experiment_1_years(
    input_file: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Đọc file draft panel, xử lý Experiment 1 độc lập từng năm (2010-2024),

    và xuất các file panel_exp1_{year}.csv vào thư mục PROCESSED_DATA_DIR / "experiment_1".
    """
    if input_file is None:
        input_file = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"

    if output_dir is None:
        output_dir = paths.PROCESSED_DATA_EXPERIMENT1_DIR

    if not input_file.exists():
        print(f"[ERROR] Không tìm thấy file dữ liệu gốc: {input_file}")
        return

    # Tạo thư mục xuất dữ liệu nếu chưa tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đọc file panel data
    df_panel = pd.read_csv(input_file)

    # Lấy danh sách các năm có trong file
    available_years = sorted(df_panel["year"].unique())

    print("=======================================================")
    print("BẮT ĐẦU XỬ LÝ VÀ XUẤT DỮ LIỆU CỦA EXPERIMENT 1")
    print("=======================================================\n")

    # 2. Vòng lặp xử lý và xuất CSV cho từng năm
    for year in available_years:
        df_exp1_year = process_experiment_1_for_year(
            df_panel=df_panel, target_year=int(year)
        )

        if df_exp1_year.empty:
            print(f"[SKIP] Không có dữ liệu cho năm {year}")
            continue

        output_path = output_dir / f"panel_exp1_{year}.csv"
        df_exp1_year.to_csv(output_path, index=False, encoding="utf-8")

        print(
            f"✅ [SUCCESS] Năm {year}: {len(df_exp1_year)} quốc gia -> {output_path.name}"
        )

    print("\n=======================================================")
    print(f"📁 TẤT CẢ FILE EXPERIMENT 1 ĐÃ ĐƯỢC LƯU TẠI: {output_dir}")
    print("=======================================================")

    


def preprocess_and_split_experiment_2(
    input_file: Path | None = None,
    output_dir: Path | None = None,
    train_end_year: int = 2018,
    validation_end_year: int = 2020,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tiền xử lý và chia dữ liệu cho Experiment 2 - Classification.

    Split:
        Train      : 2010–2018
        Validation: 2019–2020
        Test       : 2021–2024

    Tất cả transformer và scaler chỉ FIT trên TRAIN
    để tránh data leakage.

    Output:
        experiment_2/
            train.csv
            validation.csv
            test.csv
    """

    if input_file is None:
        input_file = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"

    if output_dir is None:
        output_dir = paths.PROCESSED_DATA_EXPERIMENT2_DIR

    if not input_file.exists():
        print(f"[ERROR] Không tìm thấy file dữ liệu: {input_file}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. ĐỌC DỮ LIỆU
    # ============================================================

    df = pd.read_csv(input_file)

    # ============================================================
    # 2. CHIA TRAIN / VALIDATION / TEST THEO THỜI GIAN
    # ============================================================

    train_df = df[df["year"] <= train_end_year].copy()

    validation_df = df[
        (df["year"] > train_end_year)
        & (df["year"] <= validation_end_year)
    ].copy()

    test_df = df[df["year"] > validation_end_year].copy()

    print("\n=======================================================")
    print("EXPERIMENT 2 - CLASSIFICATION")
    print("=======================================================")
    print(
        f"Train      : {train_df['year'].min()}–{train_df['year'].max()}"
    )
    print(
        f"Validation : {validation_df['year'].min()}–"
        f"{validation_df['year'].max()}"
    )
    print(
        f"Test       : {test_df['year'].min()}–{test_df['year'].max()}"
    )

    # ============================================================
    # 3. TRANSFORMATION
    # ============================================================

    log_cols = [
        "gdp_per_capita_ppp",
        "population_total",
    ]

    log1p_cols = [
        "general_government_final_consumption_expenditure_pct_gdp",
        "co2_emissions_per_capita",
        "under_5_mortality_rate_per_1000",
        "trade_pct_gdp",
        "unemployment_rate_pct",
        "industry_pct_gdp",
        "agriculture_pct_gdp",
        "fertility_rate_births_per_woman",
    ]

    # -------------------------
    # a) Log transform
    # -------------------------

    for col in log_cols:
        train_df[col] = np.log(train_df[col])
        validation_df[col] = np.log(validation_df[col])
        test_df[col] = np.log(test_df[col])

    # -------------------------
    # b) Log1p transform
    # -------------------------

    for col in log1p_cols:
        train_df[col] = np.log1p(train_df[col])
        validation_df[col] = np.log1p(validation_df[col])
        test_df[col] = np.log1p(test_df[col])

    # -------------------------
    # c) Yeo-Johnson
    # -------------------------

    pt = PowerTransformer(method="yeo-johnson")

    train_df["inflation_gdp_deflator_annual_pct"] = pt.fit_transform(
        train_df[["inflation_gdp_deflator_annual_pct"]]
    )

    validation_df["inflation_gdp_deflator_annual_pct"] = pt.transform(
        validation_df[["inflation_gdp_deflator_annual_pct"]]
    )

    test_df["inflation_gdp_deflator_annual_pct"] = pt.transform(
        test_df[["inflation_gdp_deflator_annual_pct"]]
    )

    # -------------------------
    # d) Reflect + Log
    # -------------------------

    max_elec = train_df["access_to_electricity_pct"].max()

    train_df["access_to_electricity_pct"] = np.log(
        (max_elec + 1) - train_df["access_to_electricity_pct"]
    )

    validation_df["access_to_electricity_pct"] = np.log(
        (max_elec + 1) - validation_df["access_to_electricity_pct"]
    )

    test_df["access_to_electricity_pct"] = np.log(
        (max_elec + 1) - test_df["access_to_electricity_pct"]
    )

    # ============================================================
    # 4. SCALING
    # ============================================================

    id_cols = [
        "country_code_3",
        "country_name",
        "year",
    ]

    # GIỮ NGUYÊN LOGIC FEATURE_COLS CỦA BẠN
    feature_cols = [
        c for c in train_df.columns
        if c not in id_cols
    ]

    scaler = StandardScaler()

    train_df[feature_cols] = scaler.fit_transform(
        train_df[feature_cols]
    )

    validation_df[feature_cols] = scaler.transform(
        validation_df[feature_cols]
    )

    test_df[feature_cols] = scaler.transform(
        test_df[feature_cols]
    )

    # ============================================================
    # 5. EXPORT
    # ============================================================

    train_df.to_csv(
        output_dir / "train.csv",
        index=False,
        encoding="utf-8",
    )

    validation_df.to_csv(
        output_dir / "validation.csv",
        index=False,
        encoding="utf-8",
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n[SUCCESS] Experiment 2 đã hoàn thành.")
    print(f"  Train      : {len(train_df)} rows")
    print(f"  Validation : {len(validation_df)} rows")
    print(f"  Test       : {len(test_df)} rows")
    print(f"  Output     : {output_dir}")

    return train_df, validation_df, test_df
'''
def preprocess_and_split_experiment_3(
    input_file: Path | None = None,
    output_dir: Path | None = None,
    train_end_year: int = 2018,
    validation_end_year: int = 2020,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tiền xử lý và tạo sequence cho Experiment 3 - Forecasting.

    Mỗi sample:

        X_t  -> fingerprint tại năm t
        y_t1 -> fingerprint tại năm t+1

    Ví dụ:

        2010 -> 2011
        2011 -> 2012
        ...
        2017 -> 2018

    Split theo TARGET YEAR:

        Train:
            target_year <= 2018

        Validation:
            2019 <= target_year <= 2020

        Test:
            target_year >= 2021

    Output:
        experiment_3/
            train_sequences.csv
            validation_sequences.csv
            test_sequences.csv
    """

    if input_file is None:
        input_file = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"

    if output_dir is None:
        output_dir = paths.PROCESSED_DATA_EXPERIMENT3_DIR

    if not input_file.exists():
        print(f"[ERROR] Không tìm thấy file dữ liệu: {input_file}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. ĐỌC DỮ LIỆU
    # ============================================================

    df = pd.read_csv(input_file)

    df = df.sort_values(
        ["country_code_3", "year"]
    ).reset_index(drop=True)

    # ============================================================
    # 2. CHIA RAW PANEL THEO THỜI GIAN
    # ============================================================

    # Train/Validation/Test vẫn dựa trên target year.
    #
    # Tuy nhiên preprocessing sẽ FIT trên TRAIN trước.
    #
    # Sequence được tạo sau đó.

    # ============================================================
    # 3. TRANSFORMATION
    # ============================================================

    log_cols = [
        "gdp_per_capita_ppp",
        "population_total",
    ]

    log1p_cols = [
        "general_government_final_consumption_expenditure_pct_gdp",
        "co2_emissions_per_capita",
        "under_5_mortality_rate_per_1000",
        "trade_pct_gdp",
        "unemployment_rate_pct",
        "industry_pct_gdp",
        "agriculture_pct_gdp",
        "fertility_rate_births_per_woman",
    ]

    # Fit transformation parameters trên TRAIN PERIOD
    train_mask = df["year"] <= train_end_year

    # -------------------------
    # a) Log
    # -------------------------

    for col in log_cols:
        df.loc[train_mask, col] = np.log(
            df.loc[train_mask, col]
        )

        df.loc[~train_mask, col] = np.log(
            df.loc[~train_mask, col]
        )

    # -------------------------
    # b) Log1p
    # -------------------------

    for col in log1p_cols:
        df.loc[train_mask, col] = np.log1p(
            df.loc[train_mask, col]
        )

        df.loc[~train_mask, col] = np.log1p(
            df.loc[~train_mask, col]
        )

    # -------------------------
    # c) Yeo-Johnson
    # -------------------------

    pt = PowerTransformer(method="yeo-johnson")

    pt.fit(
        df.loc[
            train_mask,
            ["inflation_gdp_deflator_annual_pct"]
        ]
    )

    df.loc[
        train_mask,
        "inflation_gdp_deflator_annual_pct"
    ] = pt.transform(
        df.loc[
            train_mask,
            ["inflation_gdp_deflator_annual_pct"]
        ]
    ).ravel()

    df.loc[
        ~train_mask,
        "inflation_gdp_deflator_annual_pct"
    ] = pt.transform(
        df.loc[
            ~train_mask,
            ["inflation_gdp_deflator_annual_pct"]
        ]
    ).ravel()

    # -------------------------
    # d) Reflect + Log
    # -------------------------

    max_elec = df.loc[
        train_mask,
        "access_to_electricity_pct"
    ].max()

    df.loc[
        train_mask,
        "access_to_electricity_pct"
    ] = np.log(
        (max_elec + 1)
        - df.loc[
            train_mask,
            "access_to_electricity_pct"
        ]
    )

    df.loc[
        ~train_mask,
        "access_to_electricity_pct"
    ] = np.log(
        (max_elec + 1)
        - df.loc[
            ~train_mask,
            "access_to_electricity_pct"
        ]
    )

    # ============================================================
    # 4. SCALE
    # ============================================================

    id_cols = [
        "country_code_3",
        "country_name",
        "year",
    ]

    # GIỮ NGUYÊN feature_cols
    feature_cols = [
        c for c in df.columns
        if c not in id_cols
    ]

    scaler = StandardScaler()

    scaler.fit(
        df.loc[
            train_mask,
            feature_cols
        ]
    )

    df.loc[
        :,
        feature_cols
    ] = scaler.transform(
        df[feature_cols]
    )

    # ============================================================
    # 5. TẠO LAG → NEXT-YEAR SEQUENCE
    # ============================================================

    sequence_rows = []

    for country_code, country_df in df.groupby(
        "country_code_3"
    ):

        country_df = country_df.sort_values("year")

        for i in range(len(country_df) - 1):

            current = country_df.iloc[i]
            next_row = country_df.iloc[i + 1]

            current_year = int(current["year"])
            target_year = int(next_row["year"])

            # Chỉ tạo sequence nếu thực sự là năm kế tiếp
            if target_year != current_year + 1:
                continue

            row = {
                "country_code_3": country_code,
                "country_name": current["country_name"],
                "year": current_year,
                "target_year": target_year,
            }

            # X_t
            for col in feature_cols:
                row[f"x_{col}"] = current[col]

            # y_(t+1)
            for col in feature_cols:
                row[f"y_{col}"] = next_row[col]

            sequence_rows.append(row)

    sequences = pd.DataFrame(sequence_rows)

    # ============================================================
    # 6. SPLIT THEO TARGET YEAR
    # ============================================================

    train_sequences = sequences[
        sequences["target_year"] <= train_end_year
    ].copy()

    validation_sequences = sequences[
        (sequences["target_year"] > train_end_year)
        & (sequences["target_year"] <= validation_end_year)
    ].copy()

    test_sequences = sequences[
        sequences["target_year"] > validation_end_year
    ].copy()

    # ============================================================
    # 7. EXPORT
    # ============================================================

    train_sequences.to_csv(
        output_dir / "train_sequences.csv",
        index=False,
        encoding="utf-8",
    )

    validation_sequences.to_csv(
        output_dir / "validation_sequences.csv",
        index=False,
        encoding="utf-8",
    )

    test_sequences.to_csv(
        output_dir / "test_sequences.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n=======================================================")
    print("EXPERIMENT 3 - FORECASTING")
    print("=======================================================")

    print(
        f"Train sequences      : "
        f"{len(train_sequences)}"
    )

    print(
        f"Validation sequences : "
        f"{len(validation_sequences)}"
    )

    print(
        f"Test sequences       : "
        f"{len(test_sequences)}"
    )

    print(f"Output: {output_dir}")

    return (
        train_sequences,
        validation_sequences,
        test_sequences,
    )
'''

def preprocess_and_split_experiment_3(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    train_end_year: int = 2017,
    val_end_year: int = 2019,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tiền xử lý và chia dữ liệu cho Experiment 3 - Forecasting.

    Quy trình:
    1. Đọc 3 file (train.csv, validation.csv, test.csv) từ PROCESSED_DATA_EXPERIMENT2_DIR.
    2. Gộp 3 file lại thành full_df để không bị đứt đoạn năm ranh giới khi shift(-1).
    3. Tạo 20 cột Target Y cho năm t+1 (shift(-1) theo country_code_3).
    4. Loại bỏ các dòng năm 2024 (do thiếu năm 2025 làm Target Y).
    5. Tách lại 3 tập Train / Validation / Test theo mốc thời gian:
        - Train     : year <= 2017 (Năm t: 2010–2017 -> Dự báo t+1: 2011–2018)
        - Validation: 2017 < year <= 2019 (Năm t: 2018–2019 -> Dự báo t+1: 2019–2020)
        - Test      : year > 2019 (Năm t: 2020–2023 -> Dự báo t+1: 2021–2024)
    6. Xuất ra 3 file train.csv, validation.csv, test.csv tại PROCESSED_DATA_EXPERIMENT3_DIR.
    """
    if input_dir is None:
        input_dir = paths.PROCESSED_DATA_EXPERIMENT2_DIR

    if output_dir is None:
        output_dir = paths.PROCESSED_DATA_EXPERIMENT3_DIR

    # 1. KIỂM TRA FILE ĐẦU VÀO TỪ EXP 2
    train_path = input_dir / "train.csv"
    val_path = input_dir / "validation.csv"
    test_path = input_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            print(f"[ERROR] Không tìm thấy file dữ liệu Exp 2: {p}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("EXPERIMENT 3 - FORECASTING PREPROCESSING")
    print("=======================================================")

    # 2. ĐỌC VÀ GỘP 3 FILE DỮ LIỆU TỪ EXP 2
    exp2_train = pd.read_csv(train_path)
    exp2_val = pd.read_csv(val_path)
    exp2_test = pd.read_csv(test_path)

    full_df = pd.concat([exp2_train, exp2_val, exp2_test], ignore_index=True)

    # Bắt buộc sắp xếp theo Quốc gia và Năm
    full_df = full_df.sort_values(["country_code_3", "year"]).reset_index(drop=True)

    # 3. XÁC ĐỊNH CÁC CỘT FEATURES (X) VÀ TẠO CỦA CỘT TARGET (Y cho năm t+1)
    id_cols = ["country_code_3", "country_name", "year"]
    feature_cols = [c for c in full_df.columns if c not in id_cols]
    target_cols = [f"{c}_target_next_year" for c in feature_cols]

    # Shift (-1) theo từng country_code_3 để lấy đáp án năm t+1
    for col, t_col in zip(feature_cols, target_cols):
        full_df[t_col] = full_df.groupby("country_code_3")[col].shift(-1)

    # Loại bỏ năm 2024 (bị NaN ở các cột Target Y do chưa có dữ liệu năm 2025)
    full_df_clean = full_df.dropna(subset=target_cols).reset_index(drop=True)

    # 4. CHIA LẠI TRAIN / VALIDATION / TEST THEO THỜI GIAN (NĂM t)
    train_df = full_df_clean[full_df_clean["year"] <= train_end_year].copy()

    validation_df = full_df_clean[
        (full_df_clean["year"] > train_end_year)
        & (full_df_clean["year"] <= val_end_year)
    ].copy()

    test_df = full_df_clean[full_df_clean["year"] > val_end_year].copy()

    # 5. XUẤT RA 3 FILE CSV TẠI PROCESSED_DATA_EXPERIMENT3_DIR
    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8")
    validation_df.to_csv(output_dir / "validation.csv", index=False, encoding="utf-8")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8")

    print("✅ [SUCCESS] Tiền xử lý dữ liệu cho Experiment 3 hoàn tất!")
    print(f"  Train      : {len(train_df)} dòng | Năm t: {train_df['year'].min()}–{train_df['year'].max()} (Dự báo t+1: {train_df['year'].min()+1}–{train_df['year'].max()+1})")
    print(f"  Validation : {len(validation_df)} dòng | Năm t: {validation_df['year'].min()}–{validation_df['year'].max()} (Dự báo t+1: {validation_df['year'].min()+1}–{validation_df['year'].max()+1})")
    print(f"  Test       : {len(test_df)} dòng | Năm t: {test_df['year'].min()}–{test_df['year'].max()} (Dự báo t+1: {test_df['year'].min()+1}–{test_df['year'].max()+1})")
    print(f"  Tổng số cột: {full_df_clean.shape[1]} (3 Cột ID + {len(feature_cols)} Cột X + {len(target_cols)} Cột Y)")
    print(f"  Thư mục    : {output_dir}")

    return train_df, validation_df, test_df

if __name__ == "__main__":
    #impute_all_worldbank_raw_files()
    #impute_all_ourworldindata_raw_files()

    panel_df = combine_all_drafts_to_panel_data(
        drop_countries=False,
        drop_countries_with_missing=True
    )

    preprocess_and_export_all_experiment_1_years()

    preprocess_and_split_experiment_2()
    
    preprocess_and_split_experiment_3()


