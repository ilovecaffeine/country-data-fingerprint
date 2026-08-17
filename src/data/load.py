# src/data/load.py
# cd "C:\Users\admin\Documents\Code_for_fun\country-data-fingerprint"
# python -m src.data.load
from pathlib import Path
import shutil
import pandas as pd
from config import paths


def populate_draft_files() -> None:
    """Sao chép toàn bộ nội dung từ file countries_194.csv

    dán/ghi đè vào tất cả các file draft từ năm 2010 đến 2024.
    """
    source_file = paths.COUNTRY_LIST_CSV

    if not source_file.exists():
        print(f"[ERROR] File nguồn không tồn tại: {source_file}")
        return

    for year in paths.YEARS:
        target_file = paths.get_draft_csv_path(year)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target_file)
        print(f"[COPY] Đã dán nội dung vào: {target_file.name}")

    print("=> Hoàn thành copy dữ liệu vào tất cả các file draft!")


def clear_draft_files() -> None:
    """Xóa sạch nội dung (làm rỗng file) của tất cả các file draft

    từ năm 2010 đến 2024 mà không xóa bản thân file.
    """
    for year in paths.YEARS:
        target_file = paths.get_draft_csv_path(year)

        if target_file.exists():
            with open(target_file, "w", encoding="utf-8"):
                pass
            print(f"[CLEAR] Đã xóa sạch nội dung file: {target_file.name}")
        else:
            print(f"[SKIP] File không tồn tại: {target_file.name}")

    print("=> Hoàn thành xóa nội dung tất cả các file draft!")


def process_and_join_gdp_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu GDP, lọc theo 194 quốc gia chủ quyền,

    sau đó ghép (join) cột GDP của từng năm (2010-2024)
    vào file draft_{year}.csv tương ứng.
    """
    if input_file is not None:
        gdp_raw_file = input_file
    else:
        gdp_raw_file = (
            paths.RAW_GDP_IMPUTED_CSV if use_imputed else paths.RAW_GDP_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý GDP [{file_status}]: {gdp_raw_file.name}")

    if not gdp_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_gdp_raw = pd.read_csv(gdp_raw_file, skiprows=4)
    df_gdp_filtered = df_gdp_raw[df_gdp_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_gdp_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "gdp_per_capita_ppp" in df_draft.columns:
            df_draft = df_draft.drop(columns=["gdp_per_capita_ppp"])

        df_gdp_year = df_gdp_filtered[["Country Code", year_str]].rename(
            columns={year_str: "gdp_per_capita_ppp"}
        )
        df_merged = pd.merge(df_draft, df_gdp_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'gdp_per_capita_ppp' ({file_status}) vào {draft_file.name}")


def process_and_join_gdp_growth_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tăng trưởng GDP, lọc theo 194 quốc gia,

    và ghép (join) cột tăng trưởng của từng năm vào file draft_{year}.csv
    """
    if input_file is not None:
        gdp_growth_raw_file = input_file
    else:
        gdp_growth_raw_file = (
            paths.RAW_GDP_GROWTH_IMPUTED_CSV
            if use_imputed
            else paths.RAW_GDP_GROWTH_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý GDP Growth [{file_status}]: {gdp_growth_raw_file.name}")

    if not gdp_growth_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_gdp_growth_raw = pd.read_csv(gdp_growth_raw_file, skiprows=4)
    df_gdp_growth_filtered = df_gdp_growth_raw[df_gdp_growth_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_gdp_growth_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "gdp_growth_annual_pct" in df_draft.columns:
            df_draft = df_draft.drop(columns=["gdp_growth_annual_pct"])

        df_growth_year = df_gdp_growth_filtered[["Country Code", year_str]].rename(
            columns={year_str: "gdp_growth_annual_pct"}
        )
        df_merged = pd.merge(df_draft, df_growth_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'gdp_growth_annual_pct' ({file_status}) vào {draft_file.name}")


def process_and_join_govt_expenditure_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Chi tiêu tiêu dùng cuối cùng của chính phủ (% GDP)."""
    if input_file is not None:
        govt_raw_file = input_file
    else:
        govt_raw_file = (
            paths.RAW_GOVT_EXPENDITURE_IMPUTED_CSV
            if use_imputed
            else paths.RAW_GOVT_EXPENDITURE_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Chi tiêu chính phủ [{file_status}]: {govt_raw_file.name}")

    if not govt_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_govt_raw = pd.read_csv(govt_raw_file, skiprows=4)
    df_govt_filtered = df_govt_raw[df_govt_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_govt_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        col_name = "general_government_final_consumption_expenditure_pct_gdp"
        if col_name in df_draft.columns:
            df_draft = df_draft.drop(columns=[col_name])

        df_govt_year = df_govt_filtered[["Country Code", year_str]].rename(columns={year_str: col_name})
        df_merged = pd.merge(df_draft, df_govt_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột '{col_name}' ({file_status}) vào {draft_file.name}")


def process_and_join_inflation_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Lạm phát."""
    if input_file is not None:
        inflation_raw_file = input_file
    else:
        inflation_raw_file = (
            paths.RAW_INFLATION_IMPUTED_CSV
            if use_imputed
            else paths.RAW_INFLATION_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Lạm phát [{file_status}]: {inflation_raw_file.name}")

    if not inflation_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_inflation_raw = pd.read_csv(inflation_raw_file, skiprows=4)
    df_inflation_filtered = df_inflation_raw[df_inflation_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_inflation_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "inflation_gdp_deflator_annual_pct" in df_draft.columns:
            df_draft = df_draft.drop(columns=["inflation_gdp_deflator_annual_pct"])

        df_inflation_year = df_inflation_filtered[["Country Code", year_str]].rename(
            columns={year_str: "inflation_gdp_deflator_annual_pct"}
        )
        df_merged = pd.merge(df_draft, df_inflation_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'inflation_gdp_deflator_annual_pct' ({file_status}) vào {draft_file.name}")


def process_and_join_unemployment_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tỷ lệ Thất nghiệp."""
    if input_file is not None:
        unemployment_raw_file = input_file
    else:
        unemployment_raw_file = (
            paths.RAW_UNEMPLOYMENT_IMPUTED_CSV
            if use_imputed
            else paths.RAW_UNEMPLOYMENT_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Thất nghiệp [{file_status}]: {unemployment_raw_file.name}")

    if not unemployment_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_unemployment_raw = pd.read_csv(unemployment_raw_file, skiprows=4)
    df_unemployment_filtered = df_unemployment_raw[df_unemployment_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_unemployment_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "unemployment_rate_pct" in df_draft.columns:
            df_draft = df_draft.drop(columns=["unemployment_rate_pct"])

        df_unemployment_year = df_unemployment_filtered[["Country Code", year_str]].rename(
            columns={year_str: "unemployment_rate_pct"}
        )
        df_merged = pd.merge(df_draft, df_unemployment_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'unemployment_rate_pct' ({file_status}) vào {draft_file.name}")


def process_and_join_population_growth_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tốc độ tăng dân số."""
    if input_file is not None:
        pop_growth_raw_file = input_file
    else:
        pop_growth_raw_file = (
            paths.RAW_POPULATION_GROWTH_IMPUTED_CSV
            if use_imputed
            else paths.RAW_POPULATION_GROWTH_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Tăng dân số [{file_status}]: {pop_growth_raw_file.name}")

    if not pop_growth_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_pop_growth_raw = pd.read_csv(pop_growth_raw_file, skiprows=4)
    df_pop_growth_filtered = df_pop_growth_raw[df_pop_growth_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_pop_growth_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "population_growth_annual_pct" in df_draft.columns:
            df_draft = df_draft.drop(columns=["population_growth_annual_pct"])

        df_pop_growth_year = df_pop_growth_filtered[["Country Code", year_str]].rename(
            columns={year_str: "population_growth_annual_pct"}
        )
        df_merged = pd.merge(df_draft, df_pop_growth_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'population_growth_annual_pct' ({file_status}) vào {draft_file.name}")


def process_and_join_population_total_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tổng dân số."""
    if input_file is not None:
        pop_total_raw_file = input_file
    else:
        pop_total_raw_file = (
            paths.RAW_POPULATION_TOTAL_IMPUTED_CSV
            if use_imputed
            else paths.RAW_POPULATION_TOTAL_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Tổng dân số [{file_status}]: {pop_total_raw_file.name}")

    if not pop_total_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_pop_total_raw = pd.read_csv(pop_total_raw_file, skiprows=4)
    df_pop_total_filtered = df_pop_total_raw[df_pop_total_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_pop_total_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "population_total" in df_draft.columns:
            df_draft = df_draft.drop(columns=["population_total"])

        df_pop_total_year = df_pop_total_filtered[["Country Code", year_str]].rename(
            columns={year_str: "population_total"}
        )
        df_merged = pd.merge(df_draft, df_pop_total_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'population_total' ({file_status}) vào {draft_file.name}")


def process_and_join_urban_population_pct_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tỷ lệ dân số đô thị."""
    if input_file is not None:
        urb_pct_raw_file = input_file
    else:
        urb_pct_raw_file = (
            paths.RAW_URBAN_POPULATION_PCT_IMPUTED_CSV
            if use_imputed
            else paths.RAW_URBAN_POPULATION_PCT_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Dân số đô thị (%)[{file_status}]: {urb_pct_raw_file.name}")

    if not urb_pct_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_urb_pct_raw = pd.read_csv(urb_pct_raw_file, skiprows=4)
    df_urb_pct_filtered = df_urb_pct_raw[df_urb_pct_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_urb_pct_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "urban_population_pct" in df_draft.columns:
            df_draft = df_draft.drop(columns=["urban_population_pct"])

        df_urb_pct_year = df_urb_pct_filtered[["Country Code", year_str]].rename(
            columns={year_str: "urban_population_pct"}
        )
        df_merged = pd.merge(df_draft, df_urb_pct_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'urban_population_pct' ({file_status}) vào {draft_file.name}")


def process_and_join_urban_population_growth_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tốc độ tăng dân số đô thị."""
    if input_file is not None:
        urb_grow_raw_file = input_file
    else:
        urb_grow_raw_file = (
            paths.RAW_URBAN_POPULATION_GROWTH_IMPUTED_CSV
            if use_imputed
            else paths.RAW_URBAN_POPULATION_GROWTH_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Tăng dân số đô thị [{file_status}]: {urb_grow_raw_file.name}")

    if not urb_grow_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_urb_grow_raw = pd.read_csv(urb_grow_raw_file, skiprows=4)
    df_urb_grow_filtered = df_urb_grow_raw[df_urb_grow_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_urb_grow_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "urban_population_growth_annual_pct" in df_draft.columns:
            df_draft = df_draft.drop(columns=["urban_population_growth_annual_pct"])

        df_urb_grow_year = df_urb_grow_filtered[["Country Code", year_str]].rename(
            columns={year_str: "urban_population_growth_annual_pct"}
        )
        df_merged = pd.merge(df_draft, df_urb_grow_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'urban_population_growth_annual_pct' ({file_status}) vào {draft_file.name}")


def process_and_join_life_expectancy_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tuổi thọ trung bình khi sinh."""
    if input_file is not None:
        life_exp_raw_file = input_file
    else:
        life_exp_raw_file = (
            paths.RAW_LIFE_EXPECTANCY_IMPUTED_CSV
            if use_imputed
            else paths.RAW_LIFE_EXPECTANCY_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Tuổi thọ trung bình [{file_status}]: {life_exp_raw_file.name}")

    if not life_exp_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_life_exp_raw = pd.read_csv(life_exp_raw_file, skiprows=4)
    df_life_exp_filtered = df_life_exp_raw[df_life_exp_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_life_exp_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "life_expectancy_years" in df_draft.columns:
            df_draft = df_draft.drop(columns=["life_expectancy_years"])

        df_life_exp_year = df_life_exp_filtered[["Country Code", year_str]].rename(
            columns={year_str: "life_expectancy_years"}
        )
        df_merged = pd.merge(df_draft, df_life_exp_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'life_expectancy_years' ({file_status}) vào {draft_file.name}")


def process_and_join_fertility_rate_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tỷ lệ sinh."""
    if input_file is not None:
        fertility_raw_file = input_file
    else:
        fertility_raw_file = (
            paths.RAW_FERTILITY_RATE_IMPUTED_CSV
            if use_imputed
            else paths.RAW_FERTILITY_RATE_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Tỷ lệ sinh [{file_status}]: {fertility_raw_file.name}")

    if not fertility_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_fertility_raw = pd.read_csv(fertility_raw_file, skiprows=4)
    df_fertility_filtered = df_fertility_raw[df_fertility_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_fertility_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "fertility_rate_births_per_woman" in df_draft.columns:
            df_draft = df_draft.drop(columns=["fertility_rate_births_per_woman"])

        df_fertility_year = df_fertility_filtered[["Country Code", year_str]].rename(
            columns={year_str: "fertility_rate_births_per_woman"}
        )
        df_merged = pd.merge(df_draft, df_fertility_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'fertility_rate_births_per_woman' ({file_status}) vào {draft_file.name}")


def process_and_join_under_5_mortality_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tỷ lệ tử vong ở trẻ dưới 5 tuổi."""
    if input_file is not None:
        mortality_raw_file = input_file
    else:
        mortality_raw_file = (
            paths.RAW_UNDER_5_MORTALITY_IMPUTED_CSV
            if use_imputed
            else paths.RAW_UNDER_5_MORTALITY_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Tử vong dưới 5 tuổi [{file_status}]: {mortality_raw_file.name}")

    if not mortality_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_mortality_raw = pd.read_csv(mortality_raw_file, skiprows=4)
    df_mortality_filtered = df_mortality_raw[df_mortality_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_mortality_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "under_5_mortality_rate_per_1000" in df_draft.columns:
            df_draft = df_draft.drop(columns=["under_5_mortality_rate_per_1000"])

        df_mortality_year = df_mortality_filtered[["Country Code", year_str]].rename(
            columns={year_str: "under_5_mortality_rate_per_1000"}
        )
        df_merged = pd.merge(df_draft, df_mortality_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'under_5_mortality_rate_per_1000' ({file_status}) vào {draft_file.name}")


def process_and_join_access_to_electricity_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tỷ lệ tiếp cận điện lưới."""
    if input_file is not None:
        elec_raw_file = input_file
    else:
        elec_raw_file = (
            paths.RAW_ACCESS_TO_ELECTRICITY_IMPUTED_CSV
            if use_imputed
            else paths.RAW_ACCESS_TO_ELECTRICITY_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Tiếp cận điện lưới [{file_status}]: {elec_raw_file.name}")

    if not elec_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_elec_raw = pd.read_csv(elec_raw_file, skiprows=4)
    df_elec_filtered = df_elec_raw[df_elec_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_elec_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "access_to_electricity_pct" in df_draft.columns:
            df_draft = df_draft.drop(columns=["access_to_electricity_pct"])

        df_elec_year = df_elec_filtered[["Country Code", year_str]].rename(
            columns={year_str: "access_to_electricity_pct"}
        )
        df_merged = pd.merge(df_draft, df_elec_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'access_to_electricity_pct' ({file_status}) vào {draft_file.name}")

def process_and_join_owid_co2_emissions_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu CO2 emissions per capita từ OWID (Long format)."""
    if input_file is not None:
        co2_raw_file = input_file
    else:
        co2_raw_file = (
            paths.RAW_CO2_EMISSIONS_OWID_IMPUTED_CSV
            if use_imputed
            else paths.RAW_CO2_EMISSIONS_OWID_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Phát thải CO2 OWID [{file_status}]: {co2_raw_file.name}")

    if not co2_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_co2_raw = pd.read_csv(co2_raw_file)

    df_co2_filtered = df_co2_raw[
        (df_co2_raw["Code"].isin(df_countries["country_code_3"]))
        & (df_co2_raw["Year"].isin(paths.YEARS))
    ].copy()

    df_co2_pivot = df_co2_filtered.pivot(index="Code", columns="Year", values="CO₂ emissions per capita")

    for year in paths.YEARS:
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year not in df_co2_pivot.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "co2_emissions_per_capita" in df_draft.columns:
            df_draft = df_draft.drop(columns=["co2_emissions_per_capita"])

        df_co2_year = df_co2_pivot[[year]].reset_index()
        df_co2_year.columns = ["Code", "co2_emissions_per_capita"]

        df_merged = pd.merge(df_draft, df_co2_year, left_on="country_code_3", right_on="Code", how="left")
        df_merged = df_merged.drop(columns=["Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'co2_emissions_per_capita' ({file_status}) vào {draft_file.name}")


def process_and_join_adult_schooling_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Số năm đi học trung bình từ OWID (Long format)."""
    if input_file is not None:
        schooling_raw_file = input_file
    else:
        schooling_raw_file = (
            paths.RAW_ADULT_SCHOOLING_OWID_IMPUTED_CSV
            if use_imputed
            else paths.RAW_ADULT_SCHOOLING_OWID_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Số năm đi học OWID [{file_status}]: {schooling_raw_file.name}")

    if not schooling_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_schooling_raw = pd.read_csv(schooling_raw_file)

    df_schooling_filtered = df_schooling_raw[
        (df_schooling_raw["Code"].isin(df_countries["country_code_3"]))
        & (df_schooling_raw["Year"].isin(paths.YEARS))
    ].copy()

    df_schooling_pivot = df_schooling_filtered.pivot(index="Code", columns="Year", values="Both genders")

    for year in paths.YEARS:
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year not in df_schooling_pivot.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "mean_years_of_schooling_adults" in df_draft.columns:
            df_draft = df_draft.drop(columns=["mean_years_of_schooling_adults"])

        df_schooling_year = df_schooling_pivot[[year]].reset_index()
        df_schooling_year.columns = ["Code", "mean_years_of_schooling_adults"]

        df_merged = pd.merge(df_draft, df_schooling_year, left_on="country_code_3", right_on="Code", how="left")
        df_merged = df_merged.drop(columns=["Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'mean_years_of_schooling_adults' ({file_status}) vào {draft_file.name}")

def process_and_join_internet_users_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tỷ lệ người dùng Internet."""
    if input_file is not None:
        net_raw_file = input_file
    else:
        net_raw_file = (
            paths.RAW_INTERNET_USERS_IMPUTED_CSV
            if use_imputed
            else paths.RAW_INTERNET_USERS_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Người dùng Internet [{file_status}]: {net_raw_file.name}")

    if not net_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_net_raw = pd.read_csv(net_raw_file, skiprows=4)
    df_net_filtered = df_net_raw[df_net_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_net_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "internet_users_pct_population" in df_draft.columns:
            df_draft = df_draft.drop(columns=["internet_users_pct_population"])

        df_net_year = df_net_filtered[["Country Code", year_str]].rename(
            columns={year_str: "internet_users_pct_population"}
        )
        df_merged = pd.merge(df_draft, df_net_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'internet_users_pct_population' ({file_status}) vào {draft_file.name}")


def process_and_join_trade_pct_gdp_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tỷ lệ Thương mại (% GDP)."""
    if input_file is not None:
        trade_raw_file = input_file
    else:
        trade_raw_file = (
            paths.RAW_TRADE_IMPUTED_CSV
            if use_imputed
            else paths.RAW_TRADE_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Thương mại (% GDP) [{file_status}]: {trade_raw_file.name}")

    if not trade_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_trade_raw = pd.read_csv(trade_raw_file, skiprows=4)
    df_trade_filtered = df_trade_raw[df_trade_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_trade_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "trade_pct_gdp" in df_draft.columns:
            df_draft = df_draft.drop(columns=["trade_pct_gdp"])

        df_trade_year = df_trade_filtered[["Country Code", year_str]].rename(
            columns={year_str: "trade_pct_gdp"}
        )
        df_merged = pd.merge(df_draft, df_trade_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'trade_pct_gdp' ({file_status}) vào {draft_file.name}")


def process_and_join_agriculture_pct_gdp_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tỷ trọng Nông nghiệp trong GDP."""
    if input_file is not None:
        agr_raw_file = input_file
    else:
        agr_raw_file = (
            paths.RAW_AGRICULTURE_IMPUTED_CSV
            if use_imputed
            else paths.RAW_AGRICULTURE_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Nông nghiệp (% GDP) [{file_status}]: {agr_raw_file.name}")

    if not agr_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_agr_raw = pd.read_csv(agr_raw_file, skiprows=4)
    df_agr_filtered = df_agr_raw[df_agr_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_agr_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "agriculture_pct_gdp" in df_draft.columns:
            df_draft = df_draft.drop(columns=["agriculture_pct_gdp"])

        df_agr_year = df_agr_filtered[["Country Code", year_str]].rename(
            columns={year_str: "agriculture_pct_gdp"}
        )
        df_merged = pd.merge(df_draft, df_agr_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'agriculture_pct_gdp' ({file_status}) vào {draft_file.name}")


def process_and_join_industry_pct_gdp_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tỷ trọng Công nghiệp trong GDP."""
    if input_file is not None:
        ind_raw_file = input_file
    else:
        ind_raw_file = (
            paths.RAW_INDUSTRY_IMPUTED_CSV
            if use_imputed
            else paths.RAW_INDUSTRY_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý Công nghiệp (% GDP) [{file_status}]: {ind_raw_file.name}")

    if not ind_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File không tồn tại!")
        return

    df_countries = pd.read_csv(country_file)
    df_ind_raw = pd.read_csv(ind_raw_file, skiprows=4)
    df_ind_filtered = df_ind_raw[df_ind_raw["Country Code"].isin(df_countries["country_code_3"])].copy()

    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)
        if not draft_file.exists() or year_str not in df_ind_filtered.columns:
            continue

        df_draft = pd.read_csv(draft_file)
        if "industry_pct_gdp" in df_draft.columns:
            df_draft = df_draft.drop(columns=["industry_pct_gdp"])

        df_ind_year = df_ind_filtered[["Country Code", year_str]].rename(
            columns={year_str: "industry_pct_gdp"}
        )
        df_merged = pd.merge(df_draft, df_ind_year, left_on="country_code_3", right_on="Country Code", how="left")
        df_merged = df_merged.drop(columns=["Country Code"], errors="ignore")
        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(f"[SUCCESS] Đã ghép cột 'industry_pct_gdp' ({file_status}) vào {draft_file.name}")


def process_and_join_services_pct_gdp_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Đọc dữ liệu Tỷ trọng Dịch vụ trong GDP (Imputed hoặc Chưa Imputed),

    lọc theo 194 quốc gia, và ghép (join) cột của từng năm (2010-2024)
    vào file draft_{year}.csv với tên cột 'services_pct_gdp'.

    Parameters
    ----------
    use_imputed : bool, default=True
        Nếu True, đọc file đã Impute (paths.RAW_SERVICES_IMPUTED_CSV).
        Nếu False, đọc file chưa Impute gốc (paths.RAW_SERVICES_CSV).
    input_file : Path, optional
        Tùy chọn truyền đường dẫn file trực tiếp (nếu muốn chỉ định file riêng).
    """
    # 1. Xác định đường dẫn file dựa vào cờ use_imputed
    if input_file is not None:
        srv_raw_file = input_file
    else:
        srv_raw_file = (
            paths.RAW_SERVICES_IMPUTED_CSV
            if use_imputed
            else paths.RAW_SERVICES_CSV 
        )

    country_file = paths.COUNTRY_LIST_CSV

    file_status = "Đã Impute" if use_imputed else "Chưa Impute (Raw)"
    print(f"🔄 Đang xử lý file Dịch vụ (% GDP) [{file_status}]: {srv_raw_file.name}")

    if not srv_raw_file.exists():
        print(f"[ERROR] Không tìm thấy file Dịch vụ: {srv_raw_file}")
        return
    if not country_file.exists():
        print(f"[ERROR] Không tìm thấy file 194 quốc gia: {country_file}")
        return

    # 2. Đọc dữ liệu
    df_countries = pd.read_csv(country_file)
    df_srv_raw = pd.read_csv(srv_raw_file, skiprows=4)

    df_srv_filtered = df_srv_raw[
        df_srv_raw["Country Code"].isin(df_countries["country_code_3"])
    ].copy()

    # 3. Ghép vào từng file draft_{year}.csv
    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)

        if not draft_file.exists():
            print(f"[SKIP] File draft chưa tồn tại: {draft_file.name}")
            continue

        if year_str not in df_srv_filtered.columns:
            print(f"[WARNING] Năm {year_str} không có trong file Dịch vụ!")
            continue

        df_draft = pd.read_csv(draft_file)

        # Nếu cột 'services_pct_gdp' đã tồn tại trước đó, xóa đi để đè dữ liệu mới
        if "services_pct_gdp" in df_draft.columns:
            df_draft = df_draft.drop(columns=["services_pct_gdp"])

        df_srv_year = df_srv_filtered[["Country Code", year_str]].rename(
            columns={year_str: "services_pct_gdp"}
        )

        df_merged = pd.merge(
            df_draft,
            df_srv_year,
            left_on="country_code_3",
            right_on="Country Code",
            how="left",
        )

        if "Country Code" in df_merged.columns:
            df_merged = df_merged.drop(columns=["Country Code"])

        df_merged.to_csv(draft_file, index=False, encoding="utf-8")
        print(
            f"[SUCCESS] Đã ghép cột 'services_pct_gdp' ({file_status}) vào file {draft_file.name}"
        )

    print(f"=> Hoàn tất quá trình xử lý và ghép dữ liệu Dịch vụ (% GDP) [{file_status}]!")

def build_and_populate_all_draft_data(
    use_imputed: bool = True,
    clear_existing: bool = True,
) -> None:
    """Hàm tổng hợp: Khởi tạo các file draft và nạp/ghép toàn bộ 20 chỉ số kinh tế - xã hội.

    Parameters
    ----------
    use_imputed : bool, default=True
        Nếu True, nạp các file dữ liệu đã Impute cho tất cả các chỉ số.
        Nếu False, nạp các file dữ liệu gốc chưa Impute (Raw).
    clear_existing : bool, default=True
        Nếu True, xóa sạch nội dung các file draft cũ trước khi nạp lại.
    """
    file_status = "ĐÃ IMPUTE" if use_imputed else "CHƯA IMPUTE (RAW)"
    
    print("=======================================================")
    print(f"🚀 BẮT ĐẦU XỬ LÝ VÀ GHÉP TẤT CẢ CHỈ SỐ DRAFT [{file_status}]")
    print("=======================================================\n")

    # 1. Xóa nội dung draft cũ nếu bật cờ clear_existing
    if clear_existing:
        print("--- Xóa nội dung draft ---")
        clear_draft_files()

    # 2. Khởi tạo khung file draft
    print("--- Nạp dữ liệu nền vào draft ---")
    populate_draft_files()

    print(f"\n--- Bắt đầu ghép 20 chỉ số [{file_status}] ---")

    # 3. Danh sách tất cả các hàm ghép chỉ số
    join_functions = [
        process_and_join_gdp_data,
        process_and_join_gdp_growth_data,
        process_and_join_govt_expenditure_data,
        process_and_join_inflation_data,
        process_and_join_unemployment_data,
        process_and_join_population_total_data,
        process_and_join_population_growth_data,
        process_and_join_urban_population_pct_data,
        process_and_join_urban_population_growth_data,
        process_and_join_life_expectancy_data,
        process_and_join_fertility_rate_data,
        process_and_join_under_5_mortality_data,
        process_and_join_access_to_electricity_data,
        process_and_join_owid_co2_emissions_data,
        process_and_join_adult_schooling_data,
        process_and_join_internet_users_data,
        process_and_join_trade_pct_gdp_data,
        process_and_join_agriculture_pct_gdp_data,
        process_and_join_industry_pct_gdp_data,
        process_and_join_services_pct_gdp_data,
    ]

    # 4. Vòng lặp tự động gọi từng hàm với cờ use_imputed
    for func in join_functions:
        try:
            func(use_imputed=use_imputed)
        except TypeError:
            # Nếu hàm đó không nhận tham số use_imputed (ví dụ OWID CO2) thì gọi hàm không tham số
            func()

    print("\n=======================================================")
    print(f"🎉 HOÀN TẤT TẤT CẢ BƯỚC NẠP VÀ GHÉP DỮ LIỆU DRAFT [{file_status}]!")
    print("=======================================================")

if __name__ == "__main__":
    # Chạy ghép toàn bộ dữ liệu CHƯA IMPUTE (Gốc) hoặc dữ liệu ĐÃ IMPUTE
    build_and_populate_all_draft_data(use_imputed=False)