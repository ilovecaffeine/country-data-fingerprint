# src/data/load.py
# cd country-data-fingerprint
# python -m src.data.load
from pathlib import Path
import shutil
import pandas as pd
from config import paths


def populate_draft_files() -> None:
    """Copy all content from countries_194.csv

    and paste/overwrite into all draft files from 2010 to 2024.
    """
    source_file = paths.COUNTRY_LIST_CSV

    if not source_file.exists():
        print(f"[ERROR] Source file does not exist: {source_file}")
        return

    for year in paths.YEARS:
        target_file = paths.get_draft_csv_path(year)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target_file)
        print(f"[COPY] Content pasted into: {target_file.name}")

    print("=> Completed copying data into all draft files!")


def clear_draft_files() -> None:
    """Clear the content (empty file) of all draft files

    from 2010 to 2024 without deleting the file itself.
    """
    for year in paths.YEARS:
        target_file = paths.get_draft_csv_path(year)

        if target_file.exists():
            with open(target_file, "w", encoding="utf-8"):
                pass
            print(f"[CLEAR] Cleared file content: {target_file.name}")
        else:
            print(f"[SKIP] File does not exist: {target_file.name}")

    print("=> Completed clearing content of all draft files!")


def process_and_join_gdp_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read GDP data, filter by 194 sovereign countries,

    then join each year's GDP column (2010-2024)
    into the corresponding draft_{year}.csv file.
    """
    if input_file is not None:
        gdp_raw_file = input_file
    else:
        gdp_raw_file = (
            paths.RAW_GDP_IMPUTED_CSV if use_imputed else paths.RAW_GDP_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing GDP [{file_status}]: {gdp_raw_file.name}")

    if not gdp_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'gdp_per_capita_ppp' ({file_status}) into {draft_file.name}")


def process_and_join_gdp_growth_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read GDP Growth data, filter by 194 countries,

    and join each year's growth column into draft_{year}.csv
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
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing GDP Growth [{file_status}]: {gdp_growth_raw_file.name}")

    if not gdp_growth_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'gdp_growth_annual_pct' ({file_status}) into {draft_file.name}")


def process_and_join_govt_expenditure_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read General government final consumption expenditure (% of GDP) data."""
    if input_file is not None:
        govt_raw_file = input_file
    else:
        govt_raw_file = (
            paths.RAW_GOVT_EXPENDITURE_IMPUTED_CSV
            if use_imputed
            else paths.RAW_GOVT_EXPENDITURE_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Government Expenditure [{file_status}]: {govt_raw_file.name}")

    if not govt_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column '{col_name}' ({file_status}) into {draft_file.name}")


def process_and_join_inflation_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Inflation data."""
    if input_file is not None:
        inflation_raw_file = input_file
    else:
        inflation_raw_file = (
            paths.RAW_INFLATION_IMPUTED_CSV
            if use_imputed
            else paths.RAW_INFLATION_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Inflation [{file_status}]: {inflation_raw_file.name}")

    if not inflation_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'inflation_gdp_deflator_annual_pct' ({file_status}) into {draft_file.name}")


def process_and_join_unemployment_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Unemployment Rate data."""
    if input_file is not None:
        unemployment_raw_file = input_file
    else:
        unemployment_raw_file = (
            paths.RAW_UNEMPLOYMENT_IMPUTED_CSV
            if use_imputed
            else paths.RAW_UNEMPLOYMENT_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Unemployment [{file_status}]: {unemployment_raw_file.name}")

    if not unemployment_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'unemployment_rate_pct' ({file_status}) into {draft_file.name}")


def process_and_join_population_growth_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Population Growth Rate data."""
    if input_file is not None:
        pop_growth_raw_file = input_file
    else:
        pop_growth_raw_file = (
            paths.RAW_POPULATION_GROWTH_IMPUTED_CSV
            if use_imputed
            else paths.RAW_POPULATION_GROWTH_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Population Growth [{file_status}]: {pop_growth_raw_file.name}")

    if not pop_growth_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'population_growth_annual_pct' ({file_status}) into {draft_file.name}")


def process_and_join_population_total_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Total Population data."""
    if input_file is not None:
        pop_total_raw_file = input_file
    else:
        pop_total_raw_file = (
            paths.RAW_POPULATION_TOTAL_IMPUTED_CSV
            if use_imputed
            else paths.RAW_POPULATION_TOTAL_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Total Population [{file_status}]: {pop_total_raw_file.name}")

    if not pop_total_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'population_total' ({file_status}) into {draft_file.name}")


def process_and_join_urban_population_pct_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Urban Population (% of total population) data."""
    if input_file is not None:
        urb_pct_raw_file = input_file
    else:
        urb_pct_raw_file = (
            paths.RAW_URBAN_POPULATION_PCT_IMPUTED_CSV
            if use_imputed
            else paths.RAW_URBAN_POPULATION_PCT_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Urban Population (%) [{file_status}]: {urb_pct_raw_file.name}")

    if not urb_pct_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'urban_population_pct' ({file_status}) into {draft_file.name}")


def process_and_join_urban_population_growth_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Urban Population Growth Rate data."""
    if input_file is not None:
        urb_grow_raw_file = input_file
    else:
        urb_grow_raw_file = (
            paths.RAW_URBAN_POPULATION_GROWTH_IMPUTED_CSV
            if use_imputed
            else paths.RAW_URBAN_POPULATION_GROWTH_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Urban Population Growth [{file_status}]: {urb_grow_raw_file.name}")

    if not urb_grow_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'urban_population_growth_annual_pct' ({file_status}) into {draft_file.name}")


def process_and_join_life_expectancy_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Life Expectancy at Birth data."""
    if input_file is not None:
        life_exp_raw_file = input_file
    else:
        life_exp_raw_file = (
            paths.RAW_LIFE_EXPECTANCY_IMPUTED_CSV
            if use_imputed
            else paths.RAW_LIFE_EXPECTANCY_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Life Expectancy [{file_status}]: {life_exp_raw_file.name}")

    if not life_exp_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'life_expectancy_years' ({file_status}) into {draft_file.name}")


def process_and_join_fertility_rate_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Fertility Rate data."""
    if input_file is not None:
        fertility_raw_file = input_file
    else:
        fertility_raw_file = (
            paths.RAW_FERTILITY_RATE_IMPUTED_CSV
            if use_imputed
            else paths.RAW_FERTILITY_RATE_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Fertility Rate [{file_status}]: {fertility_raw_file.name}")

    if not fertility_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'fertility_rate_births_per_woman' ({file_status}) into {draft_file.name}")


def process_and_join_under_5_mortality_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Under-5 Mortality Rate data."""
    if input_file is not None:
        mortality_raw_file = input_file
    else:
        mortality_raw_file = (
            paths.RAW_UNDER_5_MORTALITY_IMPUTED_CSV
            if use_imputed
            else paths.RAW_UNDER_5_MORTALITY_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Under-5 Mortality [{file_status}]: {mortality_raw_file.name}")

    if not mortality_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'under_5_mortality_rate_per_1000' ({file_status}) into {draft_file.name}")


def process_and_join_access_to_electricity_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Access to Electricity data."""
    if input_file is not None:
        elec_raw_file = input_file
    else:
        elec_raw_file = (
            paths.RAW_ACCESS_TO_ELECTRICITY_IMPUTED_CSV
            if use_imputed
            else paths.RAW_ACCESS_TO_ELECTRICITY_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Access to Electricity [{file_status}]: {elec_raw_file.name}")

    if not elec_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'access_to_electricity_pct' ({file_status}) into {draft_file.name}")

def process_and_join_owid_co2_emissions_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read CO2 emissions per capita data from OWID (Long format)."""
    if input_file is not None:
        co2_raw_file = input_file
    else:
        co2_raw_file = (
            paths.RAW_CO2_EMISSIONS_OWID_IMPUTED_CSV
            if use_imputed
            else paths.RAW_CO2_EMISSIONS_OWID_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing OWID CO2 Emissions [{file_status}]: {co2_raw_file.name}")

    if not co2_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'co2_emissions_per_capita' ({file_status}) into {draft_file.name}")


def process_and_join_adult_schooling_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Mean years of schooling data from OWID (Long format)."""
    if input_file is not None:
        schooling_raw_file = input_file
    else:
        schooling_raw_file = (
            paths.RAW_ADULT_SCHOOLING_OWID_IMPUTED_CSV
            if use_imputed
            else paths.RAW_ADULT_SCHOOLING_OWID_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing OWID Mean Years of Schooling [{file_status}]: {schooling_raw_file.name}")

    if not schooling_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'mean_years_of_schooling_adults' ({file_status}) into {draft_file.name}")

def process_and_join_internet_users_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Internet Users (% of population) data."""
    if input_file is not None:
        net_raw_file = input_file
    else:
        net_raw_file = (
            paths.RAW_INTERNET_USERS_IMPUTED_CSV
            if use_imputed
            else paths.RAW_INTERNET_USERS_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Internet Users [{file_status}]: {net_raw_file.name}")

    if not net_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'internet_users_pct_population' ({file_status}) into {draft_file.name}")


def process_and_join_trade_pct_gdp_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Trade (% of GDP) data."""
    if input_file is not None:
        trade_raw_file = input_file
    else:
        trade_raw_file = (
            paths.RAW_TRADE_IMPUTED_CSV
            if use_imputed
            else paths.RAW_TRADE_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Trade (% of GDP) [{file_status}]: {trade_raw_file.name}")

    if not trade_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'trade_pct_gdp' ({file_status}) into {draft_file.name}")


def process_and_join_agriculture_pct_gdp_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Agriculture, forestry, and fishing, value added (% of GDP) data."""
    if input_file is not None:
        agr_raw_file = input_file
    else:
        agr_raw_file = (
            paths.RAW_AGRICULTURE_IMPUTED_CSV
            if use_imputed
            else paths.RAW_AGRICULTURE_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Agriculture (% of GDP) [{file_status}]: {agr_raw_file.name}")

    if not agr_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'agriculture_pct_gdp' ({file_status}) into {draft_file.name}")


def process_and_join_industry_pct_gdp_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Industry (including construction), value added (% of GDP) data."""
    if input_file is not None:
        ind_raw_file = input_file
    else:
        ind_raw_file = (
            paths.RAW_INDUSTRY_IMPUTED_CSV
            if use_imputed
            else paths.RAW_INDUSTRY_CSV
        )

    country_file = paths.COUNTRY_LIST_CSV
    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Industry (% of GDP) [{file_status}]: {ind_raw_file.name}")

    if not ind_raw_file.exists() or not country_file.exists():
        print(f"[ERROR] File does not exist!")
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
        print(f"[SUCCESS] Joined column 'industry_pct_gdp' ({file_status}) into {draft_file.name}")


def process_and_join_services_pct_gdp_data(
    use_imputed: bool = True,
    input_file: Path | None = None,
) -> None:
    """Read Services, value added (% of GDP) data (Imputed or Unimputed),

    filter by 194 countries, and join each year's column (2010-2024)
    into draft_{year}.csv with column name 'services_pct_gdp'.

    Parameters
    ----------
    use_imputed : bool, default=True
        If True, reads the imputed file (paths.RAW_SERVICES_IMPUTED_CSV).
        If False, reads the raw unimputed file (paths.RAW_SERVICES_CSV).
    input_file : Path, optional
        Optional file path override (if specifying a custom file path).
    """
    # 1. Determine file path based on use_imputed flag
    if input_file is not None:
        srv_raw_file = input_file
    else:
        srv_raw_file = (
            paths.RAW_SERVICES_IMPUTED_CSV
            if use_imputed
            else paths.RAW_SERVICES_CSV 
        )

    country_file = paths.COUNTRY_LIST_CSV

    file_status = "Imputed" if use_imputed else "Unimputed (Raw)"
    print(f"🔄 Processing Services (% of GDP) file [{file_status}]: {srv_raw_file.name}")

    if not srv_raw_file.exists():
        print(f"[ERROR] Services file not found: {srv_raw_file}")
        return
    if not country_file.exists():
        print(f"[ERROR] 194 Countries file not found: {country_file}")
        return

    # 2. Read data
    df_countries = pd.read_csv(country_file)
    df_srv_raw = pd.read_csv(srv_raw_file, skiprows=4)

    df_srv_filtered = df_srv_raw[
        df_srv_raw["Country Code"].isin(df_countries["country_code_3"])
    ].copy()

    # 3. Join into each draft_{year}.csv file
    for year in paths.YEARS:
        year_str = str(year)
        draft_file = paths.get_draft_csv_path(year)

        if not draft_file.exists():
            print(f"[SKIP] Draft file does not exist yet: {draft_file.name}")
            continue

        if year_str not in df_srv_filtered.columns:
            print(f"[WARNING] Year {year_str} is not in the Services file!")
            continue

        df_draft = pd.read_csv(draft_file)

        # If 'services_pct_gdp' column already exists, drop it to overwrite with new data
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
            f"[SUCCESS] Joined column 'services_pct_gdp' ({file_status}) into file {draft_file.name}"
        )

    print(f"=> Finished processing and joining Services (% of GDP) data [{file_status}]!")

def build_and_populate_all_draft_data(
    use_imputed: bool = True,
    clear_existing: bool = True,
) -> None:
    """Master function: Initialize draft files and load/join all 20 socio-economic indicators.

    Parameters
    ----------
    use_imputed : bool, default=True
        If True, loads imputed data files for all indicators.
        If False, loads raw unimputed data files.
    clear_existing : bool, default=True
        If True, clears content of existing draft files before reloading.
    """
    file_status = "IMPUTED" if use_imputed else "UNIMPUTED (RAW)"
    
    print("=======================================================")
    print(f"🚀 STARTING PROCESSING AND JOINING ALL DRAFT INDICATORS [{file_status}]")
    print("=======================================================\n")

    # 1. Clear old draft content if clear_existing flag is enabled
    if clear_existing:
        print("--- Clearing draft content ---")
        clear_draft_files()

    # 2. Initialize draft file frames
    print("--- Loading baseline data into draft ---")
    populate_draft_files()

    print(f"\n--- Starting joining 20 indicators [{file_status}] ---")

    # 3. List of all indicator joining functions
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

    # 4. Automatically loop and call each function with use_imputed flag
    for func in join_functions:
        try:
            func(use_imputed=use_imputed)
        except TypeError:
            func()

    print("\n=======================================================")
    print(f"🎉 COMPLETED ALL DRAFT DATA LOADING AND JOINING STEPS [{file_status}]!")
    print("=======================================================")

if __name__ == "__main__":
    # Execute joining all UNIMPUTED (Raw) data or IMPUTED data
    build_and_populate_all_draft_data(use_imputed=True)