# src/data/transform.py
# cd country-data-fingerprint
# python -m src.data.transform
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
from config import paths
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Default list of countries/territories with excessive missing data
DEFAULT_COUNTRIES_TO_DROP = [
    # 10 countries with severe missing data / initial microstates
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

    # Countries missing 100% Govt Expend & Trade (or Unemployment) continuously for 15 years
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
    "AFG",  # Afghanistan (missing 10/15 consecutive years from 2010-2019)

    # Group of newly added countries (missing 100% of an indicator or fragmented 2017-2024)
    "KIR",  # Kiribati (missing 100% unemployment)
    "SYC",  # Seychelles (missing 100% unemployment)
    "MHL",  # Marshall Islands (missing 100% unemployment)
    "YEM",  # Yemen (missing 100% GDP PPP)
    "CUB",  # Cuba (missing 100% GDP PPP)
    "BDI",  # Burundi (missing 100% Trade)
    "LAO",  # Laos (missing Govt Expend & Trade from 2017-2024)
    "SDN",  # Sudan (missing Internet users)
    "WSM",  # Samoa (missing Internet users from 2015-2024)
    "VUT",  # Vanuatu (missing Internet users from 2016-2024)
    "TKM",  # Turkmenistan (missing Internet users from 2017-2024)
]


def combine_all_drafts_to_panel_data(
    drop_countries: bool = True,
    countries_to_drop: Optional[list[str]] = None,
    drop_countries_with_missing: bool = False,
) -> pd.DataFrame:
    """Reads draft_{year}.csv files, inserts the 'year' column at index 2,
    optionally filters out specified countries or any countries with missing data,
    and merges them into a single Panel Data file at data/raw/draft_panel_2010_2024.csv.

    Parameters
    ----------
    drop_countries : bool, optional
        If True, drops the specified list of countries. Default is True.
    countries_to_drop : list[str], optional
        List of country codes (ISO3) to remove. If None, uses DEFAULT_COUNTRIES_TO_DROP.
    drop_countries_with_missing : bool, optional
        If True, drops ALL countries that have at least 1 missing/NaN cell across 2010-2024
        (retains only 100% clean countries). Default is False.
    """
    if countries_to_drop is None:
        countries_to_drop = DEFAULT_COUNTRIES_TO_DROP

    all_dfs = []

    for year in paths.YEARS:
        draft_file = paths.get_draft_csv_path(year)

        if not draft_file.exists():
            print(f"[SKIP] File does not exist: {draft_file.name}")
            continue

        df = pd.read_csv(draft_file)

        # Avoid duplicate column error if 'year' already exists
        if "year" in df.columns:
            df = df.drop(columns=["year"])

        # Insert 'year' column at index 2
        df.insert(loc=2, column="year", value=int(year))

        all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] No draft files found to combine!")
        return pd.DataFrame()

    # Concatenate all tables vertically
    panel_df = pd.concat(all_dfs, ignore_index=True)

    # Filter out countries if drop_countries is True
    total_before = panel_df["country_code_3"].nunique()

    if drop_countries and countries_to_drop:
        panel_df = panel_df[~panel_df["country_code_3"].isin(countries_to_drop)].copy()
        total_after = panel_df["country_code_3"].nunique()
        print(f"🗑️ Filtered out {total_before - total_after} countries from the target list.")

    # Drop all countries with missing data if flag is set
    if drop_countries_with_missing:
        before_nan_drop = panel_df["country_code_3"].nunique()
        
        # Identify country codes with at least 1 NaN cell
        nan_countries = panel_df[panel_df.isna().any(axis=1)]["country_code_3"].unique()
        
        # Keep only countries not in nan_countries
        panel_df = panel_df[~panel_df["country_code_3"].isin(nan_countries)].copy()
        after_nan_drop = panel_df["country_code_3"].nunique()
        
        print(
            f"🧹 [WARNING FLAG] Completely removed {before_nan_drop - after_nan_drop} countries containing missing data "
            f"(Retained {after_nan_drop} fully clean countries)."
        )

    total_final = panel_df["country_code_3"].nunique()

    # Sort data by Country Code and Year sequentially
    panel_df = panel_df.sort_values(by=["country_code_3", "year"]).reset_index(drop=True)

    # Save to draft_panel_2010_2024.csv
    output_path = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(output_path, index=False, encoding="utf-8")

    print(
        f"✅ SUCCESSFULLY CREATED PANEL DATA TABLE! ({len(panel_df)} rows x {len(panel_df.columns)} cols | {total_final} countries)"
    )
    print(f"📁 Saved at: {output_path}")

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
    """Time-series imputation for World Bank Raw files (Wide Format).

    - Preserves the first 4 metadata header lines in the exported file.
    - Evaluates [eval_start, eval_end]. If missing count <= max_missing_allowed:
    - Applies linear interpolation + ffill + bfill on range [impute_start, impute_end].
    - Exports to _imputed.csv in the same directory.
    """
    if not input_file.exists():
        print(f"[ERROR] Source file does not exist: {input_file}")
        return

    # Read and store first 4 metadata lines
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        header_lines = [f.readline() for _ in range(4)]

    # Read World Bank dataset (skipping 4 metadata lines)
    df_raw = pd.read_csv(input_file, skiprows=4)

    # Identify existing year columns
    years_eval = [str(y) for y in range(eval_start, eval_end + 1) if str(y) in df_raw.columns]
    years_impute = [str(y) for y in range(impute_start, impute_end + 1) if str(y) in df_raw.columns]

    if not years_eval or not years_impute:
        print(f"[WARNING] Sufficient year columns not found in file: {input_file.name}")
        return

    # Count NaNs in 2010-2024 range for each row
    missing_count = df_raw[years_eval].isna().sum(axis=1)

    # Filter rows meeting condition: missing <= max_missing_allowed
    eligible_mask = missing_count <= max_missing_allowed

    # Perform imputation ONLY on eligible rows across 1995-2024 range
    if eligible_mask.any():
        block_to_impute = df_raw.loc[eligible_mask, years_impute].astype(float)

        block_imputed = (
            block_to_impute.interpolate(method="linear", axis=1)
            .ffill(axis=1)
            .bfill(axis=1)
        )

        df_raw.loc[eligible_mask, years_impute] = block_imputed

    # Save _imputed.csv: write metadata header first, then append data (mode='a')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.writelines(header_lines)

    df_raw.to_csv(output_file, mode="a", index=False, encoding="utf-8")

    total_rows = len(df_raw)
    imputed_rows = eligible_mask.sum()
    skipped_rows = total_rows - imputed_rows

    print(f"[SUCCESS] Processed: {input_file.name}")
    print(f"          -> Imputed successfully (1995-2024): {imputed_rows}/{total_rows} countries")
    print(f"          -> Skipped (missing > {max_missing_allowed}): {skipped_rows} countries")
    print(f"          -> Output file: {output_file.name}\n")


def impute_all_worldbank_raw_files() -> None:
    """Runs imputation on ALL Raw World Bank files defined in config paths."""
    print("=======================================================")
    print("STARTING IMPUTATION FOR RAW WORLD BANK FILES (1995 - 2024)")
    print("=======================================================\n")

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

    print("=> ALL RAW WORLD BANK FILES IMPUTED SUCCESSFULLY!")


def impute_ourworldindata_raw_file(
    input_file: Path,
    output_file: Path,
    eval_start: int = 2010,
    eval_end: int = 2024,
    impute_start: int = 1995,
    impute_end: int = 2024,
    max_missing_allowed: int = 7,
) -> None:
    """Time-series imputation for Our World in Data Raw files (Long Format).

    - Temporarily converts to Wide Format to count missing values in 2010-2024.
    - If missing <= 7: Performs linear interpolation + ffill + bfill (1995-2024).
    - Reshapes back to Long Format and outputs _imputed.csv.
    """
    if not input_file.exists():
        print(f"[ERROR] OWID source file does not exist: {input_file}")
        return

    df_raw = pd.read_csv(input_file)

    id_cols = [c for c in ["Entity", "Code"] if c in df_raw.columns]
    if not id_cols:
        print(f"[ERROR] Entity/Code columns not found in: {input_file.name}")
        return

    val_cols = [c for c in df_raw.columns if c not in id_cols + ["Year"]]
    if not val_cols:
        print(f"[ERROR] Indicator value column not found in: {input_file.name}")
        return

    val_col = val_cols[0]

    # Pivot Long to Wide Format (Rows: Entity/Code, Columns: Year)
    df_pivot = df_raw.pivot(index=id_cols, columns="Year", values=val_col)

    all_impute_years = list(range(impute_start, impute_end + 1))
    df_pivot = df_pivot.reindex(
        columns=sorted(set(df_pivot.columns).union(all_impute_years))
    )

    years_eval = [y for y in range(eval_start, eval_end + 1) if y in df_pivot.columns]
    years_impute = [y for y in range(impute_start, impute_end + 1) if y in df_pivot.columns]

    missing_count = df_pivot[years_eval].isna().sum(axis=1)
    eligible_mask = missing_count <= max_missing_allowed

    if eligible_mask.any():
        block_to_impute = df_pivot.loc[eligible_mask, years_impute].astype(float)

        block_imputed = (
            block_to_impute.interpolate(method="linear", axis=1)
            .ffill(axis=1)
            .bfill(axis=1)
        )

        df_pivot.loc[eligible_mask, years_impute] = block_imputed

    # Melt back from Wide Format to standard OWID Long Format
    df_imputed_long = df_pivot.reset_index().melt(
        id_vars=id_cols, var_name="Year", value_name=val_col
    )

    df_imputed_long["Year"] = df_imputed_long["Year"].astype(int)
    df_imputed_long = df_imputed_long.dropna(subset=[val_col]).reset_index(drop=True)

    orig_cols = [c for c in df_raw.columns if c in df_imputed_long.columns]
    df_imputed_long = df_imputed_long[orig_cols].sort_values(by=id_cols + ["Year"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_imputed_long.to_csv(output_file, index=False, encoding="utf-8")

    total_entities = len(df_pivot)
    imputed_entities = eligible_mask.sum()
    skipped_entities = total_entities - imputed_entities

    print(f"[SUCCESS OWID] Processed: {input_file.name}")
    print(f"               -> Imputed successfully (1995-2024): {imputed_entities}/{total_entities} entities")
    print(f"               -> Skipped (missing > {max_missing_allowed}): {skipped_entities} entities")
    print(f"               -> Output file: {output_file.name}\n")


def impute_all_ourworldindata_raw_files() -> None:
    """Runs imputation on ALL Raw Our World in Data files defined in config paths."""
    print("=======================================================")
    print("STARTING IMPUTATION FOR RAW OUR WORLD IN DATA FILES (1995 - 2024)")
    print("=======================================================\n")

    owid_file_pairs = [
        (paths.RAW_CO2_EMISSIONS_OWID_CSV, paths.RAW_CO2_EMISSIONS_OWID_IMPUTED_CSV),
        (paths.RAW_ADULT_SCHOOLING_OWID_CSV, paths.RAW_ADULT_SCHOOLING_OWID_IMPUTED_CSV),
    ]

    for raw_path, imputed_path in owid_file_pairs:
        impute_ourworldindata_raw_file(input_file=raw_path, output_file=imputed_path)

    print("=> ALL RAW OUR WORLD IN DATA FILES IMPUTED SUCCESSFULLY!")


def process_experiment_1_for_year(
    df_panel: pd.DataFrame, target_year: int
) -> pd.DataFrame:
    # 1. STEP 1: FILTER DATA FOR TARGET YEAR ONLY
    df_year = df_panel[df_panel["year"] == target_year].copy()

    # 2. STEP 2: SKEWNESS TRANSFORMATIONS (Applied independently per year)
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

    # c) Yeo-Johnson transform for Inflation
    pt = PowerTransformer(method="yeo-johnson")
    df_year["inflation_gdp_deflator_annual_pct"] = pt.fit_transform(
        df_year[["inflation_gdp_deflator_annual_pct"]]
    )

    # d) Reflect + Log transform for Electricity access
    max_elec = df_year["access_to_electricity_pct"].max()
    df_year["access_to_electricity_pct"] = np.log(
        (max_elec + 1) - df_year["access_to_electricity_pct"]
    )

    # 3. STEP 3: SCALING (FIT & TRANSFORM ONLY ON TARGET YEAR)
    id_cols = ["country_code_3", "country_name", "year"]
    feature_cols = [c for c in df_year.columns if c not in id_cols]

    scaler = StandardScaler()
    df_year[feature_cols] = scaler.fit_transform(df_year[feature_cols])

    return df_year


def preprocess_and_export_all_experiment_1_years(
    input_file: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Reads draft panel, processes Experiment 1 independently for each year (2010-2024),
    and exports panel_exp1_{year}.csv files to PROCESSED_DATA_DIR / "experiment_1".
    """
    if input_file is None:
        input_file = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"

    if output_dir is None:
        output_dir = paths.PROCESSED_DATA_EXPERIMENT1_DIR

    if not input_file.exists():
        print(f"[ERROR] Source data file not found: {input_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    df_panel = pd.read_csv(input_file)
    available_years = sorted(df_panel["year"].unique())

    print("=======================================================")
    print("PROCESSING AND EXPORTING EXPERIMENT 1 DATA")
    print("=======================================================\n")

    for year in available_years:
        df_exp1_year = process_experiment_1_for_year(
            df_panel=df_panel, target_year=int(year)
        )

        if df_exp1_year.empty:
            print(f"[SKIP] No data available for year {year}")
            continue

        output_path = output_dir / f"panel_exp1_{year}.csv"
        df_exp1_year.to_csv(output_path, index=False, encoding="utf-8")

        print(
            f"✅ [SUCCESS] Year {year}: {len(df_exp1_year)} countries -> {output_path.name}"
        )

    print("\n=======================================================")
    print(f"📁 ALL EXPERIMENT 1 FILES SAVED AT: {output_dir}")
    print("=======================================================")


def preprocess_and_split_experiment_2(
    input_file: Path | None = None,
    output_dir: Path | None = None,
    train_end_year: int = 2018,
    validation_end_year: int = 2020,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocesses and splits data for Experiment 2 - Classification.

    Split:
        Train      : 2010–2018
        Validation : 2019–2020
        Test       : 2021–2024

    All transformers and scalers are FIT ONLY ON TRAIN to prevent data leakage.
    """
    if input_file is None:
        input_file = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"

    if output_dir is None:
        output_dir = paths.PROCESSED_DATA_EXPERIMENT2_DIR

    if not input_file.exists():
        print(f"[ERROR] Data file not found: {input_file}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. READ DATA
    df = pd.read_csv(input_file)

    # 2. TEMPORAL TRAIN / VALIDATION / TEST SPLIT
    train_df = df[df["year"] <= train_end_year].copy()
    validation_df = df[
        (df["year"] > train_end_year) & (df["year"] <= validation_end_year)
    ].copy()
    test_df = df[df["year"] > validation_end_year].copy()

    print("\n=======================================================")
    print("EXPERIMENT 2 - CLASSIFICATION")
    print("=======================================================")
    print(f"Train      : {train_df['year'].min()}–{train_df['year'].max()}")
    print(f"Validation : {validation_df['year'].min()}–{validation_df['year'].max()}")
    print(f"Test       : {test_df['year'].min()}–{test_df['year'].max()}")

    # 3. TRANSFORMATION
    log_cols = ["gdp_per_capita_ppp", "population_total"]
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

    # Log transform
    for col in log_cols:
        train_df[col] = np.log(train_df[col])
        validation_df[col] = np.log(validation_df[col])
        test_df[col] = np.log(test_df[col])

    # Log1p transform
    for col in log1p_cols:
        train_df[col] = np.log1p(train_df[col])
        validation_df[col] = np.log1p(validation_df[col])
        test_df[col] = np.log1p(test_df[col])

    # Yeo-Johnson transform
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

    # Reflect + Log transform
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

    # 4. SCALING
    id_cols = ["country_code_3", "country_name", "year"]
    feature_cols = [c for c in train_df.columns if c not in id_cols]

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    validation_df[feature_cols] = scaler.transform(validation_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # 5. EXPORT
    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8")
    validation_df.to_csv(output_dir / "validation.csv", index=False, encoding="utf-8")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8")

    print("\n[SUCCESS] Experiment 2 completed.")
    print(f"  Train      : {len(train_df)} rows")
    print(f"  Validation : {len(validation_df)} rows")
    print(f"  Test       : {len(test_df)} rows")
    print(f"  Output     : {output_dir}")

    return train_df, validation_df, test_df


def preprocess_and_split_experiment_3(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    train_end_year: int = 2017,
    val_end_year: int = 2019,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocesses and splits data for Experiment 3 - Forecasting.

    Pipeline:
    1. Reads 3 files (train.csv, validation.csv, test.csv) from PROCESSED_DATA_EXPERIMENT2_DIR.
    2. Concatenates them into full_df to prevent boundary gaps when shifting time steps.
    3. Creates 20 target Y columns for year t+1 using shift(-1) grouped by country_code_3.
    4. Drops rows for 2024 (lacking year 2025 as target Y).
    5. Re-splits into Train / Validation / Test by time threshold:
        - Train     : year <= 2017 (Year t: 2010–2017 -> Forecast t+1: 2011–2018)
        - Validation: 2017 < year <= 2019 (Year t: 2018–2019 -> Forecast t+1: 2019–2020)
        - Test      : year > 2019 (Year t: 2020–2023 -> Forecast t+1: 2021–2024)
    6. Exports train.csv, validation.csv, test.csv to PROCESSED_DATA_EXPERIMENT3_DIR.
    """
    if input_dir is None:
        input_dir = paths.PROCESSED_DATA_EXPERIMENT2_DIR

    if output_dir is None:
        output_dir = paths.PROCESSED_DATA_EXPERIMENT3_DIR

    # 1. CHECK INPUT FILES FROM EXP 2
    train_path = input_dir / "train.csv"
    val_path = input_dir / "validation.csv"
    test_path = input_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            print(f"[ERROR] Exp 2 data file not found: {p}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("EXPERIMENT 3 - FORECASTING PREPROCESSING")
    print("=======================================================")

    # 2. READ AND MERGE DATA FROM EXP 2
    exp2_train = pd.read_csv(train_path)
    exp2_val = pd.read_csv(val_path)
    exp2_test = pd.read_csv(test_path)

    full_df = pd.concat([exp2_train, exp2_val, exp2_test], ignore_index=True)
    full_df = full_df.sort_values(["country_code_3", "year"]).reset_index(drop=True)

    # 3. DEFINE FEATURES (X) AND TARGETS (Y for year t+1)
    id_cols = ["country_code_3", "country_name", "year"]
    feature_cols = [c for c in full_df.columns if c not in id_cols]
    target_cols = [f"{c}_target_next_year" for c in feature_cols]

    # Shift (-1) per country code to get target for year t+1
    for col, t_col in zip(feature_cols, target_cols):
        full_df[t_col] = full_df.groupby("country_code_3")[col].shift(-1)

    # Drop 2024 rows (NaNs in target columns due to missing 2025 data)
    full_df_clean = full_df.dropna(subset=target_cols).reset_index(drop=True)

    # 4. TIME-BASED RE-SPLIT (BY YEAR t)
    train_df = full_df_clean[full_df_clean["year"] <= train_end_year].copy()
    validation_df = full_df_clean[
        (full_df_clean["year"] > train_end_year)
        & (full_df_clean["year"] <= val_end_year)
    ].copy()
    test_df = full_df_clean[full_df_clean["year"] > val_end_year].copy()

    # 5. EXPORT FILES TO PROCESSED_DATA_EXPERIMENT3_DIR
    train_df.to_csv(output_dir / "train.csv", index=False, encoding="utf-8")
    validation_df.to_csv(output_dir / "validation.csv", index=False, encoding="utf-8")
    test_df.to_csv(output_dir / "test.csv", index=False, encoding="utf-8")

    print("✅ [SUCCESS] Experiment 3 data preprocessing complete!")
    print(
        f"  Train      : {len(train_df)} rows | Year t: {train_df['year'].min()}–{train_df['year'].max()} "
        f"(Forecast t+1: {train_df['year'].min()+1}–{train_df['year'].max()+1})"
    )
    print(
        f"  Validation : {len(validation_df)} rows | Year t: {validation_df['year'].min()}–{validation_df['year'].max()} "
        f"(Forecast t+1: {validation_df['year'].min()+1}–{validation_df['year'].max()+1})"
    )
    print(
        f"  Test       : {len(test_df)} rows | Year t: {test_df['year'].min()}–{test_df['year'].max()} "
        f"(Forecast t+1: {test_df['year'].min()+1}–{test_df['year'].max()+1})"
    )
    print(f"  Total cols : {full_df_clean.shape[1]} (3 ID cols + {len(feature_cols)} X cols + {len(target_cols)} Y cols)")
    print(f"  Output dir : {output_dir}")

    return train_df, validation_df, test_df


if __name__ == "__main__":
    impute_all_worldbank_raw_files()
    impute_all_ourworldindata_raw_files()

    panel_df = combine_all_drafts_to_panel_data(
        drop_countries=False,
        drop_countries_with_missing=True
    )

    preprocess_and_export_all_experiment_1_years()
    preprocess_and_split_experiment_2()
    preprocess_and_split_experiment_3()