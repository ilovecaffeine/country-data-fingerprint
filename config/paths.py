# config/paths.py
from pathlib import Path

# Root directory & main directories
CONFIG_DIR = Path(__file__).resolve().parent
ROOT_DIR = CONFIG_DIR.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"

# Experiment directories
PROCESSED_DATA_EXPERIMENT1_DIR = PROCESSED_DATA_DIR / "experiment_1"
PROCESSED_DATA_EXPERIMENT2_DIR = PROCESSED_DATA_DIR / "experiment_2"
PROCESSED_DATA_EXPERIMENT3_DIR = PROCESSED_DATA_DIR / "experiment_3"

RESULTS_EXPERIMENT1_DIR = RESULTS_DIR / "experiment_1"
RESULTS_EXPERIMENT2_DIR = RESULTS_DIR / "experiment_2"
RESULTS_EXPERIMENT3_DIR = RESULTS_DIR / "experiment_3"
RESULTS_EXPERIMENT4_DIR = RESULTS_DIR / "experiment_4"

# Automatically create directories if they do not exist
for folder in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    PROCESSED_DATA_EXPERIMENT1_DIR,
    PROCESSED_DATA_EXPERIMENT2_DIR,
    PROCESSED_DATA_EXPERIMENT3_DIR,
    RESULTS_EXPERIMENT1_DIR,
    RESULTS_EXPERIMENT2_DIR,
    RESULTS_EXPERIMENT3_DIR,
    RESULTS_EXPERIMENT4_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# Timeline & Country list file
YEARS = range(2010, 2025)
COUNTRY_LIST_CSV = RAW_DATA_DIR / "countries_194.csv"

# Dictionary & Helper functions
DRAFT_CSVS = {year: RAW_DATA_DIR / f"draft_{year}.csv" for year in YEARS}


def get_draft_csv_path(year: int) -> Path:
    return RAW_DATA_DIR / f"draft_{year}.csv"


def get_panel_experiment_1_csv_path(year: int) -> Path:
    return PROCESSED_DATA_EXPERIMENT1_DIR / f"panel_exp1_{year}.csv"


# ==============================================================================
# RAW & IMPUTED DATA PATHS
# ==============================================================================
def _get_raw_pair(folder: str, filename: str) -> tuple[Path, Path]:
    """Helper function to create a path pair (RAW, IMPUTED) to avoid redundant code."""
    base_path = RAW_DATA_DIR / folder / filename
    imputed_path = RAW_DATA_DIR / folder / f"{Path(filename).stem}_imputed.csv"
    return base_path, imputed_path


# World Bank Datasets
RAW_GDP_CSV, RAW_GDP_IMPUTED_CSV = _get_raw_pair(
    "API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_33608",
    "API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_33608.csv",
)
RAW_GDP_GROWTH_CSV, RAW_GDP_GROWTH_IMPUTED_CSV = _get_raw_pair(
    "API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_33455",
    "API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_33455.csv",
)
RAW_GOVT_EXPENDITURE_CSV, RAW_GOVT_EXPENDITURE_IMPUTED_CSV = _get_raw_pair(
    "API_NE.CON.GOVT.ZS_DS2_en_csv_v2_38789",
    "API_NE.CON.GOVT.ZS_DS2_en_csv_v2_38789.csv",
)
RAW_INFLATION_CSV, RAW_INFLATION_IMPUTED_CSV = _get_raw_pair(
    "API_NY.GDP.DEFL.KD.ZG_DS2_en_csv_v2_34350",
    "API_NY.GDP.DEFL.KD.ZG_DS2_en_csv_v2_34350.csv",
)
RAW_UNEMPLOYMENT_CSV, RAW_UNEMPLOYMENT_IMPUTED_CSV = _get_raw_pair(
    "API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_33398",
    "API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_33398.csv",
)
RAW_POPULATION_TOTAL_CSV, RAW_POPULATION_TOTAL_IMPUTED_CSV = _get_raw_pair(
    "API_SP.POP.TOTL_DS2_en_csv_v2_33112",
    "API_SP.POP.TOTL_DS2_en_csv_v2_33112.csv",
)
RAW_POPULATION_GROWTH_CSV, RAW_POPULATION_GROWTH_IMPUTED_CSV = _get_raw_pair(
    "API_SP.POP.GROW_DS2_en_csv_v2_35913",
    "API_SP.POP.GROW_DS2_en_csv_v2_35913.csv",
)
RAW_URBAN_POPULATION_PCT_CSV, RAW_URBAN_POPULATION_PCT_IMPUTED_CSV = (
    _get_raw_pair(
        "API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_33901",
        "API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_33901.csv",
    )
)
RAW_URBAN_POPULATION_GROWTH_CSV, RAW_URBAN_POPULATION_GROWTH_IMPUTED_CSV = (
    _get_raw_pair(
        "API_SP.URB.GROW_DS2_en_csv_v2_38007",
        "API_SP.URB.GROW_DS2_en_csv_v2_38007.csv",
    )
)
RAW_LIFE_EXPECTANCY_CSV, RAW_LIFE_EXPECTANCY_IMPUTED_CSV = _get_raw_pair(
    "API_SP.DYN.LE00.IN_DS2_en_csv_v2_408",
    "API_SP.DYN.LE00.IN_DS2_en_csv_v2_408.csv",
)
RAW_FERTILITY_RATE_CSV, RAW_FERTILITY_RATE_IMPUTED_CSV = _get_raw_pair(
    "API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_33381",
    "API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_33381.csv",
)
RAW_UNDER_5_MORTALITY_CSV, RAW_UNDER_5_MORTALITY_IMPUTED_CSV = _get_raw_pair(
    "API_SH.DYN.MORT_DS2_en_csv_v2_34194",
    "API_SH.DYN.MORT_DS2_en_csv_v2_34194.csv",
)
RAW_ACCESS_TO_ELECTRICITY_CSV, RAW_ACCESS_TO_ELECTRICITY_IMPUTED_CSV = (
    _get_raw_pair(
        "API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_33377",
        "API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_33377.csv",
    )
)
RAW_INTERNET_USERS_CSV, RAW_INTERNET_USERS_IMPUTED_CSV = _get_raw_pair(
    "API_IT.NET.USER.ZS_DS2_en_csv_v2_33086",
    "API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv",
)
RAW_TRADE_CSV, RAW_TRADE_IMPUTED_CSV = _get_raw_pair(
    "API_NE.TRD.GNFS.ZS_DS2_en_csv_v2_171",
    "API_NE.TRD.GNFS.ZS_DS2_en_csv_v2_171.csv",
)
RAW_AGRICULTURE_CSV, RAW_AGRICULTURE_IMPUTED_CSV = _get_raw_pair(
    "API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_33230",
    "API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_33230.csv",
)
RAW_INDUSTRY_CSV, RAW_INDUSTRY_IMPUTED_CSV = _get_raw_pair(
    "API_NV.IND.TOTL.ZS_DS2_en_csv_v2_102950",
    "API_NV.IND.TOTL.ZS_DS2_en_csv_v2_102950.csv",
)
RAW_SERVICES_CSV, RAW_SERVICES_IMPUTED_CSV = _get_raw_pair(
    "API_NV.SRV.TOTL.ZS_DS2_en_csv_v2_35142",
    "API_NV.SRV.TOTL.ZS_DS2_en_csv_v2_35142.csv",
)

# OWID Datasets (Long Format)
RAW_CO2_EMISSIONS_OWID_CSV, RAW_CO2_EMISSIONS_OWID_IMPUTED_CSV = _get_raw_pair(
    "co-emissions-per-capita", "co-emissions-per-capita.csv"
)
RAW_ADULT_SCHOOLING_OWID_CSV, RAW_ADULT_SCHOOLING_OWID_IMPUTED_CSV = (
    _get_raw_pair(
        "years-of-schooling", "average-years-of-schooling-among-adults.csv"
    )
)