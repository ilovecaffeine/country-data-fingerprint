# Data Dictionary & Feature Codebook

This document provides a comprehensive specification of the **20 macroeconomic and socio-economic indicators** that constitute the **Country Data Fingerprint** dataset ($2010–2024$).

---

## 1. Identifiers & Metadata Columns

Before the 20 socio-economic features, each observation in the panel data is indexed by three primary metadata columns:

| Column Name | Data Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `country_code_3` | String (ISO 3166-1 alpha-3) | `VNM`, `USA`, `DEU` | 3-letter ISO country code (Primary key). |
| `country_name` | String | `Vietnam`, `United States` | Official short name of the country. |
| `year` | Integer | `2010`, `2018`, `2024` | Observation year ($2010 \le t \le 2024$). |

---

## 2. Master Feature Dictionary

Below is the master dictionary mapping each feature name to its data provider, official series code, dataset description, and raw file source path in `data/raw/`:

| Feature Name in Dataset | Provider | Series Code | Official Series Name | Raw File Source (`data/raw/`) |
| :--- | :--- | :--- | :--- | :--- |
| `gdp_per_capita_ppp` | World Bank WDI | `NY.GDP.PCAP.PP.KD` | GDP per capita, PPP (constant 2021 international $) | `API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_33608/...` |
| `population_total` | World Bank WDI | `SP.POP.TOTL` | Population, total | `API_SP.POP.TOTL_DS2_en_csv_v2_33112/...` |
| `urban_population_pct` | World Bank WDI | `SP.URB.TOTL.IN.ZS` | Urban population (% of total population) | `API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_33901/...` |
| `co2_emissions_per_capita` | Our World in Data | `co-emissions-per-capita` | CO2 emissions per capita (metric tons per person) | `co-emissions-per-capita/...` |
| `mean_years_of_schooling_adults` | Our World in Data | `years-of-schooling` | Average years of schooling among adults (years) | `years-of-schooling/...` |
| `agriculture_pct_gdp` | World Bank WDI | `NV.AGR.TOTL.ZS` | Agriculture, forestry, and fishing, value added (% of GDP) | `API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_33230/...` |
| `under_5_mortality_rate_per_1000` | World Bank WDI | `SH.DYN.MORT` | Mortality rate, under-5 (per 1,000 live births) | `API_SH.DYN.MORT_DS2_en_csv_v2_34194/...` |
| `fertility_rate_births_per_woman` | World Bank WDI | `SP.DYN.TFRT.IN` | Fertility rate, total (births per woman) | `API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_33381/...` |
| `trade_pct_gdp` | World Bank WDI | `NE.TRD.GNFS.ZS` | Trade (% of GDP) | `API_NE.TRD.GNFS.ZS_DS2_en_csv_v2_171/...` |
| `industry_pct_gdp` | World Bank WDI | `NV.IND.TOTL.ZS` | Industry (including construction), value added (% of GDP) | `API_NV.IND.TOTL.ZS_DS2_en_csv_v2_102950/...` |
| `unemployment_rate_pct` | World Bank WDI | `SL.UEM.TOTL.ZS` | Unemployment, total (% of total labor force) (modeled ILO estimate) | `API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_33398/...` |
| `life_expectancy_years` | World Bank WDI | `SP.DYN.LE00.IN` | Life expectancy at birth, total (years) | `API_SP.DYN.LE00.IN_DS2_en_csv_v2_408/...` |
| `general_government_final_consumption_expenditure_pct_gdp` | World Bank WDI | `NE.CON.GOVT.ZS` | General government final consumption expenditure (% of GDP) | `API_NE.CON.GOVT.ZS_DS2_en_csv_v2_38789/...` |
| `services_pct_gdp` | World Bank WDI | `NV.SRV.TOTL.ZS` | Services, value added (% of GDP) | `API_NV.SRV.TOTL.ZS_DS2_en_csv_v2_35142/...` |
| `population_growth_annual_pct` | World Bank WDI | `SP.POP.GROW` | Population growth (annual %) | `API_SP.POP.GROW_DS2_en_csv_v2_35913/...` |
| `access_to_electricity_pct` | World Bank WDI | `EG.ELC.ACCS.ZS` | Access to electricity (% of population) | `API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_33377/...` |
| `urban_population_growth_annual_pct` | World Bank WDI | `SP.URB.GROW` | Urban population growth (annual %) | `API_SP.URB.GROW_DS2_en_csv_v2_38007/...` |
| `internet_users_pct_population` | World Bank WDI | `IT.NET.USER.ZS` | Individuals using the Internet (% of population) | `API_IT.NET.USER.ZS_DS2_en_csv_v2_33086/...` |
| `inflation_gdp_deflator_annual_pct` | World Bank WDI | `NY.GDP.DEFL.KD.ZG` | Inflation, GDP deflator (annual %) | `API_NY.GDP.DEFL.KD.ZG_DS2_en_csv_v2_34350/...` |
| `gdp_growth_annual_pct` | World Bank WDI | `NY.GDP.PCAP.KD.ZG` | GDP per capita growth (annual %) | `API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_33455/...` |

---

## 3. Categorized Feature Codebook by Macroeconomic Domain

To facilitate domain-level analysis, the 20 features are classified into **5 socio-economic domains**:

### 3.1 Macroeconomic Performance & Openness
* **`gdp_per_capita_ppp`**: GDP per capita based on Purchasing Power Parity (PPP) converted to constant 2021 international dollars.
  * *Unit*: Constant 2021 International $ | *Source File*: `API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_33608/API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_33608.csv`
* **`gdp_growth_annual_pct`**: Annual percentage growth rate of GDP per capita based on local currency.
  * *Unit*: Annual % | *Source File*: `API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_33455/API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_33455.csv`
* **`inflation_gdp_deflator_annual_pct`**: Annual inflation rate as measured by the rate of change in the GDP deflator.
  * *Unit*: Annual % | *Source File*: `API_NY.GDP.DEFL.KD.ZG_DS2_en_csv_v2_34350/API_NY.GDP.DEFL.KD.ZG_DS2_en_csv_v2_34350.csv`
* **`trade_pct_gdp`**: The sum of exports and imports of goods and services measured as a share of Gross Domestic Product.
  * *Unit*: % of GDP | *Source File*: `API_NE.TRD.GNFS.ZS_DS2_en_csv_v2_171/API_NE.TRD.GNFS.ZS_DS2_en_csv_v2_171.csv`
* **`general_government_final_consumption_expenditure_pct_gdp`**: Government final consumption expenditure including all government current expenditures for purchases of goods and services.
  * *Unit*: % of GDP | *Source File*: `API_NE.CON.GOVT.ZS_DS2_en_csv_v2_38789/API_NE.CON.GOVT.ZS_DS2_en_csv_v2_38789.csv`

---

### 3.2 Economic Sectoral Structure
* **`agriculture_pct_gdp`**: Net output of agriculture, forestry, and fishing sectors as a percentage of GDP.
  * *Unit*: % of GDP | *Source File*: `API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_33230/API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_33230.csv`
* **`industry_pct_gdp`**: Net output of mining, manufacturing, construction, electricity, water, and gas sectors as a percentage of GDP.
  * *Unit*: % of GDP | *Source File*: `API_NV.IND.TOTL.ZS_DS2_en_csv_v2_102950/API_NV.IND.TOTL.ZS_DS2_en_csv_v2_102950.csv`
* **`services_pct_gdp`**: Net output of wholesale/retail trade, transport, government, financial, professional, and personal services as a percentage of GDP.
  * *Unit*: % of GDP | *Source File*: `API_NV.SRV.TOTL.ZS_DS2_en_csv_v2_35142/API_NV.SRV.TOTL.ZS_DS2_en_csv_v2_35142.csv`

---

### 3.3 Demographics & Labor
* **`population_total`**: Midyear total population regardless of legal status or citizenship.
  * *Unit*: People | *Source File*: `API_SP.POP.TOTL_DS2_en_csv_v2_33112/API_SP.POP.TOTL_DS2_en_csv_v2_33112.csv`
* **`population_growth_annual_pct`**: Exponential annual population growth rate.
  * *Unit*: Annual % | *Source File*: `API_SP.POP.GROW_DS2_en_csv_v2_35913/API_SP.POP.GROW_DS2_en_csv_v2_35913.csv`
* **`urban_population_pct`**: Share of the total population living in urban areas as defined by national statistical offices.
  * *Unit*: % of Total Population | *Source File*: `API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_33901/API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_33901.csv`
* **`urban_population_growth_annual_pct`**: Annual growth rate of the urban population.
  * *Unit*: Annual % | *Source File*: `API_SP.URB.GROW_DS2_en_csv_v2_38007/API_SP.URB.GROW_DS2_en_csv_v2_38007.csv`
* **`unemployment_rate_pct`**: Share of the labor force that is without work but available for and seeking employment (modeled ILO estimate).
  * *Unit*: % of Labor Force | *Source File*: `API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_33398/API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_33398.csv`

---

### 3.4 Health, Education & Human Development
* **`mean_years_of_schooling_adults`**: Average number of completed years of education received by people aged 25 and older.
  * *Unit*: Years | *Source File*: `years-of-schooling/average-years-of-schooling-among-adults.csv`
* **`life_expectancy_years`**: Number of years a newborn infant would live if prevailing patterns of mortality at birth were to stay the same.
  * *Unit*: Years | *Source File*: `API_SP.DYN.LE00.IN_DS2_en_csv_v2_408/API_SP.DYN.LE00.IN_DS2_en_csv_v2_408.csv`
* **`under_5_mortality_rate_per_1000`**: Probability per 1,000 live births that a newborn child will die before reaching age five.
  * *Unit*: Per 1,000 Live Births | *Source File*: `API_SH.DYN.MORT_DS2_en_csv_v2_34194/API_SH.DYN.MORT_DS2_en_csv_v2_34194.csv`
* **`fertility_rate_births_per_woman`**: Number of children that would be born to a woman if she were to live to the end of her childbearing years.
  * *Unit*: Births per Woman | *Source File*: `API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_33381/API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_33381.csv`

---

### 3.5 Infrastructure & Environmental Sustainability
* **`co2_emissions_per_capita`**: Production-based carbon dioxide emissions from fossil fuels and industry divided by population.
  * *Unit*: Metric Tons per Person | *Source File*: `co-emissions-per-capita/co-emissions-per-capita.csv`
* **`access_to_electricity_pct`**: Percentage of population with access to electricity.
  * *Unit*: % of Population | *Source File*: `API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_33377/API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_33377.csv`
* **`internet_users_pct_population`**: Individuals who have used the Internet (from any location) in the last 3 months.
  * *Unit*: % of Population | *Source File*: `API_IT.NET.USER.ZS_DS2_en_csv_v2_33086/API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv`

---

## 4. Raw Data Directory Storage Mapping

To ensure full reproducibility, raw indicator datasets are extracted and maintained under the `data/raw/` directory structure:

```text
data/raw/
├── API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_33377/
│   └── API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_33377.csv
├── API_NE.CON.GOVT.ZS_DS2_en_csv_v2_38789/
│   └── API_NE.CON.GOVT.ZS_DS2_en_csv_v2_38789.csv
├── API_NE.TRD.GNFS.ZS_DS2_en_csv_v2_171/
│   └── API_NE.TRD.GNFS.ZS_DS2_en_csv_v2_171.csv
├── API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_33230/
│   └── API_NV.AGR.TOTL.ZS_DS2_en_csv_v2_33230.csv
├── API_NV.IND.TOTL.ZS_DS2_en_csv_v2_102950/
│   └── API_NV.IND.TOTL.ZS_DS2_en_csv_v2_102950.csv
├── API_NV.SRV.TOTL.ZS_DS2_en_csv_v2_35142/
│   └── API_NV.SRV.TOTL.ZS_DS2_en_csv_v2_35142.csv
├── API_NY.GDP.DEFL.KD.ZG_DS2_en_csv_v2_34350/
│   └── API_NY.GDP.DEFL.KD.ZG_DS2_en_csv_v2_34350.csv
├── API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_33455/
│   └── API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_33455.csv
├── API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_33608/
│   └── API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_33608.csv
├── API_SH.DYN.MORT_DS2_en_csv_v2_34194/
│   └── API_SH.DYN.MORT_DS2_en_csv_v2_34194.csv
├── API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_33398/
│   └── API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_33398.csv
├── API_SP.DYN.LE00.IN_DS2_en_csv_v2_408/
│   └── API_SP.DYN.LE00.IN_DS2_en_csv_v2_408.csv
├── API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_33381/
│   └── API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_33381.csv
├── API_SP.POP.GROW_DS2_en_csv_v2_35913/
│   └── API_SP.POP.GROW_DS2_en_csv_v2_35913.csv
├── API_SP.POP.TOTL_DS2_en_csv_v2_33112/
│   └── API_SP.POP.TOTL_DS2_en_csv_v2_33112.csv
├── API_SP.URB.GROW_DS2_en_csv_v2_38007/
│   └── API_SP.URB.GROW_DS2_en_csv_v2_38007.csv
├── API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_33901/
│   └── API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_33901.csv
├── API_IT.NET.USER.ZS_DS2_en_csv_v2_33086/
│   └── API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv
├── co-emissions-per-capita/
│   └── co-emissions-per-capita.csv
└── years-of-schooling/
    └── average-years-of-schooling-among-adults.csv
```