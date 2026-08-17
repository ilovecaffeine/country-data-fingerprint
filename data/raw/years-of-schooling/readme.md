# Average years of schooling among adults - Data package

This data package contains the data that powers the chart ["Average years of schooling among adults"](https://ourworldindata.org/grapher/years-of-schooling?v=1&csvType=full&useColumnShortNames=false&level=all&metric_type=average_years_schooling&sex=both) on the Our World in Data website. It was downloaded on August 11, 2026.

### Active Filters

A filtered subset of the full data was downloaded. The following filters were applied:

## CSV Structure

The high level structure of the CSV file is that each row is an observation for an entity (usually a country or region) and a timepoint (usually a year).

The first two columns in the CSV file are "Entity" and "Code". "Entity" is the name of the entity (e.g. "United States"). "Code" is the OWID internal entity code that we use if the entity is a country or region. For most countries, this is the same as the [iso alpha-3](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3) code of the entity (e.g. "USA") - for non-standard countries like historical countries these are custom codes.

The third column is either "Year" or "Day". If the data is annual, this is "Year" and contains only the year as an integer. If the column is "Day", the column contains a date string in the form "YYYY-MM-DD".

The final column is the data column, which is the time series that powers the chart. If the CSV data is downloaded using the "full data" option, then the column corresponds to the time series below. If the CSV data is downloaded using the "only selected data visible in the chart" option then the data column is transformed depending on the chart type and thus the association with the time series might not be as straightforward.


## Metadata.json structure

The .metadata.json file contains metadata about the data package. The "charts" key contains information to recreate the chart, like the title, subtitle etc.. The "columns" key contains information about each of the columns in the csv, like the unit, timespan covered, citation for the data etc..

## About the data

Our World in Data is almost never the original producer of the data - almost all of the data we use has been compiled by others. If you want to re-use data, it is your responsibility to ensure that you adhere to the sources' license and to credit them correctly. Please note that a single time series may have more than one source - e.g. when we stich together data from different time periods by different producers or when we calculate per capita metrics using population data from a second source.

## Detailed information about the data


## Average years of schooling – UNDP
Average number of years (excluding years spent repeating individual grades) adults over 25 years participated in formal education.
Last updated: May 7, 2025  
Next update: September 2026  
Date range: 1990–2023  
Unit: years  


### How to cite this data

#### In-line citation
If you have limited space (e.g. in data visualizations), you can use this abbreviated in-line citation:  
UNDP, Human Development Report (2025) – with minor processing by Our World in Data

#### Full citation
UNDP, Human Development Report (2025) – with minor processing by Our World in Data. “Average years of schooling – UNDP” [dataset]. UNDP, Human Development Report, “Human Development Report” [original data].
Source: UNDP, Human Development Report (2025) – with minor processing by Our World In Data

### What you should know about this data
- This indicator shows the average number of years that adults in a country have spent in formal education.
- It reflects the overall educational attainment of the population based on what they have already completed.
- The calculation converts each person's highest completed education level into years - for example, someone who finished high school counts as having roughly 12 years of schooling, while someone who never attended school counts as 0 years.
- The data comes from censuses and surveys of adults aged 25 and older, including only formal education starting from primary school.
- This indicator captures how much schooling adults have accumulated over their lifetimes, showing the results of past investments in education systems.
- Higher values indicate a population with stronger educational foundations, but the measure does not account for education quality or informal learning.
- The data may not reflect recent progress in countries with infrequent surveys or outdated census information.
- UNDP originally obtained this indicator from: Barro and Lee (2018), Eurostat (2024), ICF Macro Demographic and Health Surveys (various years), UNESCO Institute for Statistics (2024) and UNICEF Multiple Indicator Cluster Surveys (various years).

### Source

#### UNDP, Human Development Report – Human Development Report
Retrieved on: 2025-05-07  
Retrieved from: https://hdr.undp.org/  


    