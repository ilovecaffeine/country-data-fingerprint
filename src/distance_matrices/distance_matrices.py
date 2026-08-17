# src/distance_matrices/distance_matrices.py
# cd country-data-fingerprint
# python -m src.distance_matrices.distance_matrices
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from config import paths


def compute_euclidean_matrix_scipy(
    df: pd.DataFrame, id_cols: list[str] = ["country_code_3"]
) -> pd.DataFrame:
    """Computes the Euclidean distance matrix between all rows in a DataFrame."""
    labels = df[id_cols[0]].values
    feature_cols = [
        c
        for c in df.columns
        if c not in id_cols + ["country_name", "year"]
    ]
    features = df[feature_cols].values

    # Compute NxN symmetric distance matrix
    dist_matrix = squareform(pdist(features, metric="euclidean"))

    # Return distance matrix as DataFrame with country codes as index/columns
    return pd.DataFrame(dist_matrix, index=labels, columns=labels)


def find_closest_countries(
    dist_df: pd.DataFrame, country_code: str = "VNM", top_k: int = 5
) -> pd.Series:
    """Finds the top_k closest countries in Euclidean distance to a target country."""
    if country_code not in dist_df.index:
        raise ValueError(
            f"Country code '{country_code}' not found in the distance matrix!"
        )

    # Get row, sort ascending, skip self (position 0)
    return dist_df.loc[country_code].sort_values().iloc[1 : top_k + 1]


def export_experiment_1_distance_matrices(
    exp1_input_dir: Path | None = None,
    output_dir: Path | None = None,
    target_year: int | None = None,
) -> None:
    """Reads data from PROCESSED_DATA_EXPERIMENT1_DIR, computes Euclidean distance

    matrices, and exports them to CSV files.

    Parameters
    ----------
    exp1_input_dir : Path, optional
        Directory containing Experiment 1 CSV files. Defaults to config paths.
    output_dir : Path, optional
        Output directory for exported distance matrices.
    target_year : int, optional
        Specific year (e.g., 2024) to process. If None, processes ALL years.
    """
    if exp1_input_dir is None:
        exp1_input_dir = paths.PROCESSED_DATA_EXPERIMENT1_DIR

    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT1_DIR / "distance_matrices"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather CSV files to process
    if target_year is not None:
        csv_files = [exp1_input_dir / f"panel_exp1_{target_year}.csv"]
    else:
        csv_files = sorted(exp1_input_dir.glob("panel_exp1_*.csv"))

    if not csv_files:
        print(
            f"[ERROR] No Experiment 1 data files found in: {exp1_input_dir}"
        )
        return

    print("=======================================================")
    print("STARTING EUCLIDEAN DISTANCE MATRIX CALCULATION & EXPORT")
    print("=======================================================\n")

    for file_path in csv_files:
        if not file_path.exists():
            print(f"[SKIP] File does not exist: {file_path.name}")
            continue

        # Extract year from filename (panel_exp1_2024.csv -> 2024)
        year_str = file_path.stem.split("_")[-1]

        # 2. Read dataset
        df_year = pd.read_csv(file_path)

        # 3. Compute distance matrix
        dist_matrix_df = compute_euclidean_matrix_scipy(
            df_year, id_cols=["country_code_3"]
        )

        # 4. Save matrix to CSV
        output_matrix_path = output_dir / f"euclidean_matrix_{year_str}.csv"
        dist_matrix_df.to_csv(output_matrix_path, encoding="utf-8")

        print(f"✅ Exported matrix for year {year_str} -> {output_matrix_path.name}")

        # 5. Display sample Top 5 closest to Vietnam (if VNM present)
        if "VNM" in dist_matrix_df.index:
            top_vnm = find_closest_countries(
                dist_matrix_df, country_code="VNM", top_k=5
            )
            print(f"   📍 Top 5 closest countries to Vietnam (VNM) in {year_str}:")
            for code, dist in top_vnm.items():
                print(f"      - {code}: {dist:.4f}")
            print("-" * 55)

    print("\n=======================================================")
    print(f"📁 ALL DISTANCE MATRICES SAVED AT: {output_dir}")
    print("=======================================================")


def compute_cosine_similarity_matrix_scipy(
    df: pd.DataFrame, id_cols: list[str] = ["country_code_3"]
) -> pd.DataFrame:
    """Computes Cosine Similarity matrix between all rows in a DataFrame.

    Returns
    -------
    pd.DataFrame
        Similarity matrix with values in range [-1, 1].
        Values closer to 1 indicate higher similarity in country fingerprint structures.
    """
    labels = df[id_cols[0]].values
    feature_cols = [
        c
        for c in df.columns
        if c not in id_cols + ["country_name", "year"]
    ]
    features = df[feature_cols].values

    # pdist with metric='cosine' calculates Cosine Distance = 1 - Cosine Similarity
    cosine_dist_matrix = squareform(pdist(features, metric="cosine"))

    # Convert to Cosine Similarity
    cosine_sim_matrix = 1.0 - cosine_dist_matrix

    return pd.DataFrame(cosine_sim_matrix, index=labels, columns=labels)


def find_most_similar_countries(
    sim_df: pd.DataFrame, country_code: str = "VNM", top_k: int = 5
) -> pd.Series:
    """Finds top_k countries with highest Cosine Similarity to a target country."""
    if country_code not in sim_df.index:
        raise ValueError(
            f"Country code '{country_code}' not found in the similarity matrix!"
        )

    # Get row, sort descending, skip self (first position = 1.0)
    return sim_df.loc[country_code].sort_values(ascending=False).iloc[1 : top_k + 1]


def export_experiment_1_cosine_matrices(
    exp1_input_dir: Path | None = None,
    output_dir: Path | None = None,
    target_year: int | None = None,
) -> None:
    """Reads data from PROCESSED_DATA_EXPERIMENT1_DIR, computes Cosine Similarity

    matrices, and exports to CSV files (`cosine_similarity_matrix_{year}.csv`).

    Parameters
    ----------
    exp1_input_dir : Path, optional
        Directory containing Experiment 1 CSV files. Defaults to config paths.
    output_dir : Path, optional
        Output directory for exported Cosine Similarity matrices.
    target_year : int, optional
        Specific year (e.g., 2024) to process. If None, processes ALL years.
    """
    if exp1_input_dir is None:
        exp1_input_dir = paths.PROCESSED_DATA_EXPERIMENT1_DIR

    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT1_DIR / "cosine_matrices"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gather CSV files to process
    if target_year is not None:
        csv_files = [exp1_input_dir / f"panel_exp1_{target_year}.csv"]
    else:
        csv_files = sorted(exp1_input_dir.glob("panel_exp1_*.csv"))

    if not csv_files:
        print(
            f"[ERROR] No Experiment 1 data files found in: {exp1_input_dir}"
        )
        return

    print("=======================================================")
    print("STARTING COSINE SIMILARITY MATRIX CALCULATION & EXPORT")
    print("=======================================================\n")

    for file_path in csv_files:
        if not file_path.exists():
            print(f"[SKIP] File does not exist: {file_path.name}")
            continue

        # Extract year from filename (panel_exp1_2024.csv -> 2024)
        year_str = file_path.stem.split("_")[-1]

        # 2. Read dataset
        df_year = pd.read_csv(file_path)

        # 3. Compute Cosine Similarity matrix
        cosine_sim_df = compute_cosine_similarity_matrix_scipy(
            df_year, id_cols=["country_code_3"]
        )

        # 4. Save matrix to CSV
        output_matrix_path = output_dir / f"cosine_similarity_matrix_{year_str}.csv"
        cosine_sim_df.to_csv(output_matrix_path, encoding="utf-8")

        print(
            f"✅ Exported Cosine Similarity matrix for year {year_str} -> {output_matrix_path.name}"
        )

        # 5. Display sample Top 5 most similar to Vietnam (if VNM present)
        if "VNM" in cosine_sim_df.index:
            top_vnm = find_most_similar_countries(
                cosine_sim_df, country_code="VNM", top_k=5
            )
            print(
                f"   📍 Top 5 most similar countries to Vietnam (VNM) in {year_str}:"
            )
            for code, sim in top_vnm.items():
                print(f"      - {code}: {sim:.4f}")
            print("-" * 55)

    print("\n=======================================================")
    print(f"📁 ALL SIMILARITY MATRICES SAVED AT: {output_dir}")
    print("=======================================================")


def generate_euclidean_neighbors_summary(
    matrices_dir: Path | None = None,
    output_file: Path | None = None,
    raw_panel_file: Path | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Summarizes top K nearest neighbors by Euclidean Distance across all years (2010-2024)

    into a single Wide Format CSV:
      - Columns: country_code_3 | country_name | rank | 2010 | 2011 | ... | 2024
      - Cell format: "distance (country_code_B)" (e.g., "2.8729 (KHM)")
    """
    if matrices_dir is None:
        matrices_dir = paths.RESULTS_EXPERIMENT1_DIR / "distance_matrices"

    if output_file is None:
        output_file = paths.RESULTS_EXPERIMENT1_DIR / "euclidean_neighbors.csv"

    if raw_panel_file is None:
        raw_panel_file = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"

    # 1. Map country_code_3 to country_name
    country_name_map = {}
    if raw_panel_file.exists():
        df_raw = pd.read_csv(raw_panel_file)
        country_name_map = (
            df_raw.drop_duplicates(subset=["country_code_3"])
            .set_index("country_code_3")["country_name"]
            .to_dict()
        )

    # 2. Gather exported Euclidean matrix files
    matrix_files = sorted(matrices_dir.glob("euclidean_matrix_*.csv"))

    if not matrix_files:
        print(f"[ERROR] No matrix files found in: {matrices_dir}")
        return pd.DataFrame()

    print("=======================================================")
    print("GENERATING EUCLIDEAN NEIGHBORS SUMMARY (WIDE FORMAT)")
    print("=======================================================\n")

    # Storage structure: records_dict[country][rank][year] = "dist (code)"
    records_dict = {}
    all_years = []

    for file_path in matrix_files:
        year_str = file_path.stem.split("_")[-1]
        all_years.append(year_str)

        # Read distance matrix
        df_matrix = pd.read_csv(file_path, index_col=0)

        for country_a in df_matrix.index:
            if country_a not in records_dict:
                records_dict[country_a] = {
                    r: {} for r in range(1, top_k + 1)
                }

            # Find top K nearest neighbors (excluding country_a itself)
            row = df_matrix.loc[country_a].sort_values(ascending=True)
            top_neighbors = row.iloc[1 : top_k + 1]

            for rank_idx, (country_b, dist_val) in enumerate(
                top_neighbors.items(), start=1
            ):
                formatted_value = f"{dist_val:.4f} ({country_b})"
                records_dict[country_a][rank_idx][year_str] = formatted_value

    all_years = sorted(list(set(all_years)), key=lambda x: int(x))

    # 3. Convert records dictionary into Wide Format DataFrame
    rows = []
    for country_code in sorted(records_dict.keys()):
        c_name = country_name_map.get(country_code, country_code)

        for r in range(1, top_k + 1):
            row_dict = {
                "country_code_3": country_code,
                "country_name": c_name,
                "rank": r,
            }
            # Populate year values
            for y in all_years:
                row_dict[y] = records_dict[country_code][r].get(y, None)

            rows.append(row_dict)

    summary_df = pd.DataFrame(rows)

    # 4. Export summary to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"✅ SUCCESSFULLY EXPORTED SUMMARY: {output_file}")
    print(f"📊 Total rows: {len(summary_df)} ({len(records_dict)} countries x {top_k} ranks)")
    print("=======================================================\n")

    return summary_df


def generate_cosine_neighbors_summary(
    matrices_dir: Path | None = None,
    output_file: Path | None = None,
    raw_panel_file: Path | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """Summarizes top K most similar neighbors by Cosine Similarity across all years (2010-2024)

    into a single Wide Format CSV:
      - Columns: country_code_3 | country_name | rank | 2010 | 2011 | ... | 2024
      - Cell format: "similarity (country_code_B)" (e.g., "0.8072 (KHM)")
    """
    if matrices_dir is None:
        matrices_dir = paths.RESULTS_EXPERIMENT1_DIR / "cosine_matrices"

    if output_file is None:
        output_file = paths.RESULTS_EXPERIMENT1_DIR / "cosine_neighbors.csv"

    if raw_panel_file is None:
        raw_panel_file = paths.RAW_DATA_DIR / "draft_panel_2010_2024.csv"

    # 1. Map country_code_3 to country_name
    country_name_map = {}
    if raw_panel_file.exists():
        df_raw = pd.read_csv(raw_panel_file)
        country_name_map = (
            df_raw.drop_duplicates(subset=["country_code_3"])
            .set_index("country_code_3")["country_name"]
            .to_dict()
        )

    # 2. Gather exported Cosine Similarity matrix files
    matrix_files = sorted(matrices_dir.glob("*cosine*matrix_*.csv"))

    if not matrix_files:
        # Fallback search for any CSV files in directory
        matrix_files = sorted(matrices_dir.glob("*.csv"))

    if not matrix_files:
        print(f"[ERROR] No matrix files found in: {matrices_dir}")
        return pd.DataFrame()

    print("=======================================================")
    print("GENERATING COSINE NEIGHBORS SUMMARY (WIDE FORMAT)")
    print("=======================================================\n")

    # Storage structure: records_dict[country][rank][year] = "sim (code)"
    records_dict = {}
    all_years = []

    for file_path in matrix_files:
        year_str = file_path.stem.split("_")[-1]
        all_years.append(year_str)

        # Read similarity matrix
        df_matrix = pd.read_csv(file_path, index_col=0)

        for country_a in df_matrix.index:
            if country_a not in records_dict:
                records_dict[country_a] = {
                    r: {} for r in range(1, top_k + 1)
                }

            # Find top K highest Cosine Similarity neighbors (sort DESCENDING, skip self at index 0)
            row = df_matrix.loc[country_a].sort_values(ascending=False)
            top_neighbors = row.iloc[1 : top_k + 1]

            for rank_idx, (country_b, sim_val) in enumerate(
                top_neighbors.items(), start=1
            ):
                formatted_value = f"{sim_val:.4f} ({country_b})"
                records_dict[country_a][rank_idx][year_str] = formatted_value

    all_years = sorted(list(set(all_years)), key=lambda x: int(x))

    # 3. Convert records dictionary into Wide Format DataFrame
    rows = []
    for country_code in sorted(records_dict.keys()):
        c_name = country_name_map.get(country_code, country_code)

        for r in range(1, top_k + 1):
            row_dict = {
                "country_code_3": country_code,
                "country_name": c_name,
                "rank": r,
            }
            # Populate year values
            for y in all_years:
                row_dict[y] = records_dict[country_code][r].get(y, None)

            rows.append(row_dict)

    summary_df = pd.DataFrame(rows)

    # 4. Export summary to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"✅ SUCCESSFULLY EXPORTED SUMMARY: {output_file}")
    print(f"📊 Total rows: {len(summary_df)} ({len(records_dict)} countries x {top_k} ranks)")
    print("=======================================================\n")

    return summary_df


if __name__ == "__main__":
    # Export Euclidean distance matrices for all years (2010 -> 2024)
    export_experiment_1_distance_matrices()

    # Export Cosine Similarity matrices for all years
    export_experiment_1_cosine_matrices()

    # Generate summary CSVs in Wide Format
    generate_euclidean_neighbors_summary()
    generate_cosine_neighbors_summary()