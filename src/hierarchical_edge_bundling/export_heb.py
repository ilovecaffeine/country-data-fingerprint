# src/hierarchical_edge_bundling/export_heb.py
# cd country-data-fingerprint
# python -m src.hierarchical_edge_bundling.export_heb

import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import paths


# Continent Lookup Table for 194 countries (ISO3 codes - Excluding Vatican City/VAT)
continent_map = {
    # ================= Asia - 47 countries =================
    'AFG': 'Asia', 'ARE': 'Asia', 'ARM': 'Asia', 'AZE': 'Asia', 'BGD': 'Asia', 
    'BHR': 'Asia', 'BRN': 'Asia', 'KHM': 'Asia', 'CHN': 'Asia', 'CYP': 'Asia', 
    'GEO': 'Asia', 'IDN': 'Asia', 'IND': 'Asia', 'IRN': 'Asia', 'IRQ': 'Asia', 
    'ISR': 'Asia', 'JOR': 'Asia', 'JPN': 'Asia', 'KAZ': 'Asia', 'KGZ': 'Asia', 
    'KOR': 'Asia', 'KWT': 'Asia', 'LAO': 'Asia', 'LBN': 'Asia', 'LKA': 'Asia', 
    'MDV': 'Asia', 'MMR': 'Asia', 'MNG': 'Asia', 'MYS': 'Asia', 'NPL': 'Asia', 
    'OMN': 'Asia', 'PAK': 'Asia', 'PHL': 'Asia', 'PRK': 'Asia', 'PSE': 'Asia', 
    'QAT': 'Asia', 'SAU': 'Asia', 'SGP': 'Asia', 'SYR': 'Asia', 'THA': 'Asia', 
    'TJK': 'Asia', 'TKM': 'Asia', 'TLS': 'Asia', 'TUR': 'Asia', 'UZB': 'Asia', 
    'VNM': 'Asia', 'YEM': 'Asia', 'BTN': 'Asia',

    # ================= Europe - 43 countries =================
    'ALB': 'Europe', 'AND': 'Europe', 'AUT': 'Europe', 'BEL': 'Europe', 'BGR': 'Europe', 
    'BIH': 'Europe', 'BLR': 'Europe', 'CHE': 'Europe', 'CZE': 'Europe', 'DEU': 'Europe', 
    'DNK': 'Europe', 'ESP': 'Europe', 'EST': 'Europe', 'FIN': 'Europe', 'FRA': 'Europe', 
    'GBR': 'Europe', 'GRC': 'Europe', 'HRV': 'Europe', 'HUN': 'Europe', 'IRL': 'Europe', 
    'ISL': 'Europe', 'ITA': 'Europe', 'LIE': 'Europe', 'LTU': 'Europe', 'LUX': 'Europe', 
    'LVA': 'Europe', 'MCO': 'Europe', 'MDA': 'Europe', 'MKD': 'Europe', 'MLT': 'Europe', 
    'MNE': 'Europe', 'NLD': 'Europe', 'NOR': 'Europe', 'POL': 'Europe', 'PRT': 'Europe', 
    'ROU': 'Europe', 'RUS': 'Europe', 'SMR': 'Europe', 'SRB': 'Europe', 'SVK': 'Europe', 
    'SVN': 'Europe', 'SWE': 'Europe', 'UKR': 'Europe', 

    # ================= Africa - 54 countries =================
    'AGO': 'Africa', 'BDI': 'Africa', 'BEN': 'Africa', 'BFA': 'Africa', 'BWA': 'Africa', 
    'CAF': 'Africa', 'CIV': 'Africa', 'CMR': 'Africa', 'COD': 'Africa', 'COG': 'Africa', 
    'COM': 'Africa', 'CPV': 'Africa', 'DJI': 'Africa', 'DZA': 'Africa', 'EGY': 'Africa', 
    'ERI': 'Africa', 'ESW': 'Africa', 'ETH': 'Africa', 'GAB': 'Africa', 'GHA': 'Africa', 
    'GIN': 'Africa', 'GMB': 'Africa', 'GNB': 'Africa', 'GNQ': 'Africa', 'LBR': 'Africa', 
    'LBY': 'Africa', 'LSO': 'Africa', 'MAR': 'Africa', 'MDG': 'Africa', 'MLI': 'Africa', 
    'MOZ': 'Africa', 'MRT': 'Africa', 'MUS': 'Africa', 'MWI': 'Africa', 'NAM': 'Africa', 
    'NER': 'Africa', 'NGA': 'Africa', 'RWA': 'Africa', 'SDN': 'Africa', 'SEN': 'Africa', 
    'SLE': 'Africa', 'SOM': 'Africa', 'SSD': 'Africa', 'STP': 'Africa', 'SYC': 'Africa', 
    'TCD': 'Africa', 'TGO': 'Africa', 'TUN': 'Africa', 'TZA': 'Africa', 'UGA': 'Africa', 
    'ZAF': 'Africa', 'ZMB': 'Africa', 'ZWE': 'Africa', 'KEN': 'Africa', 'SWZ': 'Africa',

    # ================= North America - 23 countries =================
    'ATG': 'North America', 'BHS': 'North America', 'BLZ': 'North America', 
    'BRB': 'North America', 'CAN': 'North America', 'CRI': 'North America', 
    'CUB': 'North America', 'DMA': 'North America', 'DOM': 'North America', 
    'GRD': 'North America', 'GTM': 'North America', 'HND': 'North America', 
    'HTI': 'North America', 'JAM': 'North America', 'KNA': 'North America', 
    'LCA': 'North America', 'MEX': 'North America', 'NIC': 'North America', 
    'PAN': 'North America', 'SLV': 'North America', 'TTO': 'North America', 
    'USA': 'North America', 'VCT': 'North America',

    # ================= South America - 12 countries =================
    'ARG': 'South America', 'BOL': 'South America', 'BRA': 'South America', 
    'CHL': 'South America', 'COL': 'South America', 'ECU': 'South America', 
    'GUY': 'South America', 'PER': 'South America', 'PRY': 'South America', 
    'SUR': 'South America', 'URY': 'South America', 'VEN': 'South America',

    # ================= Oceania - 14 countries =================
    'AUS': 'Oceania', 'FJI': 'Oceania', 'FSM': 'Oceania', 'KIR': 'Oceania', 
    'MHL': 'Oceania', 'NRU': 'Oceania', 'NZL': 'Oceania', 'PLW': 'Oceania', 
    'PNG': 'Oceania', 'SLB': 'Oceania', 'TON': 'Oceania', 'TUV': 'Oceania', 
    'VUT': 'Oceania', 'WSM': 'Oceania'
}


def get_close_distance_pairs_by_year(
    year: int | str,
    top_k: int = 3,
    matrices_dir: Path | None = None,
) -> pd.DataFrame:
    """Reads the Euclidean distance matrix for a given year and extracts the Top-K nearest neighbors (excluding self)."""
    if matrices_dir is None:
        matrices_dir = paths.RESULTS_EXPERIMENT1_DIR / "distance_matrices"

    file_path = matrices_dir / f"euclidean_matrix_{year}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"[ERROR] Matrix file not found: {file_path}")

    # 1. Read distance matrix
    df_mat = pd.read_csv(file_path, index_col=0)

    # 2. Extract Top-K nearest neighbors for each country (excluding self)
    pairs = []
    for country, row in df_mat.iterrows():
        # ENSURE: Remove the country itself from comparison
        row_no_self = row.drop(labels=[country], errors="ignore")
        
        # Extract top_k countries with smallest distance
        top_k_neighbors = row_no_self.nsmallest(top_k).index.tolist()
        
        for neighbor in top_k_neighbors:
            pairs.append({"Country1": country, "Country2": neighbor})

    df_pairs = pd.DataFrame(pairs)

    # 3. Sort undirected pairs, remove duplicates and self-loops (if any)
    sorted_pairs = np.sort(df_pairs[["Country1", "Country2"]].values, axis=1)
    df_filtered = (
        pd.DataFrame(sorted_pairs, columns=["Country1", "Country2"])
        .drop_duplicates()
    )
    
    # Additional filter to remove identical country pairs (Country1 == Country2)
    df_filtered = df_filtered[df_filtered["Country1"] != df_filtered["Country2"]].reset_index(drop=True)

    return df_filtered


def get_close_cosine_pairs_by_year(
    year: int | str,
    top_k: int = 3,
    matrices_dir: Path | None = None,
) -> pd.DataFrame:
    """Reads the Cosine matrix for a given year and extracts the Top-K nearest neighbors (excluding self)."""
    if matrices_dir is None:
        matrices_dir = paths.RESULTS_EXPERIMENT1_DIR / "cosine_matrices"

    file_path = matrices_dir / f"cosine_similarity_matrix_{year}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"[ERROR] Cosine matrix file not found: {file_path}")

    # 1. Read Cosine matrix
    df_mat = pd.read_csv(file_path, index_col=0)

    # 2. Extract Top-K nearest neighbors for each country (excluding self)
    pairs = []
    for country, row in df_mat.iterrows():
        row_no_self = row.drop(labels=[country], errors="ignore")
        
        # Uses nlargest assuming matrix contains Cosine Similarity (higher = closer)
        top_k_neighbors = row_no_self.nlargest(top_k).index.tolist()
        
        for neighbor in top_k_neighbors:
            pairs.append({"Country1": country, "Country2": neighbor})

    df_pairs = pd.DataFrame(pairs)

    # 3. Sort undirected pairs, remove duplicates, and remove self-loops
    sorted_pairs = np.sort(df_pairs[["Country1", "Country2"]].values, axis=1)
    df_filtered = pd.DataFrame(sorted_pairs, columns=["Country1", "Country2"]).drop_duplicates()
    df_filtered = df_filtered[df_filtered["Country1"] != df_filtered["Country2"]].reset_index(drop=True)

    return df_filtered


def generate_true_heb_html(
    df_pairs: pd.DataFrame,
    continent_map: dict[str, str],
    html_file_path: Path,
    title: str = "Hierarchical Edge Bundling"
):
    """
    Generates a standard D3.js Hierarchical Edge Bundling diagram:
    - Assigns bi-directional connections to guarantee AT LEAST k edges per country.
    - Deduplicates SVG rendered paths in D3 for smooth rendering.
    - Highlights exact k connecting links and k neighbors on hover.
    """
    col_a = df_pairs.columns[0]
    col_b = df_pairs.columns[1]

    # 1. Collect list of all unique countries present in df_pairs
    active_countries = set(df_pairs[col_a].astype(str).str.strip()).union(
        set(df_pairs[col_b].astype(str).str.strip())
    )

    # 2. Collect BI-DIRECTIONAL connections
    connections = {}
    for _, row in df_pairs.iterrows():
        c1 = str(row[col_a]).strip()
        c2 = str(row[col_b]).strip()
        if c1 != c2:
            cont1 = continent_map.get(c1, 'Other')
            cont2 = continent_map.get(c2, 'Other')
            p1 = f"World.{cont1}.{c1}"
            p2 = f"World.{cont2}.{c2}"

            # Assign bi-directional connections for both c1 and c2
            if p1 not in connections:
                connections[p1] = set()
            connections[p1].add(p2)

            if p2 not in connections:
                connections[p2] = set()
            connections[p2].add(p1)

    # 3. Build Hierarchy Tree
    continents = {}
    for country_code in sorted(active_countries):
        continent = continent_map.get(country_code, 'Other')
        if continent not in continents:
            continents[continent] = []
        
        full_path = f"World.{continent}.{country_code}"
        imports_list = list(connections.get(full_path, []))
        
        continents[continent].append({
            "name": country_code,
            "imports": imports_list
        })

    tree_data = {
        "name": "World",
        "children": [
            {
                "name": cont_name,
                "children": country_list
            }
            for cont_name, country_list in continents.items()
        ]
    }

    # 4. D3.js v7 Standard HTML/JS Template
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #ffffff;
            margin: 0;
            padding: 20px;
        }}
        h2 {{
            margin-bottom: 5px;
            color: #2c3e50;
        }}
        .node {{
            font-size: 10px;
            font-weight: 500;
            cursor: pointer;
            transition: opacity 0.2s, font-size 0.2s;
        }}
        .link {{
            stroke: #7f8c8d;
            stroke-opacity: 0.25;
            fill: none;
            pointer-events: none;
            stroke-width: 1px;
            transition: stroke-opacity 0.2s, stroke-width 0.2s;
        }}
    </style>
</head>
<body>
    <h2>{title}</h2>
    <div id="chart"></div>

    <script>
        const data = {json.dumps(tree_data)};
        const width = 1000;
        const radius = width / 2 - 110;

        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

        const tree = d3.cluster().size([2 * Math.PI, radius]);

        const root = tree(bilink(d3.hierarchy(data)
            .sort((a, b) => d3.ascending(a.height, b.height) || d3.ascending(a.data.name, b.data.name))));

        const svg = d3.select("#chart").append("svg")
            .attr("width", width)
            .attr("height", width)
            .append("g")
            .attr("transform", `translate(${{width/2}},${{width/2}})`);

        const line = d3.lineRadial()
            .curve(d3.curveBundle.beta(0.85))
            .radius(d => d.y)
            .angle(d => d.x);

        // DEDUPLICATE DRAWING PATHS (Keep only 1 unique connection per pair)
        const allOutgoing = root.leaves().flatMap(leaf => leaf.outgoing);
        const uniqueLinksMap = new Map();
        allOutgoing.forEach(l => {{
            const id1 = id(l[0]);
            const id2 = id(l[1]);
            const key = [id1, id2].sort().join("---");
            if (!uniqueLinksMap.has(key)) {{
                uniqueLinksMap.set(key, l);
            }}
        }});
        const uniqueLinks = Array.from(uniqueLinksMap.values());

        // Render SVG Links
        const link = svg.append("g")
          .selectAll("path")
          .data(uniqueLinks)
          .join("path")
            .attr("class", "link")
            .attr("d", ([i, o]) => line(i.path(o)))
            .each(function(d) {{ d.path = this; }});

        // Render Country Labels
        const node = svg.append("g")
          .selectAll("g")
          .data(root.leaves())
          .join("g")
            .attr("transform", d => `rotate(${{d.x * 180 / Math.PI - 90}}) translate(${{d.y}},0)`);

        node.append("text")
            .attr("class", "node")
            .attr("dy", "0.31em")
            .attr("x", d => d.x < Math.PI ? 6 : -6)
            .attr("text-anchor", d => d.x < Math.PI ? "start" : "end")
            .attr("transform", d => d.x >= Math.PI ? "rotate(180)" : null)
            .attr("fill", d => colorScale(d.parent.data.name))
            .text(d => d.data.name)
            .each(function(d) {{ d.textNode = this; }})
            .on("mouseover", overed)
            .on("mouseout", outed);

        function overed(event, d) {{
            // 1. Dim all unselected links and country names
            link.style("stroke-opacity", 0.04).style("stroke-width", "1px");
            d3.selectAll(".node").style("opacity", 0.15);

            // 2. Highlight selected country label
            d3.select(d.textNode)
                .style("opacity", 1)
                .attr("font-weight", "bold")
                .attr("font-size", "14px");

            // 3. Filter links connected to node d
            const activeLinks = uniqueLinks.filter(l => l[0] === d || l[1] === d);

            // 4. Highlight active links and neighbor labels
            activeLinks.forEach(l => {{
                if (l.path) {{
                    d3.select(l.path)
                        .style("stroke-opacity", 1)
                        .style("stroke-width", "2.8px")
                        .style("stroke", "#e74c3c")
                        .raise();
                }}
                if (l[0] && l[0].textNode) {{
                    d3.select(l[0].textNode)
                        .style("opacity", 1)
                        .attr("font-weight", "bold")
                        .attr("font-size", "12px");
                }}
                if (l[1] && l[1].textNode) {{
                    d3.select(l[1].textNode)
                        .style("opacity", 1)
                        .attr("font-weight", "bold")
                        .attr("font-size", "12px");
                }}
            }});
        }}

        function outed(event, d) {{
            // Restore default view state
            link.style("stroke-opacity", 0.25)
                .style("stroke-width", "1px")
                .style("stroke", "#7f8c8d");

            d3.selectAll(".node")
                .style("opacity", 1)
                .attr("font-weight", "normal")
                .attr("font-size", "10px");
        }}

        function bilink(root) {{
            const map = new Map(root.leaves().map(d => [id(d), d]));
            for (const leaf of root.leaves()) {{
                leaf.node = leaf;
                leaf.incoming = [];
                leaf.outgoing = leaf.data.imports.map(i => [leaf, map.get(i)]).filter(d => d[1]);
            }}
            for (const leaf of root.leaves()) {{
                for (const o of leaf.outgoing) {{
                    o[1].incoming.push(o);
                }}
            }}
            return root;
        }}

        function id(node) {{
            return `${{node.parent ? id(node.parent) + "." : ""}}${{node.data.name}}`;
        }}
    </script>
</body>
</html>
"""
    with open(html_file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_file_path


def plot_distance_hierarchical_edge_bundling_by_year(
    year: int | str,
    continent_map: dict[str, str],
    top_k: int = 3,
    output_dir: Path | None = None,
    filepath_html: str | None = None,
    show_fig: bool = False,
) -> Path | None:
    """Exports the Hierarchical Edge Bundling chart for a given year based on Euclidean Distance."""
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT1_DIR / "distance_heb_plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    if filepath_html is None:
        html_file_path = output_dir / f"distance_heb_{year}.html"
    else:
        html_file_path = output_dir / filepath_html

    print(f"🔄 Processing HEB plot for year {year}...")

    # 1. Extract nearest country pairs
    df_pairs = get_close_distance_pairs_by_year(year=year, top_k=top_k)
    if df_pairs.empty:
        print(f"⚠️ No country pair data available for year {year}.")
        return None

    # 2. Generate standard HEB HTML file
    chart_title = f"Hierarchical Edge Bundling (Distance) - Year {year} (Top-{top_k} Neighbors)"
    generate_true_heb_html(
        df_pairs=df_pairs,
        continent_map=continent_map,
        html_file_path=html_file_path,
        title=chart_title,
    )

    print(f"✅ HEB chart created successfully: {html_file_path}")
    return html_file_path


def export_all_years_distance_heb_plots(
    continent_map: dict[str, str],
    years: list[int] | range | None = None,
    top_k: int = 3,
):
    """Automated runner function to export Distance HEB charts across all years."""
    if years is None:
        years = paths.YEARS

    print("=======================================================")
    print("STARTING DISTANCE HIERARCHICAL EDGE BUNDLING EXPORT")
    print("=======================================================\n")

    for yr in years:
        plot_distance_hierarchical_edge_bundling_by_year(
            year=yr,
            continent_map=continent_map,
            top_k=top_k,
            show_fig=False,
        )

    print("\n=======================================================")
    print("🎉 ALL DISTANCE HEB HTML FILES CREATED SUCCESSFULLY!")
    print("=======================================================")


def plot_cosine_hierarchical_edge_bundling_by_year(
    year: int | str,
    continent_map: dict[str, str],
    top_k: int = 3,
    output_dir: Path | None = None,
    filepath_html: str | None = None,
    show_fig: bool = False,
) -> Path | None:
    """Exports the Hierarchical Edge Bundling chart for a given year based on Cosine Similarity."""
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT1_DIR / "cosine_heb_plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    if filepath_html is None:
        html_file_path = output_dir / f"cosine_heb_{year}.html"
    else:
        html_file_path = output_dir / filepath_html

    print(f"🔄 Processing Cosine HEB plot for year {year}...")

    # 1. Extract nearest country pairs based on Cosine Similarity
    df_pairs = get_close_cosine_pairs_by_year(year=year, top_k=top_k)
    if df_pairs.empty:
        print(f"⚠️ No country pair data available for year {year} (Cosine).")
        return None

    # 2. Generate standard HEB HTML file
    chart_title = f"Hierarchical Edge Bundling (Cosine Similarity) - Year {year} (Top-{top_k} Neighbors)"
    generate_true_heb_html(
        df_pairs=df_pairs,
        continent_map=continent_map,
        html_file_path=html_file_path,
        title=chart_title,
    )

    print(f"✅ Cosine HEB chart created successfully: {html_file_path}")
    return html_file_path


def export_all_years_cosine_heb_plots(
    continent_map: dict[str, str],
    years: list[int] | range | None = None,
    top_k: int = 3,
):
    """Automated runner function to export Cosine HEB charts across all years."""
    if years is None:
        years = paths.YEARS

    print("=======================================================")
    print("STARTING COSINE HIERARCHICAL EDGE BUNDLING EXPORT")
    print("=======================================================\n")

    for yr in years:
        plot_cosine_hierarchical_edge_bundling_by_year(
            year=yr,
            continent_map=continent_map,
            top_k=top_k,
            show_fig=False,
        )

    print("\n=======================================================")
    print("🎉 ALL COSINE HEB HTML FILES CREATED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    # Generate Distance and Cosine HEB visualizations
    export_all_years_distance_heb_plots(continent_map=continent_map)
    export_all_years_cosine_heb_plots(continent_map=continent_map)