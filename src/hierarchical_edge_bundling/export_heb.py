# src/hierarchical_edge_bundling/export_heb.py
# cd "C:\Users\admin\Documents\Code_for_fun\country-data-fingerprint"
# python -m src.hierarchical_edge_bundling.export_heb

import json
from pathlib import Path
import pandas as pd
import numpy as np
from config import paths


# Bảng tra cứu Châu lục cho 194 quốc gia (Mã ISO 3 - Không chứa Vatican VAT)
continent_map = {
    # ================= Châu Á (Asia) - 47 nước =================
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

    # ================= Châu Âu (Europe) - 43 nước =================
    'ALB': 'Europe', 'AND': 'Europe', 'AUT': 'Europe', 'BEL': 'Europe', 'BGR': 'Europe', 
    'BIH': 'Europe', 'BLR': 'Europe', 'CHE': 'Europe', 'CZE': 'Europe', 'DEU': 'Europe', 
    'DNK': 'Europe', 'ESP': 'Europe', 'EST': 'Europe', 'FIN': 'Europe', 'FRA': 'Europe', 
    'GBR': 'Europe', 'GRC': 'Europe', 'HRV': 'Europe', 'HUN': 'Europe', 'IRL': 'Europe', 
    'ISL': 'Europe', 'ITA': 'Europe', 'LIE': 'Europe', 'LTU': 'Europe', 'LUX': 'Europe', 
    'LVA': 'Europe', 'MCO': 'Europe', 'MDA': 'Europe', 'MKD': 'Europe', 'MLT': 'Europe', 
    'MNE': 'Europe', 'NLD': 'Europe', 'NOR': 'Europe', 'POL': 'Europe', 'PRT': 'Europe', 
    'ROU': 'Europe', 'RUS': 'Europe', 'SMR': 'Europe', 'SRB': 'Europe', 'SVK': 'Europe', 
    'SVN': 'Europe', 'SWE': 'Europe', 'UKR': 'Europe', 

    # ================= Châu Phi (Africa) - 54 nước =================
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

    # ================= Bắc Mỹ (North America) - 23 nước =================
    'ATG': 'North America', 'BHS': 'North America', 'BLZ': 'North America', 
    'BRB': 'North America', 'CAN': 'North America', 'CRI': 'North America', 
    'CUB': 'North America', 'DMA': 'North America', 'DOM': 'North America', 
    'GRD': 'North America', 'GTM': 'North America', 'HND': 'North America', 
    'HTI': 'North America', 'JAM': 'North America', 'KNA': 'North America', 
    'LCA': 'North America', 'MEX': 'North America', 'NIC': 'North America', 
    'PAN': 'North America', 'SLV': 'North America', 'TTO': 'North America', 
    'USA': 'North America', 'VCT': 'North America',

    # ================= Nam Mỹ (South America) - 12 nước =================
    'ARG': 'South America', 'BOL': 'South America', 'BRA': 'South America', 
    'CHL': 'South America', 'COL': 'South America', 'ECU': 'South America', 
    'GUY': 'South America', 'PER': 'South America', 'PRY': 'South America', 
    'SUR': 'South America', 'URY': 'South America', 'VEN': 'South America',

    # ================= Châu Đại Dương (Oceania) - 14 nước =================
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
    """Đọc ma trận khoảng cách Euclidean của một năm và lấy Top-K hàng xóm gần nhất (loại bỏ chính nó)."""
    if matrices_dir is None:
        matrices_dir = paths.RESULTS_EXPERIMENT1_DIR / "distance_matrices"

    file_path = matrices_dir / f"euclidean_matrix_{year}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file ma trận: {file_path}")

    # 1. Đọc ma trận khoảng cách
    df_mat = pd.read_csv(file_path, index_col=0)

    # 2. Lấy Top-K hàng xóm gần nhất cho từng quốc gia (loại bỏ chính nó)
    pairs = []
    for country, row in df_mat.iterrows():
        # ĐẢM BẢO: Loại bỏ chính quốc gia đó ra khỏi danh sách so sánh
        row_no_self = row.drop(labels=[country], errors="ignore")
        
        # Lấy top_k quốc gia có khoảng cách nhỏ nhất
        top_k_neighbors = row_no_self.nsmallest(top_k).index.tolist()
        
        for neighbor in top_k_neighbors:
            pairs.append({"Country1": country, "Country2": neighbor})

    df_pairs = pd.DataFrame(pairs)

    # 3. Sắp xếp cặp vô hướng và loại bỏ trùng lặp & loại bỏ self-loop (nếu có)
    sorted_pairs = np.sort(df_pairs[["Country1", "Country2"]].values, axis=1)
    df_filtered = (
        pd.DataFrame(sorted_pairs, columns=["Country1", "Country2"])
        .drop_duplicates()
    )
    
    # Bổ sung bộ lọc loại bỏ các cặp trùng tên (Country1 == Country2)
    df_filtered = df_filtered[df_filtered["Country1"] != df_filtered["Country2"]].reset_index(drop=True)

    return df_filtered
def get_close_cosine_pairs_by_year(
    year: int | str,
    top_k: int = 3,
    matrices_dir: Path | None = None,
) -> pd.DataFrame:
    """Đọc ma trận Cosine của một năm và lấy Top-K hàng xóm gần nhất (loại bỏ chính nó)."""
    if matrices_dir is None:
        matrices_dir = paths.RESULTS_EXPERIMENT1_DIR / "cosine_matrices"

    file_path = matrices_dir / f"cosine_similarity_matrix_{year}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file ma trận Cosine: {file_path}")

    # 1. Đọc ma trận Cosine
    df_mat = pd.read_csv(file_path, index_col=0)

    # 2. Lấy Top-K hàng xóm gần nhất cho từng quốc gia (loại bỏ chính nó)
    pairs = []
    for country, row in df_mat.iterrows():
        row_no_self = row.drop(labels=[country], errors="ignore")
        
        # Nếu ma trận là Cosine Distance (giá trị càng nhỏ càng gần): dùng nsmallest
        # (Nếu ma trận của bạn là Cosine Similarity - giá trị càng lớn càng gần: đổi thành nlargest)
        top_k_neighbors = row_no_self.nlargest(top_k).index.tolist()
        
        for neighbor in top_k_neighbors:
            pairs.append({"Country1": country, "Country2": neighbor})

    df_pairs = pd.DataFrame(pairs)

    # 3. Sắp xếp cặp vô hướng & xóa trùng lặp & loại bỏ self-loop
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
    Tạo biểu đồ Hierarchical Edge Bundling chuẩn D3.js:
    - Gán kết nối 2 chiều hai bên đảm bảo ĐỦ TỐI THIỂU k dây cho mỗi nước.
    - Khử trùng lặp đường vẽ SVG trong D3 để nét vẽ mịn đẹp.
    - Highlight chính xác k dây nối và k hàng xóm khi hover.
    """
    col_a = df_pairs.columns[0]
    col_b = df_pairs.columns[1]

    # 1. Lấy danh sách tất cả các quốc gia có trong df_pairs
    active_countries = set(df_pairs[col_a].astype(str).str.strip()).union(
        set(df_pairs[col_b].astype(str).str.strip())
    )

    # 2. Thu thập kết nối 2 CHIỀU (Bi-directional)
    connections = {}
    for _, row in df_pairs.iterrows():
        c1 = str(row[col_a]).strip()
        c2 = str(row[col_b]).strip()
        if c1 != c2:
            cont1 = continent_map.get(c1, 'Other')
            cont2 = continent_map.get(c2, 'Other')
            p1 = f"World.{cont1}.{c1}"
            p2 = f"World.{cont2}.{c2}"

            # Gán kết nối 2 chiều cho cả c1 và c2
            if p1 not in connections:
                connections[p1] = set()
            connections[p1].add(p2)

            if p2 not in connections:
                connections[p2] = set()
            connections[p2].add(p1)

    # 3. Tạo Cây phân cấp
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

    # 4. Mẫu HTML/JS D3.js v7 chuẩn
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

        // KHỬ TRÙNG LẶP DÂY VẼ (Chỉ giữ 1 đường nối duy nhất cho mỗi cặp)
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

        // Vẽ dây nối SVG
        const link = svg.append("g")
          .selectAll("path")
          .data(uniqueLinks)
          .join("path")
            .attr("class", "link")
            .attr("d", ([i, o]) => line(i.path(o)))
            .each(function(d) {{ d.path = this; }});

        // Tên các nước
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
            // 1. Làm mờ toàn bộ dây nối và tên nước khác
            link.style("stroke-opacity", 0.04).style("stroke-width", "1px");
            d3.selectAll(".node").style("opacity", 0.15);

            // 2. Highlight tên nước hiện tại
            d3.select(d.textNode)
                .style("opacity", 1)
                .attr("font-weight", "bold")
                .attr("font-size", "14px");

            // 3. Lọc tất cả các dây trong uniqueLinks có dính tới node d
            const activeLinks = uniqueLinks.filter(l => l[0] === d || l[1] === d);

            // 4. Highlight chính xác các dây đó và các nước hàng xóm
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
            // Khôi phục lại trạng thái ban đầu
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
    """Xuất biểu đồ Hierarchical Edge Bundling cho một năm nhất định."""
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT1_DIR / "distance_heb_plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    if filepath_html is None:
        html_file_path = output_dir / f"distance_heb_{year}.html"
    else:
        html_file_path = output_dir / filepath_html

    print(f"🔄 Đang xử lý biểu đồ HEB cho năm {year}...")

    # 1. Trích xuất dữ liệu cặp quốc gia gần nhất
    df_pairs = get_close_distance_pairs_by_year(year=year, top_k=top_k)
    if df_pairs.empty:
        print(f"⚠️ Năm {year} không có dữ liệu cặp quốc gia.")
        return None

    # 2. Sinh file HTML Hierarchical Edge Bundling chuẩn
    chart_title = f"Hierarchical Edge Bundling (Distance) - Year {year} (Top-{top_k} Neighbors)"
    generate_true_heb_html(
        df_pairs=df_pairs,
        continent_map=continent_map,
        html_file_path=html_file_path,
        title=chart_title,
    )

    print(f"✅ Đã tạo biểu đồ HEB thành công: {html_file_path}")
    return html_file_path


def export_all_years_distance_heb_plots(
    continent_map: dict[str, str],
    years: list[int] | range | None = None,
    top_k: int = 3,
):
    """Hàm chạy tự động cho tất cả các năm."""
    if years is None:
        years = paths.YEARS

    print("=======================================================")
    print("BẮT ĐẦU XUẤT CÁC BIỂU ĐỒ DISTANCE HIERARCHICAL EDGE BUNDLING")
    print("=======================================================\n")

    for yr in years:
        plot_distance_hierarchical_edge_bundling_by_year(
            year=yr,
            continent_map=continent_map,
            top_k=top_k,
            show_fig=False,
        )

    print("\n=======================================================")
    print("🎉 TẤT CẢ CÁC FILE DISTANCE HEB HTML ĐÃ ĐƯỢC TẠO HOÀN TẤT!")
    print("=======================================================")


def plot_cosine_hierarchical_edge_bundling_by_year(
    year: int | str,
    continent_map: dict[str, str],
    top_k: int = 3,
    output_dir: Path | None = None,
    filepath_html: str | None = None,
    show_fig: bool = False,
) -> Path | None:
    """Xuất biểu đồ Hierarchical Edge Bundling theo khoảng cách Cosine cho một năm nhất định."""
    if output_dir is None:
        output_dir = paths.RESULTS_EXPERIMENT1_DIR / "cosine_heb_plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    if filepath_html is None:
        html_file_path = output_dir / f"cosine_heb_{year}.html"
    else:
        html_file_path = output_dir / filepath_html

    print(f"🔄 Đang xử lý biểu đồ Cosine HEB cho năm {year}...")

    # 1. Trích xuất dữ liệu cặp quốc gia gần nhất theo khoảng cách Cosine
    df_pairs = get_close_cosine_pairs_by_year(year=year, top_k=top_k)
    if df_pairs.empty:
        print(f"⚠️ Năm {year} không có dữ liệu cặp quốc gia (Cosine).")
        return None

    # 2. Sinh file HTML Hierarchical Edge Bundling chuẩn
    chart_title = f"Hierarchical Edge Bundling (Cosine Similarity) - Year {year} (Top-{top_k} Neighbors)"
    generate_true_heb_html(
        df_pairs=df_pairs,
        continent_map=continent_map,
        html_file_path=html_file_path,
        title=chart_title,
    )

    print(f"✅ Đã tạo biểu đồ Cosine HEB thành công: {html_file_path}")
    return html_file_path


def export_all_years_cosine_heb_plots(
    continent_map: dict[str, str],
    years: list[int] | range | None = None,
    top_k: int = 3,
):
    """Hàm chạy tự động xuất biểu đồ Cosine HEB cho tất cả các năm."""
    if years is None:
        years = paths.YEARS

    print("=======================================================")
    print("BẮT ĐẦU XUẤT CÁC BIỂU ĐỒ COSINE HIERARCHICAL EDGE BUNDLING")
    print("=======================================================\n")

    for yr in years:
        plot_cosine_hierarchical_edge_bundling_by_year(
            year=yr,
            continent_map=continent_map,
            top_k=top_k,
            show_fig=False,
        )

    print("\n=======================================================")
    print("🎉 TẤT CẢ CÁC FILE COSINE HEB HTML ĐÃ ĐƯỢC TẠO HOÀN TẤT!")
    print("=======================================================")

if __name__ == "__main__":
    # ra 2 matrices distance va cosine
    export_all_years_distance_heb_plots(continent_map=continent_map)
    export_all_years_cosine_heb_plots(continent_map=continent_map)
