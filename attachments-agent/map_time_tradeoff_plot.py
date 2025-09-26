#!/usr/bin/env python3
"""
Academic-style trade-off plot for MAP@10 vs Average Time per Example
Comprehensive Retrieval Evaluation Results (Combined Dataset)

Refactored to load configuration metadata from YAML and allow configurable paths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PLOTLY_AVAILABLE = False


# Set the style for academic publications
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


# Marker generation defaults
BASE_MARKERS = [
    "o",
    "s",
    "^",
    "v",
    "D",
    "p",
    "h",
    "*",
    "P",
    "X",
    ">",
    "<",
    "+",
    "x",
    "d",
    "H",
    "8",
    "1",
    "2",
    "3",
    "4",
]
EDGE_COLORS = ["white", "black", "gray"]
MARKER_SIZES = [120, 140, 100, 160]


DEFAULT_DATA_PATH = Path(
    "output/combined_result.csv"
)
DEFAULT_CONFIG_PATH = Path(
    "config/plot/config_mapping.yaml"
)
DEFAULT_OUTPUT_DIR = Path("./")


PLOTLY_MARKER_SYMBOLS = [
    "circle",
    "square",
    "triangle-up",
    "triangle-down",
    "diamond",
    "pentagon",
    "hexagon",
    "star",
    "star-diamond",
    "x",
    "triangle-left",
    "triangle-right",
    "cross",
    "circle-open",
    "square-open",
    "triangle-up-open",
    "triangle-down-open",
    "diamond-open",
    "pentagon-open",
    "hexagon-open",
    "star-open",
    "x-open",
    "triangle-left-open",
    "triangle-right-open",
    "hourglass",
    "bowtie",
    "asterisk-open",
    "hash-open",
    "y-up-open",
    "y-down-open",
    "y-left-open",
    "y-right-open",
    "line-ew-open",
    "line-ns-open",
    "line-ne-open",
    "line-nw-open",
    "arrow-up",
    "arrow-down",
    "arrow-left",
    "arrow-right",
    "arrow-bar-up",
    "arrow-bar-down",
    "arrow-bar-left",
    "arrow-bar-right",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate trade-off plots for retrieval evaluations using configurable "
            "dataset and plotting metadata."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV dataset containing evaluation results.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML file that provides the configuration name mapping.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots and dashboards will be written.",
    )

    return parser.parse_args()


def load_config_mapping(config_path: Path) -> Dict[str, str]:
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration mapping YAML not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not data or "config_mapping" not in data:
        raise ValueError(
            "YAML config must contain a top-level 'config_mapping' dictionary"
        )

    mapping = data["config_mapping"]
    if not isinstance(mapping, dict):
        raise TypeError("'config_mapping' must be a dictionary of name mappings")

    return mapping


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["mean_average_precision", "avg_time_per_example", "config_name"])
    df = df[
        np.isfinite(df["mean_average_precision"]) &
        np.isfinite(df["avg_time_per_example"])
    ]

    if df.empty:
        raise ValueError("No valid data rows found after cleaning the dataset")

    return df


def apply_config_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["config_name_short"] = df["config_name"].map(mapping)
    unmapped = df["config_name_short"].isna()
    if unmapped.any():
        df.loc[unmapped, "config_name_short"] = df.loc[unmapped, "config_name"]
    return df


def generate_marker_styles(num_configs: int) -> Tuple[List[Dict[str, object]], int]:
    """Generate marker styles dynamically based on the number of configurations."""

    combinations: List[Dict[str, object]] = []
    for edge in EDGE_COLORS:
        for size in MARKER_SIZES:
            for marker in BASE_MARKERS:
                combinations.append({"marker": marker, "size": size, "edge": edge})

    if not combinations:
        raise RuntimeError("Failed to generate marker style combinations")

    styles: List[Dict[str, object]] = []
    for idx in range(num_configs):
        styles.append(combinations[idx % len(combinations)])

    return styles, len(combinations)


def assign_marker_info(unique_configs: Iterable[str]) -> Tuple[Dict[str, Dict[str, object]], int]:
    configs = list(unique_configs)
    styles, available = generate_marker_styles(len(configs))
    marker_info = {config: styles[idx] for idx, config in enumerate(configs)}
    return marker_info, available


def assign_colors(unique_configs: Iterable[str]) -> Tuple[Dict[str, tuple], Dict[str, str]]:
    configs = list(unique_configs)
    palette = sns.color_palette("husl", len(configs))
    colors = dict(zip(configs, palette))
    colors_plotly = dict(zip(configs, palette.as_hex()))
    return colors, colors_plotly


def assign_plotly_markers(unique_configs: Iterable[str]) -> Dict[str, str]:
    markers = {}
    symbols = PLOTLY_MARKER_SYMBOLS
    for idx, config in enumerate(unique_configs):
        markers[config] = symbols[idx % len(symbols)]
    return markers


def debug_marker_assignments(
    unique_configs: Iterable[str],
    marker_info: Dict[str, Dict[str, object]],
    available_styles: int,
    plotly_symbols: List[str],
) -> None:
    configs = list(unique_configs)
    print(f"Number of unique configurations: {len(configs)}")
    print(f"Number of available marker styles: {available_styles}")
    print(f"Number of available plotly markers: {len(plotly_symbols)}")
    print(f"Configuration mapping successful: {len(marker_info) == len(configs)}")
    print("Marker assignments (first 10):")
    for config in configs[:10]:
        style = marker_info[config]
        print(
            f"  {config}: {style['marker']} (size: {style['size']}, edge: {style['edge']})"
        )


def compute_pareto_points(df: pd.DataFrame, metric_col: str) -> List[Tuple[float, float, str]]:
    df_sorted = df.sort_values("avg_time_per_example")
    pareto_points: List[Tuple[float, float, str]] = []
    max_metric = -np.inf

    for _, row in df_sorted.iterrows():
        metric_value = row[metric_col]
        if metric_value > max_metric:
            max_metric = metric_value
            pareto_points.append(
                (
                    row["avg_time_per_example"],
                    metric_value,
                    row["config_name_short"],
                )
            )

    return pareto_points


def create_trade_off_plot(
    df: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    y_label: str,
    output_path: Path,
    colors: Dict[str, tuple],
    marker_info: Dict[str, Dict[str, object]],
    annotation_format: str = ".3f",
) -> None:
    """Create a static trade-off plot for a specific metric."""

    fig, ax = plt.subplots(figsize=(14, 10))

    for _, row in df.iterrows():
        config = row["config_name_short"]
        marker_style = marker_info[config]
        ax.scatter(
            row["avg_time_per_example"],
            row[metric_col],
            color=colors[config],
            marker=marker_style["marker"],
            s=marker_style["size"],
            alpha=0.8,
            edgecolors=marker_style["edge"],
            linewidth=1.5,
            label=config,
            zorder=3,
        )

    pareto_points = compute_pareto_points(df, metric_col)
    if len(pareto_points) > 1:
        pareto_x, pareto_y = zip(*[(x, y) for x, y, _ in pareto_points])
        ax.plot(
            pareto_x,
            pareto_y,
            "k--",
            alpha=0.5,
            linewidth=2,
            label="Pareto Frontier",
            zorder=2,
        )

    ax.set_xlabel("Average Time per Example (seconds)", fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.set_title(
        (
            f"Trade-off Analysis: {metric_name} vs. Computational Efficiency\n"
            "Comprehensive Retrieval Evaluation Results"
        ),
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    x_values = df["avg_time_per_example"].values
    y_values = df[metric_col].values

    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_padding = max(x_range * 0.1, 0.5) if x_range > 0 else 1.0
    y_padding = max(y_range * 0.05, 0.01) if y_range > 0 else 0.1

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    legend_elements = []
    for config in df["config_name_short"].unique():
        style = marker_info[config]
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="w",
                markerfacecolor=colors[config],
                markersize=10,
                markeredgecolor=style["edge"],
                markeredgewidth=1.5,
                label=config,
                linestyle="None",
            )
        )

    if len(pareto_points) > 1:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                alpha=0.5,
                linewidth=2,
                label="Pareto Frontier",
            )
        )

    ncols = min(4, max(2, len(legend_elements) // 10))
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=ncols,
    )

    best_metric = df.loc[df[metric_col].idxmax()]
    fastest = df.loc[df["avg_time_per_example"].idxmin()]

    ax.annotate(
        (
            f"Highest {metric_name}\n({best_metric['config_name_short']})\n"
            f"{metric_name}: {best_metric[metric_col]:{annotation_format}}"
        ),
        xy=(best_metric["avg_time_per_example"], best_metric[metric_col]),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"),
        fontsize=10,
        ha="left",
    )

    ax.annotate(
        (
            f"Fastest\n({fastest['config_name_short']})\n"
            f"Time: {fastest['avg_time_per_example']:.2f}s"
        ),
        xy=(fastest["avg_time_per_example"], fastest[metric_col]),
        xytext=(10, -40),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1"),
        fontsize=10,
        ha="left",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    plt.close(fig)


def create_trade_off_plot_plotly(
    df: pd.DataFrame,
    metric_col: str,
    metric_name: str,
    y_label: str,
    colors_plotly: Dict[str, str],
    plotly_markers: Dict[str, str],
    annotation_format: str = ".3f",
):
    """Create an interactive trade-off plot for a specific metric using Plotly."""

    if not PLOTLY_AVAILABLE:
        return None

    pareto_points = compute_pareto_points(df, metric_col)
    fig = go.Figure()
    seen_configs = set()

    for _, row in df.iterrows():
        config = row["config_name_short"]
        hover_text = (
            f"Config: {config}<br>"
            f"Avg Time: {row['avg_time_per_example']:.2f}s<br>"
            f"{metric_name}: {row[metric_col]:{annotation_format}}<br>"
            f"MRR: {row['mean_reciprocal_rank']:.3f}<br>"
            f"Success Rate: {row['success_rate']:.1%}"
        )

        fig.add_trace(
            go.Scatter(
                x=[row["avg_time_per_example"]],
                y=[row[metric_col]],
                mode="markers",
                marker=dict(
                    color=colors_plotly[config],
                    symbol=plotly_markers[config],
                    size=14,
                    line=dict(color="white", width=1.5),
                ),
                name=config,
                legendgroup=config,
                showlegend=config not in seen_configs,
                hovertemplate=hover_text + "<extra></extra>",
            )
        )

        seen_configs.add(config)

    if len(pareto_points) > 1:
        fig.add_trace(
            go.Scatter(
                x=[x for x, _, _ in pareto_points],
                y=[y for _, y, _ in pareto_points],
                mode="lines",
                line=dict(color="black", dash="dash", width=2),
                name="Pareto Frontier",
            )
        )

    fig.update_layout(
        title=(
            f"Trade-off Analysis: {metric_name} vs. Computational Efficiency<br>"
            "Comprehensive Retrieval Evaluation Results"
        ),
        xaxis_title="Average Time per Example (seconds)",
        yaxis_title=y_label,
        legend_title="Configuration",
        template="plotly_white",
        hovermode="closest",
        width=1000,
        height=700,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

    return fig


def save_interactive_dashboard(
    figures: List[Tuple[str, "go.Figure"]], output_path: Path
) -> None:
    dashboard_sections: List[str] = []
    for idx, (metric_label, fig) in enumerate(figures):
        include_js = "cdn" if idx == 0 else False
        snippet = pio.to_html(
            fig,
            include_plotlyjs=include_js,
            full_html=False,
            auto_play=False,
        )
        dashboard_sections.append(
            f"<section>\n<h2>{metric_label}</h2>\n"
            f"<div class=\"plot-container\">{snippet}</div>\n</section>"
        )

    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Trade-off Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ margin-bottom: 20px; }}
        h2 {{ margin-top: 0; }}
        section {{ margin-bottom: 60px; }}
        .plot-container {{ max-width: 1100px; }}
    </style>
</head>
<body>
    <h1>Comprehensive Retrieval Evaluation Results<br>Interactive Trade-off Dashboard</h1>
    {sections}
</body>
</html>"""

    output_path.write_text(
        dashboard_html.format(sections="\n".join(dashboard_sections)),
        encoding="utf-8",
    )

    print(f"Saved combined interactive dashboard to {output_path}")


def main() -> None:
    args = parse_args()

    data_path = args.data.expanduser()
    config_path = args.config.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_mapping = load_config_mapping(config_path)
    df = load_dataset(data_path)
    df = apply_config_mapping(df, config_mapping)

    unique_configs = df["config_name_short"].unique()
    colors, colors_plotly = assign_colors(unique_configs)
    marker_info, available_styles = assign_marker_info(unique_configs)
    plotly_markers = assign_plotly_markers(unique_configs)

    debug_marker_assignments(unique_configs, marker_info, available_styles, PLOTLY_MARKER_SYMBOLS)

    interactive_figures: List[Tuple[str, go.Figure]] = [] if PLOTLY_AVAILABLE else []

    plots = [
        (
            "mean_average_precision",
            "MAP@10",
            "Mean Average Precision @ 10",
            ".3f",
            f"01_map_time_tradeoff_{timestamp}.png",
            ".3f",
        ),
        (
            "mean_reciprocal_rank",
            "MRR",
            "Mean Reciprocal Rank",
            ".3f",
            f"02_mrr_time_tradeoff_{timestamp}.png",
            ".3f",
        ),
        (
            "success_rate",
            "Success Rate",
            "Success Rate",
            ".1%",
            f"03_success_rate_time_tradeoff_{timestamp}.png",
            ".1%",
        ),
    ]

    for metric_col, metric_name, y_label, annotation_format, filename, plotly_format in plots:
        output_path = output_dir / filename
        create_trade_off_plot(
            df,
            metric_col,
            metric_name,
            y_label,
            output_path,
            colors,
            marker_info,
            annotation_format=annotation_format,
        )

        if PLOTLY_AVAILABLE:
            fig = create_trade_off_plot_plotly(
                df,
                metric_col,
                metric_name,
                y_label,
                colors_plotly,
                plotly_markers,
                annotation_format=plotly_format,
            )
            if fig is not None:
                interactive_figures.append((metric_name, fig))

    if interactive_figures and PLOTLY_AVAILABLE:
        dashboard_path = output_dir / f"tradeoff_dashboard_{timestamp}.html"
        save_interactive_dashboard(interactive_figures, dashboard_path)
    elif not PLOTLY_AVAILABLE:
        print("\nPlotly is not installed. Skipping interactive plot generation.")


if __name__ == "__main__":
    main()
