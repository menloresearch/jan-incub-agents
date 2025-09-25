#!/usr/bin/env python3
"""
Academic-style trade-off plot for MAP@10 vs Average Time per Example
Comprehensive Retrieval Evaluation Results (Combined Dataset)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Set the style for academic publications
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Read the data
df = pd.read_csv('/mnt/nas/norapat/jan-incub-agents/attachments-agent/output/combined_result.csv')

# Clean and prepare the data
# Remove rows with NaN values in key columns
df = df.dropna(subset=['mean_average_precision', 'avg_time_per_example', 'config_name'])

# Remove any infinite values
df = df[np.isfinite(df['mean_average_precision']) & np.isfinite(df['avg_time_per_example'])]

if len(df) == 0:
    print("ERROR: No valid data rows found!")
    exit(1)

# Create a more readable configuration name mapping
config_mapping = {
    # BM25 configurations
    'BM25 Baseline (400 chunks)': 'BM25 Baseline (PyPDF2)',
    'BM25 + Cross-encoder (MS MARCO)': 'BM25 + Cross-encoder (PyPDF2)',
    'BM25 Baseline (400 chunks, MarkItDown)': 'BM25 (MarkItDown)',
    'BM25 Baseline (400 chunks, Docling)': 'BM25 (Docling)',
    'BM25 Baseline (400 chunks, Marker)': 'BM25 (Marker)',
    
    # SentenceTransformers configurations - PyPDF2
    'SentenceTransformers MiniLM (400 chunks)': 'ST MiniLM (PyPDF2)',
    'SentenceTransformers Qwen3 (400 chunks)': 'ST Qwen3-0.6B (PyPDF2)',
    'SentenceTransformers Qwen3 (800 chunks)': 'ST Qwen3-0.6B (800)',
    'SentenceTransformers Qwen3 (1200 chunks)': 'ST Qwen3-0.6B (1200)',
    'SentenceTransformers Qwen3 4B (400 chunks)': 'ST Qwen3-4B (PyPDF2)',
    'SentenceTransformers Qwen3 8B (400 chunks)': 'ST Qwen3-8B (PyPDF2)',
    
    # EmbeddingGemma configurations
    'SentenceTransformers EmbeddingGemma (400 chunks, PyPDF2)': 'ST EmbeddingGemma (PyPDF2)',
    'SentenceTransformers EmbeddingGemma (400 chunks, Docling)': 'ST EmbeddingGemma (Docling)',
    'ST Gemma 9B (400 chunks, Marker)': 'ST EmbeddingGemma (Marker)',
    
    # Instruct configurations
    'Instruct Qwen3 Embedding (400 chunks)': 'Instruct Qwen3-0.6B (PyPDF2)',
    'Instruct Qwen3 Embedding (800 chunks)': 'Instruct Qwen3-0.6B (800)',
    'Instruct EmbeddingGemma (400 chunks)': 'Instruct EmbeddingGemma (PyPDF2)',
    'Instruct Qwen3 + Cross-encoder (MS MARCO)': 'Instruct Qwen3 + Cross-encoder',
    
    # Cross-encoder configurations
    'ST MiniLM + Cross-encoder (MS MARCO)': 'ST MiniLM + Cross-encoder',
    'ST Qwen3 + Cross-encoder (MS MARCO)': 'ST Qwen3 + Cross-encoder',
    
    # Qwen Reranker configurations - PyPDF2
    'ST Qwen3 + Qwen Reranker': 'ST Qwen3 + Qwen Reranker (PyPDF2)',
    'ST Qwen3 + Qwen Reranker (800 chunks)': 'ST Qwen3 + Qwen Reranker (800)',
    'ST Qwen3 4B + Qwen4B Reranker': 'ST Qwen3-4B + Reranker (PyPDF2)',
    'ST Qwen3 8B + Qwen8B Reranker': 'ST Qwen3-8B + Reranker (PyPDF2)',
    
    # Processor-specific configurations - MarkItDown
    'ST Qwen3 (400 chunks, MarkItDown)': 'ST Qwen3-0.6B (MarkItDown)',
    'ST Qwen3 4B (400 chunks, MarkItDown)': 'ST Qwen3-4B (MarkItDown)',
    'ST Qwen3 8B (400 chunks, MarkItDown)': 'ST Qwen3-8B (MarkItDown)',
    'ST Qwen3 + Qwen Reranker (MarkItDown)': 'ST Qwen3 + Qwen Reranker (MarkItDown)',
    'ST Qwen3 4B + Qwen4B Reranker (MarkItDown)': 'ST Qwen3-4B + Reranker (MarkItDown)',
    'ST Qwen3 8B + Qwen8B Reranker (MarkItDown)': 'ST Qwen3-8B + Reranker (MarkItDown)',
    
    # Processor-specific configurations - Docling
    'ST Qwen3 (400 chunks, Docling)': 'ST Qwen3-0.6B (Docling)',
    'ST Qwen3 4B (400 chunks, Docling)': 'ST Qwen3-4B (Docling)',
    'ST Qwen3 8B (400 chunks, Docling)': 'ST Qwen3-8B (Docling)',
    'ST Qwen3 4B + Qwen4B Reranker (Docling)': 'ST Qwen3-4B + Reranker (Docling)',
    'ST Qwen3 8B + Qwen8B Reranker (Docling)': 'ST Qwen3-8B + Reranker (Docling)',
    
    # Processor-specific configurations - Marker
    'ST Qwen3 4B (400 chunks, Marker)': 'ST Qwen3-4B (Marker)',
    'ST Qwen3 8B (400 chunks, Marker)': 'ST Qwen3-8B (Marker)',
    
    # Additional multilingual and specialized models
    'BAAI BGE-M3 (400 chunks)': 'BAAI BGE-M3 (PyPDF2)',
    'Multilingual E5 Large (400 chunks)': 'Multilingual E5 Large (PyPDF2)',
    'Snowflake Arctic Embed Large v2.0 (400 chunks)': 'Snowflake Arctic Embed v2.0 (PyPDF2)'
}

df['config_name_short'] = df['config_name'].map(config_mapping)

# Handle unmapped config names
unmapped = df['config_name_short'].isna()
if unmapped.any():
    # Use original names for unmapped entries
    df.loc[unmapped, 'config_name_short'] = df.loc[unmapped, 'config_name']

# Define colors and markers for different categories (shared across all plots)
unique_configs = df['config_name_short'].unique()
colors_palette = sns.color_palette("husl", len(unique_configs))
colors = dict(zip(unique_configs, colors_palette))
colors_plotly = dict(zip(unique_configs, colors_palette.as_hex()))

# Use only well-tested matplotlib markers that render reliably
markers_list = ['o', 's', '^', 'v', 'D', 'p', 'h', '*', 'P', 'X', '>', '<', 
               '+', 'x', 'd', 'H', '8', '1', '2', '3', '4']

# For configurations beyond basic markers, we'll cycle through with different styles
# This gives us visual variety even when markers repeat - expanded to handle 40+ configs
marker_styles = [
    # First set - white edges, size 120
    {'marker': 'o', 'size': 120, 'edge': 'white'},
    {'marker': 's', 'size': 120, 'edge': 'white'},
    {'marker': '^', 'size': 120, 'edge': 'white'},
    {'marker': 'v', 'size': 120, 'edge': 'white'},
    {'marker': 'D', 'size': 120, 'edge': 'white'},
    {'marker': 'p', 'size': 120, 'edge': 'white'},
    {'marker': 'h', 'size': 120, 'edge': 'white'},
    {'marker': '*', 'size': 150, 'edge': 'white'},
    {'marker': 'P', 'size': 120, 'edge': 'white'},
    {'marker': 'X', 'size': 120, 'edge': 'white'},
    {'marker': '>', 'size': 120, 'edge': 'white'},
    {'marker': '<', 'size': 120, 'edge': 'white'},
    {'marker': '+', 'size': 150, 'edge': 'white'},
    {'marker': 'x', 'size': 150, 'edge': 'white'},
    {'marker': 'd', 'size': 120, 'edge': 'white'},
    {'marker': 'H', 'size': 120, 'edge': 'white'},
    {'marker': '8', 'size': 120, 'edge': 'white'},
    {'marker': '1', 'size': 120, 'edge': 'white'},
    {'marker': '2', 'size': 120, 'edge': 'white'},
    {'marker': '3', 'size': 120, 'edge': 'white'},
    # Second set - black edges, size 140
    {'marker': 'o', 'size': 140, 'edge': 'black'},
    {'marker': 's', 'size': 140, 'edge': 'black'},
    {'marker': '^', 'size': 140, 'edge': 'black'},
    {'marker': 'v', 'size': 140, 'edge': 'black'},
    {'marker': 'D', 'size': 140, 'edge': 'black'},
    {'marker': 'p', 'size': 140, 'edge': 'black'},
    {'marker': 'h', 'size': 140, 'edge': 'black'},
    {'marker': '*', 'size': 170, 'edge': 'black'},
    {'marker': 'P', 'size': 140, 'edge': 'black'},
    {'marker': 'X', 'size': 140, 'edge': 'black'},
    {'marker': '>', 'size': 140, 'edge': 'black'},
    {'marker': '<', 'size': 140, 'edge': 'black'},
    {'marker': '+', 'size': 170, 'edge': 'black'},
    {'marker': 'x', 'size': 170, 'edge': 'black'},
    {'marker': 'd', 'size': 140, 'edge': 'black'},
    {'marker': 'H', 'size': 140, 'edge': 'black'},
    {'marker': '8', 'size': 140, 'edge': 'black'},
    {'marker': '1', 'size': 140, 'edge': 'black'},
    {'marker': '2', 'size': 140, 'edge': 'black'},
    {'marker': '3', 'size': 140, 'edge': 'black'},
    # Third set - gray edges, size 100
    {'marker': 'o', 'size': 100, 'edge': 'gray'},
    {'marker': 's', 'size': 100, 'edge': 'gray'},
    {'marker': '^', 'size': 100, 'edge': 'gray'},
    {'marker': 'v', 'size': 100, 'edge': 'gray'},
    {'marker': 'D', 'size': 100, 'edge': 'gray'},
]

# Create markers dictionary with enhanced styling
markers = {}
marker_info = {}  # Store complete marker information including size and edge
for i, config in enumerate(unique_configs):
    style_idx = i % len(marker_styles)
    style = marker_styles[style_idx]
    markers[config] = style['marker']
    marker_info[config] = style

# Expanded Plotly marker symbols to handle more configurations
plotly_marker_symbols = [
    'circle', 'square', 'triangle-up', 'triangle-down', 'diamond',
    'pentagon', 'hexagon', 'star', 'star-diamond', 'x',
    'triangle-left', 'triangle-right', 'cross', 'circle-open',
    'square-open', 'triangle-up-open', 'triangle-down-open', 'diamond-open',
    'pentagon-open', 'hexagon-open', 'star-open', 'x-open',
    'triangle-left-open', 'triangle-right-open', 'hourglass', 'bowtie',
    'asterisk-open', 'hash-open', 'y-up-open', 'y-down-open',
    'y-left-open', 'y-right-open', 'line-ew-open', 'line-ns-open',
    'line-ne-open', 'line-nw-open', 'arrow-up', 'arrow-down',
    'arrow-left', 'arrow-right', 'arrow-bar-up', 'arrow-bar-down',
    'arrow-bar-left', 'arrow-bar-right'
]

# Create Plotly markers dictionary with cycling if we have more configs than markers
plotly_markers = {}
for i, config in enumerate(unique_configs):
    plotly_markers[config] = plotly_marker_symbols[i % len(plotly_marker_symbols)]

# Debug information
print(f"Number of unique configurations: {len(unique_configs)}")
print(f"Number of available marker styles: {len(marker_styles)}")
print(f"Number of available plotly markers: {len(plotly_marker_symbols)}")
print(f"Configuration mapping successful: {len(markers) == len(unique_configs)}")
print("Marker assignments:")
for i, config in enumerate(unique_configs[:10]):  # Show first 10 for brevity
    style = marker_info[config]
    print(f"  {config}: {style['marker']} (size: {style['size']}, edge: {style['edge']})")


def compute_pareto_points(df, metric_col):
    df_sorted = df.sort_values('avg_time_per_example')
    pareto_points = []
    max_metric = -np.inf

    for _, row in df_sorted.iterrows():
        metric_value = row[metric_col]
        if metric_value > max_metric:
            max_metric = metric_value
            pareto_points.append((row['avg_time_per_example'], metric_value, row['config_name_short']))

    return pareto_points

# Function to create a single plot
def create_trade_off_plot(df, metric_col, metric_name, y_label, filename_suffix, annotation_format=".3f"):
    """Create a static trade-off plot for a specific metric"""
    
    # Create the figure with academic styling - taller to accommodate legend below
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot each configuration
    for idx, row in df.iterrows():
        config = row['config_name_short']
        x = row['avg_time_per_example']
        y = row[metric_col]

        
        # Get marker style information
        marker_style = marker_info[config]
        
        ax.scatter(
            x,
            y,
            color=colors[config],
            marker=marker_style['marker'],
            s=marker_style['size'],
            alpha=0.8,
            edgecolors=marker_style['edge'],
            linewidth=1.5,
            label=config,
            zorder=3
        )
    
    # Add Pareto frontier
    pareto_points = compute_pareto_points(df, metric_col)
    if len(pareto_points) > 1:
        pareto_x, pareto_y = zip(*[(x, y) for x, y, _ in pareto_points])
        ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=2, label='Pareto Frontier', zorder=2)
    
    # Styling
    ax.set_xlabel('Average Time per Example (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(f'Trade-off Analysis: {metric_name} vs. Computational Efficiency\nComprehensive Retrieval Evaluation Results', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits with some padding - handle edge cases
    x_values = df['avg_time_per_example'].values
    y_values = df[metric_col].values
    
    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)
    
    # Add padding, but handle case where min == max
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_padding = max(x_range * 0.1, 0.5) if x_range > 0 else 1.0
    y_padding = max(y_range * 0.05, 0.01) if y_range > 0 else 0.1
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Create legend with better organization - include both color and marker shape
    legend_elements = []
    for config in df['config_name_short'].unique():
        marker_style = marker_info[config]
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker_style['marker'],
                color='w',
                markerfacecolor=colors[config],
                markersize=10,
                markeredgecolor=marker_style['edge'],
                markeredgewidth=1.5,
                label=config,
                linestyle='None'
            )
        )
    
    # Add Pareto frontier to legend if it exists
    if len(pareto_points) > 1:
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', 
                                        alpha=0.5, linewidth=2, label='Pareto Frontier'))
    
    # Place legend below the plot with multiple columns for better readability
    ncols = min(4, max(2, len(legend_elements) // 10))  # 2-4 columns based on number of items
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
              fontsize=10, frameon=True, fancybox=True, shadow=True, ncol=ncols)
    
    # Add annotations for best performers
    best_metric = df.loc[df[metric_col].idxmax()]
    fastest = df.loc[df['avg_time_per_example'].idxmin()]
    
    # Annotate best metric
    ax.annotate(f'Highest {metric_name}\n({best_metric["config_name_short"]})\n{metric_name}: {best_metric[metric_col]:{annotation_format}}',
                xy=(best_metric['avg_time_per_example'], best_metric[metric_col]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'),
                fontsize=10, ha='left')
    
    # Annotate fastest
    ax.annotate(f'Fastest\n({fastest["config_name_short"]})\nTime: {fastest["avg_time_per_example"]:.2f}s',
                xy=(fastest['avg_time_per_example'], fastest[metric_col]),
                xytext=(10, -40), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.1'),
                fontsize=10, ha='left')
    
    # Adjust layout to accommodate legend below
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend below
    
    # Save the plot (PNG only)
    plt.savefig(f'/mnt/nas/norapat/jan-incub-agents/attachments-agent/{filename_suffix}', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.show()
    plt.close()


def create_trade_off_plot_plotly(df, metric_col, metric_name, y_label, filename_base, annotation_format=".3f"):
    """Create an interactive trade-off plot for a specific metric using Plotly."""

    if not PLOTLY_AVAILABLE:
        return None, None

    pareto_points = compute_pareto_points(df, metric_col)

    fig = go.Figure()
    seen_configs = set()

    for _, row in df.iterrows():
        config = row['config_name_short']
        hover_text = (
            f"Config: {config}<br>"
            f"Avg Time: {row['avg_time_per_example']:.2f}s<br>"
            f"{metric_name}: {row[metric_col]:{annotation_format}}<br>"
            f"MRR: {row['mean_reciprocal_rank']:.3f}<br>"
            f"Success Rate: {row['success_rate']:.1%}"
        )

        fig.add_trace(
            go.Scatter(
                x=[row['avg_time_per_example']],
                y=[row[metric_col]],
                mode='markers',
                marker=dict(
                    color=colors_plotly[config],
                    symbol=plotly_markers[config],
                    size=14,
                    line=dict(color='white', width=1.5)
                ),
                name=config,
                legendgroup=config,
                showlegend=config not in seen_configs,
                hovertemplate=hover_text + '<extra></extra>'
            )
        )

        seen_configs.add(config)

    if len(pareto_points) > 1:
        fig.add_trace(
            go.Scatter(
                x=[x for x, _, _ in pareto_points],
                y=[y for _, y, _ in pareto_points],
                mode='lines',
                line=dict(color='black', dash='dash', width=2),
                name='Pareto Frontier'
            )
        )

    fig.update_layout(
        title=(
            f"Trade-off Analysis: {metric_name} vs. Computational Efficiency<br>"
            f"Comprehensive Retrieval Evaluation Results"
        ),
        xaxis_title='Average Time per Example (seconds)',
        yaxis_title=y_label,
        legend_title='Configuration',
        template='plotly_white',
        hovermode='closest',
        width=1000,
        height=700,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

    return fig, None

# Generate timestamp for consistent file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create three plots

# Track interactive outputs
interactive_outputs = []
interactive_figures = []

# Plot 1: MAP@10 vs Time
create_trade_off_plot(df, 'mean_average_precision', 'MAP@10', 'Mean Average Precision @ 10', f'01_map_time_tradeoff_{timestamp}.png')
fig_map, html_map = create_trade_off_plot_plotly(df, 'mean_average_precision', 'MAP@10', 'Mean Average Precision @ 10', f'01_map_time_tradeoff_{timestamp}', ".3f")
if fig_map:
    interactive_figures.append(('MAP@10', fig_map))
if html_map:
    interactive_outputs.append(('MAP@10', html_map))

# Plot 2: MRR vs Time  
create_trade_off_plot(df, 'mean_reciprocal_rank', 'MRR', 'Mean Reciprocal Rank', f'02_mrr_time_tradeoff_{timestamp}.png')
fig_mrr, html_mrr = create_trade_off_plot_plotly(df, 'mean_reciprocal_rank', 'MRR', 'Mean Reciprocal Rank', f'02_mrr_time_tradeoff_{timestamp}', ".3f")
if fig_mrr:
    interactive_figures.append(('MRR', fig_mrr))
if html_mrr:
    interactive_outputs.append(('MRR', html_mrr))

# Plot 3: Success Rate vs Time
create_trade_off_plot(df, 'success_rate', 'Success Rate', 'Success Rate', f'03_success_rate_time_tradeoff_{timestamp}.png', ".1%")
fig_success, html_success = create_trade_off_plot_plotly(df, 'success_rate', 'Success Rate', 'Success Rate', f'03_success_rate_time_tradeoff_{timestamp}', ".1%")
if fig_success:
    interactive_figures.append(('Success Rate', fig_success))
if html_success:
    interactive_outputs.append(('Success Rate', html_success))

# Combine interactive plots into a single dashboard
combined_dashboard_path = None
if PLOTLY_AVAILABLE and interactive_figures:
    dashboard_sections = []
    for idx, (metric_label, fig) in enumerate(interactive_figures):
        include_js = 'cdn' if idx == 0 else False
        snippet = pio.to_html(fig, include_plotlyjs=include_js, full_html=False, auto_play=False)
        dashboard_sections.append(
            f"<section>\n<h2>{metric_label}</h2>\n<div class=\"plot-container\">{snippet}</div>\n</section>"
        )

    dashboard_html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
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

    combined_dashboard_path = f"/mnt/nas/norapat/jan-incub-agents/attachments-agent/tradeoff_dashboard_{timestamp}.html"
    with open(combined_dashboard_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html.format(sections="\n".join(dashboard_sections)))

    print(f"Saved combined interactive dashboard to {combined_dashboard_path}")
elif not PLOTLY_AVAILABLE:
    print("\nPlotly is not installed. Skipping interactive plot generation.")
