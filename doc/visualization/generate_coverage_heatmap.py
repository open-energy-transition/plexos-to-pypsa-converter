"""Generate Model x Feature Coverage Heatmap.

This script creates a heatmap showing the conversion status of features across
different PLEXOS models. It reads configuration from JSON files and exports both
an interactive HTML visualization and a static PNG image.

Usage:
    python doc/visualization/generate_coverage_heatmap.py

Outputs:
    - doc/visualization/html/coverage_heatmap.html (interactive)
    - doc/visualization/image/coverage_heatmap.png (static)
"""

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_data() -> tuple[dict, dict]:
    """Load model and coverage data from JSON files."""
    script_dir = Path(__file__).parent

    # Load models
    with (script_dir / "data" / "models.json").open() as f:
        models_data = json.load(f)

    # Load coverage
    with (script_dir / "data" / "coverage.json").open() as f:
        coverage_data = json.load(f)

    return models_data, coverage_data


def create_heatmap_figure(models_data: dict, coverage_data: dict) -> go.Figure:
    """Create the heatmap figure with conversion percentages."""
    # Convert to DataFrames
    models_df = pd.DataFrame(models_data["models"])
    coverage_df = pd.DataFrame(coverage_data["coverage"])
    features = coverage_data["features"]

    # Create pivot table for heatmap
    pivot = coverage_df.pivot_table(
        index="model_id", columns="feature", values="status", aggfunc="first"
    )

    # Reorder rows to match models.json order
    pivot = pivot.reindex(models_df["id"])

    # Ensure columns are in the order specified in coverage.json
    pivot = pivot[features]

    # Fill missing values with "not_applicable" as default
    pivot = pivot.fillna("not_applicable")

    # Map status to numeric values for coloring
    status_map = {
        "converted": 4,
        "partial": 3,
        "not_yet_implemented": 2,
        "feature_not_covered": 1,
        "not_applicable": 0,
    }
    pivot_numeric = pivot.map(lambda x: status_map.get(x, 0))

    # Create nice labels for tooltips
    status_labels = {
        "converted": "Fully converted",
        "partial": "Partially converted",
        "not_yet_implemented": "Not yet converted",
        "feature_not_covered": "Feature not mapped yet",
        "not_applicable": "Not applicable",
    }
    pivot_labels = pivot.map(lambda x: status_labels.get(x, x))

    # Get model names for y-axis (instead of IDs)
    model_id_to_name = dict(zip(models_df["id"], models_df["name"], strict=False))
    y_labels = [model_id_to_name[model_id] for model_id in pivot.index]

    # Get conversion percentages
    model_id_to_pct = dict(
        zip(models_df["id"], models_df["conversion_pct"], strict=False)
    )
    conversion_pcts = [model_id_to_pct[model_id] for model_id in pivot.index]

    # Create figure with subplots (heatmap + text annotations)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.92, 0.08],
        horizontal_spacing=0.01,
        specs=[[{"type": "heatmap"}, {"type": "scatter"}]],
    )

    # Add heatmap
    heatmap = go.Heatmap(
        z=pivot_numeric.values,
        x=features,
        y=y_labels,
        colorscale=[
            [0, "lightgray"],  # not_applicable
            [0.25, "red"],  # feature_not_covered
            [0.5, "orange"],  # not_yet_implemented
            [0.75, "yellow"],  # partial
            [1, "green"],  # converted
        ],
        hovertemplate="Model: %{y}<br>Feature: %{x}<br>Status: %{text}<extra></extra>",
        text=pivot_labels.values,
        showscale=False,
        xgap=2,
        ygap=2,
    )

    fig.add_trace(heatmap, row=1, col=1)

    # Add conversion percentage annotations on the right
    # Create invisible scatter plot to position text
    fig.add_trace(
        go.Scatter(
            x=[0.5] * len(y_labels),
            y=list(range(len(y_labels))),
            mode="text",
            text=[f"{pct}%" for pct in conversion_pcts],
            textposition="middle center",
            textfont={"size": 12, "color": "black"},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # Add legend traces (invisible markers for legend only)
    legend_statuses = [
        {"status": "converted", "label": "Fully converted", "color": "green"},
        {"status": "partial", "label": "Partially converted", "color": "yellow"},
        {
            "status": "not_yet_implemented",
            "label": "Not yet converted",
            "color": "orange",
        },
        {
            "status": "feature_not_covered",
            "label": "Feature not mapped yet",
            "color": "red",
        },
        {"status": "not_applicable", "label": "Not applicable", "color": "lightgray"},
    ]

    for legend_item in legend_statuses:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": legend_item["color"]},
                name=legend_item["label"],
                showlegend=True,
                legendgroup="status",
            ),
            row=1,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title={
            "text": "Model x Feature Coverage Matrix",
            "x": 0.5,
            "xanchor": "center",
        },
        height=600,
        width=1200,
        autosize=False,
        margin={"l": 250, "r": 180, "t": 80, "b": 150},
        showlegend=True,
        legend={
            "title": {"text": "Status"},
            "orientation": "v",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 1.02,
        },
    )

    # Update x-axis for heatmap
    fig.update_xaxes(tickangle=-45, side="bottom", row=1, col=1)

    # Update y-axis for heatmap
    fig.update_yaxes(tickmode="linear", row=1, col=1)

    # Update axes for percentage column
    fig.update_xaxes(
        showticklabels=False,
        title={"text": "% Converted", "standoff": 0},
        side="top",
        range=[0, 1],
        showgrid=False,
        zeroline=False,
        row=1,
        col=2,
    )

    fig.update_yaxes(
        showticklabels=False,
        range=[-0.5, len(y_labels) - 0.5],
        showgrid=False,
        zeroline=False,
        row=1,
        col=2,
    )

    # Remove background from percentage subplot
    fig.update_layout(
        {"xaxis2.showline": False, "yaxis2.showline": False},
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def export_outputs(fig: go.Figure, output_dir: Path) -> None:
    """Export both HTML and PNG versions of the figure."""
    # Export interactive HTML
    html_path = output_dir / "html" / "coverage_heatmap.html"
    fig.write_html(str(html_path))
    logging.info("Saved interactive HTML to: %s", html_path)

    # Export static PNG
    png_path = output_dir / "image" / "coverage_heatmap.png"
    try:
        fig.write_image(str(png_path), width=1000, height=600, scale=2)
        logging.info("Saved static PNG to: %s", png_path)
    except Exception as e:
        logging.warning("WARNING: Could not save PNG image: %s", e)
        logging.info("  Hint: Install kaleido with: pip install kaleido")


def main() -> None:
    """Execture main function."""
    logging.info("Generating Model x Feature Coverage Heatmap...")

    # Get script directory
    script_dir = Path(__file__).parent

    # Load data
    logging.info("Loading data from JSON files...")
    models_data, coverage_data = load_data()
    logging.info("   - Loaded %d models", len(models_data["models"]))
    logging.info("   - Loaded %d features", len(coverage_data["features"]))
    logging.info("   - Loaded %d coverage entries", len(coverage_data["coverage"]))

    # Create figure
    logging.info("Creating heatmap figure...")
    fig = create_heatmap_figure(models_data, coverage_data)

    # Export outputs
    logging.info("Exporting outputs...")
    export_outputs(fig, script_dir)

    logging.info("\nDone! Heatmap generated successfully.")
    logging.info(
        "\nView interactive version: %s", script_dir / "html" / "coverage_heatmap.html"
    )
    logging.info("View static image: %s", script_dir / "image" / "coverage_heatmap.png")


if __name__ == "__main__":
    main()
