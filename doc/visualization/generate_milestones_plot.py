"""Generate Feature Conversion Milestones Timeline.

This script creates a timeline visualization showing major feature conversion
milestones for the PLEXOS-to-PyPSA converter. It reads milestone data from a
JSON file and exports both an interactive HTML visualization and a static PNG image.

Usage:
    python doc/visualization/generate_milestones_plot.py

Outputs:
    - doc/visualization/html/milestones.html (interactive)
    - doc/visualization/image/milestones.png (static)
"""

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_data() -> dict:
    """Load milestone data from JSON file."""
    script_dir = Path(__file__).parent
    with (script_dir / "data" / "milestones.json").open() as f:
        return json.load(f)


def create_figure(milestones_data: dict) -> go.Figure:
    """Create the milestones timeline figure."""
    # Convert to DataFrame and parse dates
    df = pd.DataFrame(milestones_data["milestones"])
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Define category colors
    category_colors = {
        "core": "#1f77b4",  # Blue
        "generation": "#2ca02c",  # Green
        "network": "#ff7f0e",  # Orange
        "storage": "#9467bd",  # Purple
        "multi_sector": "#17becf",  # Teal
        "constraints": "#d62728",  # Red
        "costs": "#8c564b",  # Brown
    }

    # Create figure
    fig = go.Figure()

    # Add thin grey background bar
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_range = (max_date - min_date).days
    padding_days = max(date_range * 0.05, 5)  # 5% padding or 5 days minimum

    fig.add_shape(
        type="rect",
        x0=min_date - pd.Timedelta(days=padding_days),
        x1=max_date + pd.Timedelta(days=padding_days),
        y0=-0.05,
        y1=0.05,
        fillcolor="lightgray",
        line={"width": 0},
        layer="below",
    )

    # Alternate label positions (above/below)
    for idx, (i, row) in enumerate(df.iterrows()):
        color = category_colors.get(row["category"], "gray")
        position = "top" if idx % 2 == 0 else "bottom"
        y_circle = 0
        y_label = 0.3 if position == "top" else -0.3

        # Add circle marker
        fig.add_trace(
            go.Scatter(
                x=[row["date"]],
                y=[y_circle],
                mode="markers",
                marker={
                    "size": 20,
                    "color": color,
                    "line": {"width": 2, "color": "white"},
                },
                name=row["category"].replace("_", " ").title(),
                legendgroup=row["category"],
                showlegend=bool(i == df[df["category"] == row["category"]].index[0]),
                hovertemplate=(
                    f"<b>{row['title']}</b><br>"
                    f"{row['description']}<br>"
                    f"Date: {row['date'].strftime('%Y-%m-%d')}"
                    "<extra></extra>"
                ),
            )
        )

        # Add connecting line from circle to label
        fig.add_shape(
            type="line",
            x0=row["date"],
            y0=y_circle,
            x1=row["date"],
            y1=y_label * 0.8,
            line={"color": color, "width": 1, "dash": "dot"},
            layer="below",
        )

        # Add label text
        fig.add_annotation(
            x=row["date"],
            y=y_label,
            text=f"<b>{row['title']}</b><br>{row['description']}",
            showarrow=False,
            font={"size": 10, "color": color},
            bgcolor="white",
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
            align="center",
            xanchor="center",
            yanchor="bottom" if position == "top" else "top",
        )

    # Update layout
    fig.update_layout(
        title={
            "text": "Feature Conversion Milestones",
            "x": 0.5,
            "xanchor": "center",
        },
        height=400,
        width=1200,
        autosize=False,
        showlegend=True,
        legend={
            "title": "Category",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        xaxis={
            "title": "Date",
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "lightgray",
        },
        yaxis={
            "visible": False,
            "range": [-0.5, 0.5],
            "fixedrange": True,
        },
        margin={"l": 50, "r": 50, "t": 100, "b": 80},
        hovermode="closest",
        plot_bgcolor="white",
    )

    return fig


def export_outputs(fig: go.Figure, output_dir: Path) -> None:
    """Export both HTML and PNG versions of the figure."""
    # Ensure output directories exist
    (output_dir / "html").mkdir(exist_ok=True)
    (output_dir / "image").mkdir(exist_ok=True)

    # Export interactive HTML
    html_path = output_dir / "html" / "milestones.html"
    fig.write_html(str(html_path))
    logging.info("Saved interactive HTML to: %s", html_path)

    # Export static PNG
    png_path = output_dir / "image" / "milestones.png"
    try:
        fig.write_image(str(png_path), width=1200, height=400, scale=2)
        logging.info("Saved static PNG to: %s", png_path)
    except Exception as e:
        logging.warning("WARNING: Could not save PNG image: %s", e)
        logging.info("  Hint: Install kaleido with: pip install kaleido")


def main() -> None:
    """Execute main function."""
    logging.info("Generating Feature Conversion Milestones Timeline...")

    # Get script directory
    script_dir = Path(__file__).parent

    # Load data
    logging.info("Loading milestone data from JSON...")
    milestones_data = load_data()
    logging.info("   - Loaded %d milestones", len(milestones_data["milestones"]))

    # Create figure
    logging.info("Creating timeline figure...")
    fig = create_figure(milestones_data)

    # Export outputs
    logging.info("Exporting outputs...")
    export_outputs(fig, script_dir)

    logging.info("\\nDone! Milestones timeline generated successfully.")
    logging.info(
        "\\nView interactive version: %s", script_dir / "html" / "milestones.html"
    )
    logging.info("View static image: %s", script_dir / "image" / "milestones.png")


if __name__ == "__main__":
    main()
