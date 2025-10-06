"""Demo Dashboard for PLEXOS-to-PyPSA Converter Progress Visualization.

This is a minimal self-contained demo showing all proposed visualizations
with realistic sample data.

To run:
    python src/visualization/demo_dashboard.py

Then open: http://localhost:8050
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, dcc, html


def generate_sample_data() -> dict:
    """Generate realistic sample data for demo dashboard."""
    # Model definitions from models.py
    models = [
        {
            "id": "aemo-2024-green",
            "name": "AEMO 2024 ISP - Green Energy Exports",
            "source": "AEMO",
            "status": "not_started",
        },
        {
            "id": "aemo-2024-isp-progressive",
            "name": "AEMO 2024 ISP - Progressive Change",
            "source": "AEMO",
            "status": "completed",
        },
        {
            "id": "aemo-2024-isp-step",
            "name": "AEMO 2024 ISP - Step Change",
            "source": "AEMO",
            "status": "not_started",
        },
        {
            "id": "caiso-irp23",
            "name": "CAISO IRP 2023 Stochastic",
            "source": "CAISO",
            "status": "in_progress",
        },
        {
            "id": "caiso-sa25",
            "name": "CAISO 2025 Summer Assessment",
            "source": "CAISO",
            "status": "not_started",
        },
        {
            "id": "nrel-118",
            "name": "NREL Extended IEEE 118-bus",
            "source": "NREL",
            "status": "not_started",
        },
        {
            "id": "sem-2024-2032",
            "name": "SEM 2024-2032 Validation Model",
            "source": "SEM",
            "status": "in_progress",
        },
        {
            "id": "marei-eu",
            "name": "European Power & Gas Model",
            "source": "UCC",
            "status": "in_progress",
        },
        {
            "id": "plexos-world-2015",
            "name": "PLEXOS-World 2015 Gold V1.1",
            "source": "UCC",
            "status": "not_started",
        },
        {
            "id": "plexos-world-spatial",
            "name": "PLEXOS-World Spatial Resolution",
            "source": "UCC",
            "status": "not_started",
        },
        {
            "id": "plexos-message",
            "name": "MESSAGEix-GLOBIOM",
            "source": "UCC",
            "status": "completed",
        },
    ]

    # PLEXOS feature hierarchy with mapping metadata
    plexos_features = {
        "Generator": {
            "Capacity": [
                {
                    "plexos_attr": "Max Capacity",
                    "pypsa_attr": "p_nom",
                    "mapping_pct": 100,
                    "complexity": 8,
                },
                {
                    "plexos_attr": "Rating Factor",
                    "pypsa_attr": "p_max_pu (time series)",
                    "mapping_pct": 90,
                    "complexity": 6,
                },
            ],
            "Efficiency": [
                {
                    "plexos_attr": "Heat Rate",
                    "pypsa_attr": "efficiency (calculated)",
                    "mapping_pct": 85,
                    "complexity": 9,
                },
                {
                    "plexos_attr": "Incr Heat Rate",
                    "pypsa_attr": "efficiency (multi-band approximation)",
                    "mapping_pct": 60,
                    "complexity": 10,
                },
            ],
            "Costs": [
                {
                    "plexos_attr": "Fuel Price",
                    "pypsa_attr": "marginal_cost (via Fuel link)",
                    "mapping_pct": 50,
                    "complexity": 7,
                },
                {
                    "plexos_attr": "VO&M Charge",
                    "pypsa_attr": "marginal_cost (additive)",
                    "mapping_pct": 100,
                    "complexity": 5,
                },
                {
                    "plexos_attr": "Capital Cost",
                    "pypsa_attr": "capital_cost",
                    "mapping_pct": 95,
                    "complexity": 6,
                },
            ],
        },
        "Node": {
            "Basic": [
                {
                    "plexos_attr": "Region",
                    "pypsa_attr": "Bus (name)",
                    "mapping_pct": 100,
                    "complexity": 3,
                },
                {
                    "plexos_attr": "Voltage",
                    "pypsa_attr": "v_nom",
                    "mapping_pct": 100,
                    "complexity": 2,
                },
            ],
        },
        "Line": {
            "Impedance": [
                {
                    "plexos_attr": "Reactance",
                    "pypsa_attr": "x",
                    "mapping_pct": 100,
                    "complexity": 4,
                },
                {
                    "plexos_attr": "Resistance",
                    "pypsa_attr": "r",
                    "mapping_pct": 100,
                    "complexity": 4,
                },
            ],
            "Limits": [
                {
                    "plexos_attr": "Max Flow",
                    "pypsa_attr": "s_nom",
                    "mapping_pct": 100,
                    "complexity": 5,
                },
                {
                    "plexos_attr": "Rating Factor",
                    "pypsa_attr": "s_max_pu (time series)",
                    "mapping_pct": 75,
                    "complexity": 7,
                },
            ],
        },
        "Storage": {
            "Energy": [
                {
                    "plexos_attr": "Max Volume",
                    "pypsa_attr": "p_nom * max_hours",
                    "mapping_pct": 90,
                    "complexity": 6,
                },
            ],
            "Efficiency": [
                {
                    "plexos_attr": "Pump Efficiency",
                    "pypsa_attr": "efficiency_store",
                    "mapping_pct": 100,
                    "complexity": 5,
                },
                {
                    "plexos_attr": "Generate Efficiency",
                    "pypsa_attr": "efficiency_dispatch",
                    "mapping_pct": 100,
                    "complexity": 5,
                },
            ],
        },
    }

    # Flatten PLEXOS features
    flat_features = []
    for component, categories in plexos_features.items():
        for category, feature_list in categories.items():
            flat_features.extend(
                [
                    {
                        "component": component,
                        "category": category,
                        "plexos_attr": feature["plexos_attr"],
                        "pypsa_attr": feature["pypsa_attr"],
                        "mapping_pct": feature["mapping_pct"],
                        "complexity": feature["complexity"],
                        "path": f"PLEXOS Features/{component}/{category}/{feature['plexos_attr']}",
                    }
                    for feature in feature_list
                ]
            )

    # Generate coverage matrix (models × features) - still needed for heatmap
    coverage_data = []
    rng = np.random.default_rng()
    for model in models:
        for feature in flat_features:
            # Assign realistic status based on model status
            if model["status"] == "not_started":
                status = rng.choice(
                    ["missing", "missing", "missing", "planned"], p=[0.7, 0.1, 0.1, 0.1]
                )
            elif model["status"] == "in_progress":
                status = rng.choice(
                    ["converted", "partial", "missing", "failed"],
                    p=[0.5, 0.2, 0.2, 0.1],
                )
            else:  # completed
                status = rng.choice(
                    ["converted", "partial", "failed"], p=[0.8, 0.15, 0.05]
                )

            coverage_data.append(
                {
                    "model_id": model["id"],
                    "model_name": model["name"],
                    "component": feature["component"],
                    "category": feature["category"],
                    "plexos_attr": feature["plexos_attr"],
                    "pypsa_attr": feature["pypsa_attr"],
                    "path": feature["path"],
                    "status": status,
                    "mapping_pct": feature["mapping_pct"],
                    "complexity": feature["complexity"],
                }
            )

    # Load timeline from JSON
    json_path = Path(__file__).parent / "data" / "model_history.json"
    with json_path.open() as f:
        timeline_data = json.load(f)

    # Convert ISO date strings to datetime objects
    timeline_events = [
        {
            **event,
            "date": datetime.fromisoformat(event["date"]),
        }
        for event in timeline_data["events"]
    ]

    return {
        "models": pd.DataFrame(models),
        "features": pd.DataFrame(flat_features),
        "coverage": pd.DataFrame(coverage_data),
        "timeline": pd.DataFrame(timeline_events),
    }


def calculate_kpis(data: dict) -> dict:
    """Calculate KPI metrics from data."""
    features = data["features"]
    models = data["models"]

    # Count unique PLEXOS attributes
    total_features = len(features)

    # Get list of unique components with features
    components_list = features["component"].unique().tolist()
    converted_components_str = ", ".join(sorted(components_list))

    models_processed = len(models[models["status"] != "not_started"])
    models_completed = len(models[models["status"] == "completed"])
    models_in_progress = len(models[models["status"] == "in_progress"])

    return {
        "total_features": total_features,
        "features_converted": total_features,  # All features defined
        "converted_components": converted_components_str,
        "models_total": len(models),
        "models_processed": models_processed,
        "models_completed": models_completed,
        "models_in_progress": models_in_progress,
    }


# ============================================================================
# CHART CREATION FUNCTIONS
# ============================================================================


def create_kpi_cards(kpis: dict) -> dbc.Row:
    """Create KPI cards showing key metrics."""
    cards = [
        {
            "title": "Features Converted",
            "value": str(kpis["features_converted"]),
            "subtitle": kpis["converted_components"]
            if kpis["converted_components"]
            else "None yet",
            "icon": "",
            "color": "success" if kpis["features_converted"] > 10 else "warning",
        },
        {
            "title": "Model Coverage",
            "value": f"{kpis['models_processed']}/{kpis['models_total']}",
            "subtitle": f"{kpis['models_completed']} completed, {kpis['models_in_progress']} in progress",
            "icon": "",
            "color": "primary",
        },
    ]

    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H6(
                                        f"{card['icon']} {card['title']}",
                                        className="text-muted",
                                    ),
                                    html.H2(card["value"], className="mb-0"),
                                    html.P(
                                        card["subtitle"], className="text-muted mb-0"
                                    ),
                                ]
                            )
                        ],
                        color=card["color"],
                        outline=True,
                    )
                ],
                width=6,
            )
            for card in cards
        ],
        className="mb-4",
    )


def create_treemap(data: dict) -> go.Figure:
    """Create hierarchical treemap of PLEXOS feature mapping."""
    features = data["features"]

    # Use features directly (no aggregation needed since each feature is unique)
    treemap_data = features.copy()

    # Add parent levels (categories and components)
    parent_data = []

    # Add category-level parents
    for comp in treemap_data["component"].unique():
        for cat in treemap_data[treemap_data["component"] == comp]["category"].unique():
            subset = treemap_data[
                (treemap_data["component"] == comp) & (treemap_data["category"] == cat)
            ]
            parent_data.append(
                {
                    "component": comp,
                    "category": cat,
                    "plexos_attr": "",
                    "pypsa_attr": "",
                    "path": f"PLEXOS Features/{comp}/{cat}",
                    "mapping_pct": subset["mapping_pct"].mean(),
                    "complexity": subset["complexity"].sum(),
                }
            )

    # Add component-level parents
    for comp in treemap_data["component"].unique():
        subset = treemap_data[treemap_data["component"] == comp]
        parent_data.append(
            {
                "component": comp,
                "category": "",
                "plexos_attr": "",
                "pypsa_attr": "",
                "path": f"PLEXOS Features/{comp}",
                "mapping_pct": subset["mapping_pct"].mean(),
                "complexity": subset["complexity"].sum(),
            }
        )

    # Add root parent
    parent_data.append(
        {
            "component": "",
            "category": "",
            "plexos_attr": "",
            "pypsa_attr": "",
            "path": "PLEXOS Features",
            "mapping_pct": treemap_data["mapping_pct"].mean(),
            "complexity": treemap_data["complexity"].sum(),
        }
    )

    all_data = pd.concat([treemap_data, pd.DataFrame(parent_data)], ignore_index=True)

    # Create labels (PLEXOS attribute names)
    all_data["label"] = all_data.apply(
        lambda x: x["plexos_attr"]
        if x["plexos_attr"]
        else (x["category"] if x["category"] else x["component"]),
        axis=1,
    )
    all_data["label"] = all_data["label"].fillna("PLEXOS Features")

    # Prepare customdata for hover (pypsa_attr, mapping_pct, complexity)
    customdata = []
    for _, row in all_data.iterrows():
        pypsa = row.get("pypsa_attr", "")
        mapping = row.get("mapping_pct", 0)
        complexity = row.get("complexity", 0)
        customdata.append([pypsa, mapping, complexity])

    fig = go.Figure(
        go.Treemap(
            ids=all_data["path"],
            labels=all_data["label"],
            parents=[
                "/".join(p.split("/")[:-1]) if "/" in p else ""
                for p in all_data["path"]
            ],
            values=all_data["complexity"],
            branchvalues="total",
            marker={
                "colors": all_data["mapping_pct"],
                "colorscale": "RdYlGn",
                "cmin": 0,
                "cmax": 100,
                "colorbar": {"title": "% Mapped to PyPSA"},
                "line": {"width": 2},
            },
            hovertemplate=(
                "<b>PLEXOS: %{label}</b><br>"
                "→ PyPSA: %{customdata[0]}<br>"
                "Mapping: %{customdata[1]:.0f}% complete<br>"
                "Complexity: %{customdata[2]:.0f}/10"
                "<extra></extra>"
            ),
            customdata=customdata,
            text=all_data["mapping_pct"].round(0).astype(str) + "%",
            textposition="middle center",
        )
    )

    fig.update_layout(
        title="PLEXOS Feature Mapping Hierarchy",
        height=500,
        autosize=False,
        margin={"t": 50, "l": 0, "r": 0, "b": 0},
    )

    return fig


def create_sankey(data: dict) -> go.Figure:
    """Create Sankey diagram showing PLEXOS → PyPSA mapping flow."""
    coverage = data["coverage"]

    # Aggregate flows: Component → Status → PyPSA Component
    flow_data = (
        coverage.groupby(["component", "status"]).size().reset_index(name="count")
    )

    # Create nodes
    components = coverage["component"].unique().tolist()
    statuses = coverage["status"].unique().tolist()  # Dynamic status list

    nodes = components + statuses

    # Create color mapping for statuses
    status_colors = {
        "converted": "green",
        "partial": "yellow",
        "failed": "red",
        "missing": "lightgray",
        "planned": "lightblue",
    }

    node_colors = ["lightblue"] * len(components) + [
        status_colors.get(status, "gray") for status in statuses
    ]

    # Create links
    links = []
    for _, row in flow_data.iterrows():
        source_idx = nodes.index(row["component"])
        target_idx = nodes.index(row["status"])
        links.append(
            {
                "source": source_idx,
                "target": target_idx,
                "value": row["count"],
            }
        )

    # Link colors based on status
    link_colors = []
    link_color_map = {
        "converted": "rgba(0, 255, 0, 0.3)",
        "partial": "rgba(255, 255, 0, 0.3)",
        "failed": "rgba(255, 0, 0, 0.3)",
        "missing": "rgba(200, 200, 200, 0.3)",
        "planned": "rgba(173, 216, 230, 0.3)",
    }

    for link in links:
        target = nodes[link["target"]]
        link_colors.append(link_color_map.get(target, "rgba(200, 200, 200, 0.3)"))

    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 15,
                    "thickness": 20,
                    "line": {"color": "black", "width": 0.5},
                    "label": nodes,
                    "color": node_colors,
                },
                link={
                    "source": [l["source"] for l in links],
                    "target": [l["target"] for l in links],
                    "value": [l["value"] for l in links],
                    "color": link_colors,
                },
            )
        ]
    )

    fig.update_layout(
        title="PLEXOS Component → Conversion Status Flow",
        height=500,
        autosize=False,
        margin={"t": 50, "l": 0, "r": 0, "b": 0},
    )

    return fig


def create_heatmap(data: dict) -> go.Figure:
    """Create models × features coverage heatmap."""
    coverage = data["coverage"]

    # Create pivot table
    pivot = coverage.pivot_table(
        index="model_name", columns="path", values="status", aggfunc="first"
    )

    # Map status to numeric values for color
    status_map = {"converted": 3, "partial": 2, "failed": 1, "missing": 0, "planned": 0}
    pivot_numeric = pivot.map(lambda x: status_map.get(x, 0))

    # Simplify column names (just show attribute)
    simplified_cols = [col.split("/")[-1] for col in pivot.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_numeric.values,
            x=simplified_cols,
            y=pivot.index,
            colorscale=[
                [0, "lightgray"],  # missing
                [0.33, "red"],  # failed
                [0.66, "yellow"],  # partial
                [1, "green"],  # converted
            ],
            hovertemplate="Model: %{y}<br>Feature: %{x}<br>Status: %{text}<extra></extra>",
            text=pivot.values,
            showscale=False,
        )
    )

    fig.update_layout(
        title="Model × Feature Coverage Matrix",
        height=600,
        autosize=False,
        xaxis={"tickangle": -45},
        yaxis={"tickmode": "linear"},
        margin={"l": 200, "r": 50, "t": 80, "b": 150},
    )

    return fig


def create_timeline(data: dict) -> go.Figure:
    """Create cumulative models processed over time."""
    timeline = data["timeline"].sort_values("date")

    # Calculate cumulative counts
    cumulative = []
    current_count = 0

    # Use timeline max date + buffer instead of datetime.now()
    end_date = timeline["date"].max() + timedelta(days=7)

    for date in pd.date_range(timeline["date"].min(), end_date, freq="D"):
        events_today = timeline[timeline["date"].dt.date == date.date()]
        current_count += len(events_today[events_today["event"] == "completed"])
        cumulative.append({"date": date, "models_completed": current_count})

    cumulative_df = pd.DataFrame(cumulative)

    fig = go.Figure()

    # Area chart
    fig.add_trace(
        go.Scatter(
            x=cumulative_df["date"],
            y=cumulative_df["models_completed"],
            mode="lines",
            fill="tozeroy",
            name="Completed Models",
            line={"color": "green", "width": 2},
        )
    )

    # Add markers for model completions
    completed_events = timeline[timeline["event"] == "completed"]
    if not completed_events.empty:
        # Calculate correct cumulative count for each completion
        completion_counts = []
        for _, event in completed_events.iterrows():
            # Find the cumulative count at this date
            matching_rows = cumulative_df[
                cumulative_df["date"].dt.date == event["date"].date()
            ]
            if not matching_rows.empty:
                count = matching_rows["models_completed"].iloc[0]
                completion_counts.append(count)
            else:
                completion_counts.append(0)

        fig.add_trace(
            go.Scatter(
                x=completed_events["date"],
                y=completion_counts,
                mode="markers",
                name="Milestones",
                marker={"size": 10, "color": "darkgreen", "symbol": "star"},
                hovertemplate="%{text}<extra></extra>",
                text=completed_events["model_name"],
            )
        )

    # Set explicit y-axis range based on actual data
    max_completed = cumulative_df["models_completed"].max()
    y_max = max(max_completed + 1, 5)  # At least show 0-5 range

    fig.update_layout(
        title="Cumulative Models Processed Over Time",
        xaxis_title="Date",
        yaxis_title="Models Completed",
        yaxis={"range": [0, y_max]},
        height=400,
        autosize=False,
        hovermode="x unified",
    )

    return fig


def create_status_bar(data: dict) -> go.Figure:
    """Create bar chart of model statuses."""
    models = data["models"]

    status_counts = models.groupby("status").size().reset_index(name="count")
    status_map = {
        "not_started": "Not Started",
        "in_progress": "In Progress",
        "completed": "Completed",
    }
    status_counts["status_label"] = status_counts["status"].map(status_map)

    color_map = {
        "Not Started": "lightgray",
        "In Progress": "orange",
        "Completed": "green",
    }

    fig = px.bar(
        status_counts,
        x="status_label",
        y="count",
        color="status_label",
        color_discrete_map=color_map,
        title="Model Conversion Status",
        labels={"status_label": "Status", "count": "Number of Models"},
    )

    fig.update_layout(
        showlegend=False,
        height=400,
        autosize=False,
    )

    return fig


# ============================================================================
# DASH APP
# ============================================================================


def create_app() -> Dash:
    """Create and configure Dash app."""
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Generate data
    data = generate_sample_data()
    kpis = calculate_kpis(data)

    # Create figures
    kpi_cards = create_kpi_cards(kpis)
    treemap_fig = create_treemap(data)
    sankey_fig = create_sankey(data)
    heatmap_fig = create_heatmap(data)
    timeline_fig = create_timeline(data)
    status_bar_fig = create_status_bar(data)

    # Layout
    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1(
                                "PLEXOS-to-PyPSA Converter Progress Dashboard",
                                className="mb-2",
                            ),
                            html.P(
                                "Visualization of converter's progress in mapping features and converting models",
                                className="text-muted mb-4",
                            ),
                        ],
                        width=12,
                    )
                ]
            ),
            kpi_cards,
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Feature Conversion", className="mb-3"),
                            dcc.Graph(figure=treemap_fig, config={"responsive": False}),
                            html.Hr(),
                            dcc.Graph(figure=sankey_fig, config={"responsive": False}),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H3("Model Conversion", className="mb-3"),
                            dcc.Graph(
                                figure=timeline_fig, config={"responsive": False}
                            ),
                            html.Hr(),
                            dcc.Graph(
                                figure=status_bar_fig, config={"responsive": False}
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3(
                                "🎯 Combined View: Model × Feature Coverage",
                                className="mb-3 mt-4",
                            ),
                            dcc.Graph(figure=heatmap_fig, config={"responsive": False}),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Hr(),
                            html.P(
                                [
                                    "Demo Dashboard | ",
                                    html.A(
                                        "PLEXOS-to-PyPSA Converter",
                                        href="https://github.com/open-energy-transition/plexos-to-pypsa-converter",
                                    ),
                                ],
                                className="text-muted text-center",
                            ),
                        ],
                        width=12,
                    )
                ]
            ),
        ],
        fluid=True,
        className="p-4",
    )

    return app


def export_html(output_path: str = "docs/progress_dashboard.html") -> None:
    """Export dashboard as standalone HTML file."""
    data = generate_sample_data()
    kpis = calculate_kpis(data)

    figures = {
        "treemap": create_treemap(data),
        "sankey": create_sankey(data),
        "heatmap": create_heatmap(data),
        "timeline": create_timeline(data),
        "status": create_status_bar(data),
    }

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PLEXOS-to-PyPSA Progress Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .kpi {{ display: inline-block; margin: 10px; padding: 20px;
                    border: 2px solid #ddd; border-radius: 8px; min-width: 200px; }}
            .chart {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>📊 PLEXOS-to-PyPSA Converter Progress Dashboard</h1>

        <div class="kpis">
            <div class="kpi">
                <h3>Feature Completeness</h3>
                <p style="font-size: 24px;">{kpis["features_converted"]}/{kpis["total_features"]}</p>
                <p>{kpis["pct_complete"]:.1f}% complete</p>
            </div>
            <div class="kpi">
                <h3>Model Coverage</h3>
                <p style="font-size: 24px;">{kpis["models_processed"]}/{kpis["models_total"]}</p>
                <p>{kpis["models_completed"]} completed</p>
            </div>
            <div class="kpi">
                <h3>Success Rate</h3>
                <p style="font-size: 24px;">{kpis["success_rate"]:.0f}%</p>
                <p>conversion quality</p>
            </div>
        </div>

        <h2>Features Converted</h2>
        <div class="chart" id="treemap"></div>
        <div class="chart" id="sankey"></div>

        <h2>Models Converted</h2>
        <div class="chart" id="timeline"></div>
        <div class="chart" id="status"></div>

        <h2>Combined Coverage</h2>
        <div class="chart" id="heatmap"></div>

        <script>
            Plotly.newPlot('treemap', {pio.to_json(figures["treemap"])["data"]},
                          {pio.to_json(figures["treemap"])["layout"]});
            Plotly.newPlot('sankey', {pio.to_json(figures["sankey"])["data"]},
                          {pio.to_json(figures["sankey"])["layout"]});
            Plotly.newPlot('heatmap', {pio.to_json(figures["heatmap"])["data"]},
                          {pio.to_json(figures["heatmap"])["layout"]});
            Plotly.newPlot('timeline', {pio.to_json(figures["timeline"])["data"]},
                          {pio.to_json(figures["timeline"])["layout"]});
            Plotly.newPlot('status', {pio.to_json(figures["status"])["data"]},
                          {pio.to_json(figures["status"])["layout"]});
        </script>
    </body>
    </html>
    """

    with Path(output_path).open("w") as f:
        f.write(html_content)

    print(f"✓ Dashboard exported to {output_path}")


if __name__ == "__main__":
    print("🚀 Starting PLEXOS-to-PyPSA Progress Dashboard...")
    print("📊 Generating sample data...")

    app = create_app()

    print("✓ Dashboard ready!")
    print("🌐 Open your browser to: http://localhost:8050")
    print("⚠️  Press Ctrl+C to stop the server")
    print()

    app.run(debug=True, port=8050)
