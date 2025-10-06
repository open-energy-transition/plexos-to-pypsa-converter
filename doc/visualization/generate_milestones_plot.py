"""Generate a milestone timeline plot."""

import pandas as pd
import plotly.express as px

data = [
    {
        "feature_id": "F001",
        "title": "Thermal gen mapping",
        "area": "mapping",
        "implemented_date": "2025-05-03",
        "owner": "alice",
        "size": 3,
        "release": "v0.3",
    },
    {
        "feature_id": "F002",
        "title": "AnnualGPG constraint",
        "area": "constraints",
        "implemented_date": "2025-05-18",
        "owner": "bob",
        "size": 5,
        "release": "v0.4",
    },
    {
        "feature_id": "F003",
        "title": "Units mapping",
        "area": "mappings",
        "implemented_date": "2025-06-01",
        "owner": "carol",
        "size": 1,
        "release": "v0.4",
    },
    {
        "feature_id": "F004",
        "title": "Topology nodes import",
        "area": "io",
        "implemented_date": "2025-06-15",
        "owner": "alice",
        "size": 2,
        "release": "v0.5",
    },
]
df = pd.DataFrame(data)
df["implemented_date"] = pd.to_datetime(df["implemented_date"])

fig = px.scatter(
    df,
    x="implemented_date",
    y=[0] * len(df),
    color="area",
    size="size",
    hover_data=["feature_id", "title", "owner", "release"],
    title="Milestone timeline — features implemented",
)
fig.update_yaxes(visible=False)  # hide the fake vertical axis
fig.update_layout(showlegend=True, height=300)
fig.show()
