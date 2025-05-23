import pandas as pd


def generate_timeslice_timeseries(network, timeslice_csv_path):
    # Read timeslice CSV
    df = pd.read_csv(timeslice_csv_path)
    # Assume first columns are Year, Month, Day, then timeslice columns
    date_cols = [col for col in df.columns if col.lower() in {"year", "month", "day"}]
    timeslice_cols = [col for col in df.columns if col not in date_cols]
    # Build datetime index
    df["datetime"] = pd.to_datetime(df[date_cols])
    df.set_index("datetime", inplace=True)
    # Reshape to long format
    df_long = df[timeslice_cols].stack().reset_index()
    df_long.columns = ["datetime", "timeslice", "value"]
    # Mark active/inactive periods
    df_long["active"] = False
    # For each timeslice, track active state
    for ts in df_long["timeslice"].unique():
        mask = df_long["timeslice"] == ts
        vals = df_long.loc[mask, "value"].values
        active = []
        is_active = False
        for v in vals:
            if v == -1:
                is_active = True
            elif v == 0:
                is_active = False
            active.append(is_active)
        df_long.loc[mask, "active"] = active
    # Filter only active timeslices
    df_active = df_long[df_long["active"]].copy()
    # Restrict to network snapshots
    df_active = df_active[df_active["datetime"].isin(network.snapshots)]
    # Output: columns = datetime, timeslice, active (True)
    return df_active[["datetime", "timeslice", "active"]]


# Example usage:
# import pypsa
# network = pypsa.Network("your_network_file.nc")
# df_timeslice = generate_timeslice_timeseries(network, "timeslice_RefYear4006.csv")
# print(df_timeslice)
