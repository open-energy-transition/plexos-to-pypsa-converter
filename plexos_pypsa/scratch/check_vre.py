import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def read_and_concat_generator_csvs(path: str) -> pd.DataFrame:
    """
    Reads all CSV files in the given path, converts each to long format,
    adds a 'generator' column (from filename), and concatenates all into a single DataFrame.

    Parameters
    ----------
    path : str
        Path to the folder containing generator CSV files.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame in long format with columns: generator, datetime, time, value.
    """
    dfs = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            gen_name = file.replace("_RefYear4006.csv", "")
            print(f"Processing file: {file} for generator: {gen_name}")
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path)
            df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])

            # Normalize column names (e.g., 1, 2, ...48 or 01, 02, ...48)
            df.columns = pd.Index(
                [
                    str(int(col))
                    if col.strip().isdigit()
                    and col not in {"Year", "Month", "Day", "datetime"}
                    else col
                    for col in df.columns
                ]
            )

            # Identify time columns
            non_date_columns = [
                col
                for col in df.columns
                if col not in {"Year", "Month", "Day", "datetime"}
            ]
            if len(non_date_columns) == 24:
                resolution = 60  # hourly
            elif len(non_date_columns) == 48:
                resolution = 30  # 30 minutes
            else:
                raise ValueError(f"Unsupported resolution in file: {file}")

            # Convert to long format
            df_long = df.melt(
                id_vars=["datetime"],
                value_vars=non_date_columns,
                var_name="time",
                value_name="value",
            )
            if resolution == 60:
                df_long["time"] = pd.to_timedelta(
                    (df_long["time"].astype(int) - 1) * 60, unit="m"
                )
            elif resolution == 30:
                df_long["time"] = pd.to_timedelta(
                    (df_long["time"].astype(int) - 1) * 30, unit="m"
                )
            df_long["series"] = df_long["datetime"].dt.floor("D") + df_long["time"]
            df_long["generator"] = gen_name
            dfs.append(df_long)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # empty if no files found


path_test = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/Traces/solar"

dfs_data = read_and_concat_generator_csvs(path_test)

# check maximum and minimum values
max_values = dfs_data.groupby("generator")["value"].max()
min_values = dfs_data.groupby("generator")["value"].min()

# are there any values in max_values that are greater than 1?
max_values[max_values > 1]
