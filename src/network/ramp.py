"""Functions to fix ramp rate conflicts after outage scheduling in a PyPSA network."""

import numpy as np
import pandas as pd
import pypsa


def fix_outage_ramp_conflicts(network: pypsa.Network) -> dict:
    summary = {}
    for gen in network.generators.index:
        p_max = network.generators_t.p_max_pu[gen]
        p_min = network.generators_t.p_min_pu[gen]
        ramp_rate = network.generators.at[gen, "ramp_limit_up"]
        static_p_max = network.generators.p_max_pu[gen]

        # Find where outage ends (recovery: p_max goes from 0 to >0)
        outage_ends = (p_max.shift(1) == 0) & (p_max > 0)
        # Find where outage starts (p_max goes from >0 to 0)
        outage_starts = (p_max.shift(1) > 0) & (p_max == 0)

        # If ramp_rate is NaN, make it 1E+30 (PLEXOS default)
        if pd.isna(ramp_rate):
            ramp_rate = 1e30

        # Handle ramp up after outage ends
        for recovery_start in outage_ends[outage_ends].index:
            target_p_min = p_min.loc[recovery_start]
            if target_p_min > 0:
                recovery_hours = int(np.ceil(target_p_min / ramp_rate))
                for h in range(recovery_hours):
                    timestamp = recovery_start + pd.Timedelta(hours=h)
                    # Only adjust if not in actual outage (p_max == 0)
                    if timestamp in p_min.index and p_max.loc[timestamp] > 0:
                        p_min.loc[timestamp] = min(h * ramp_rate, target_p_min)
                        p_max.loc[timestamp] = static_p_max

        # Handle ramp down before outage starts
        for outage_start in outage_starts[outage_starts].index:
            initial_p_min = p_min.loc[outage_start - pd.Timedelta(hours=1)]
            if initial_p_min > 0:
                ramp_down_hours = int(np.ceil(initial_p_min / ramp_rate))
                for h in range(ramp_down_hours):
                    timestamp = outage_start - pd.Timedelta(hours=h)
                    # Only adjust if not in actual outage (p_max == 0)
                    if timestamp in p_min.index and p_max.loc[timestamp] > 0:
                        p_min.loc[timestamp] = max(initial_p_min - h * ramp_rate, 0)
                        p_max.loc[timestamp] = static_p_max

        # Resolve any remaining conflicts where p_min_pu > p_max_pu
        merged_p = p_min.reset_index().merge(
            p_max.reset_index(),
            on="snapshot",
            suffixes=("_min", "_max"),
        )
        conflicts = merged_p[merged_p[f"{gen}_min"] > merged_p[f"{gen}_max"]]

        for idx in conflicts.index:
            timestamp = merged_p.loc[idx, "snapshot"]
            p_max.loc[timestamp] = p_min.loc[timestamp]

    summary["message"] = "Ramp rate conflicts adjusted."

    return {"fix_outage_ramps": summary}
