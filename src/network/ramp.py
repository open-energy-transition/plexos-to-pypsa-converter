"""Functions to fix ramp rate conflicts after outage scheduling in a PyPSA network.

DEPRECATED: This module is deprecated. Ramp conflicts are now handled automatically
by apply_outage_schedule() with ramp_aware=True (enabled by default).

See migration guide below for how to update your code.
"""

import warnings

import pypsa


def fix_outage_ramp_conflicts(network: pypsa.Network) -> dict:
    """Fix ramp rate conflicts after outage scheduling.

    DEPRECATED: This function is no longer needed.

    Ramp conflicts are now handled automatically by apply_outage_schedule()
    with ramp_aware=True (enabled by default). The new implementation:
    - Creates gradual startup/shutdown zones for generators with binding ramp constraints
    - Prevents p_min > p_max violations automatically
    - Works for ALL generators (no need to filter by ramp rate)
    - Eliminates the need for post-processing

    Migration
    ---------
    **Old code (with workarounds):**
    ```python
    # Had to filter out low-ramp generators
    events = parse_outages(
        csv_dir,
        network,
        generator_filter=lambda g: network.generators.at[g, 'ramp_limit_up'] >= 0.05
    )
    schedule = build_outage_schedule(events, network.snapshots)
    apply_outage_schedule(network, schedule)
    fix_outage_ramp_conflicts(network)  # Post-processing needed
    ```

    **New code (no workarounds needed):**
    ```python
    # Works for ALL generators automatically
    events = parse_outages(csv_dir, network)
    schedule = build_outage_schedule(events, network.snapshots)
    apply_outage_schedule(network, schedule, ramp_aware=True)  # That's it!
    ```

    Returns:
    -------
    dict
        Empty summary indicating deprecated status

    Warnings:
    --------
    This function will be removed in a future version. Please migrate to using
    apply_outage_schedule(ramp_aware=True) as shown above.

    See Also:
    --------
    network.outages.apply_outage_schedule : New ramp-aware outage application
    """
    warnings.warn(
        "fix_outage_ramp_conflicts() is deprecated and will be removed in a future version. "
        "Ramp conflicts are now handled automatically by apply_outage_schedule(ramp_aware=True). "
        "See the function docstring for migration instructions.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Return empty summary - no action taken
    return {"fix_outage_ramps": {"message": "Deprecated - no action taken"}}
