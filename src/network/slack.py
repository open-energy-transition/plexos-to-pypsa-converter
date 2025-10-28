"""Add slack generators (load spillage and load shedding) to the SEM model."""

import logging

import pypsa

logger = logging.getLogger(__name__)


def add_slack_generators(network: pypsa.Network) -> dict:
    """Add load spillage and load shedding generators and carriers to the SEM model.

    Should be called after parse_outages in the workflow.
    """
    buses = network.buses.index

    summary = {}

    # Add load spillage generator and carrier
    network.add(
        "Generator",
        name=buses,
        suffix=" load spillage",
        bus=buses,
        carrier="load spillage",
        marginal_cost=-5000,
        p_min_pu=-1,
        p_max_pu=0,
        p_nom=10e6,
    )
    network.add(
        "Carrier",
        name="load spillage",
        color="#df8e23",
        nice_name="Load spillage",
    )

    # Add load shedding generator and carrier
    network.add(
        "Generator",
        name=buses,
        suffix=" load shedding",
        bus=buses,
        carrier="load shedding",
        marginal_price=5000,
        p_nom=10e6,
    )
    network.add(
        "Carrier",
        name="load shedding",
        color="#dd2e23",
        nice_name="Load shedding",
    )

    summary["slack_generators_added"] = len(buses)
    logger.info("Added slack generators for: %d buses", len(buses))

    return {"add_slack": summary}
