import logging
from statistics import mean

from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum
from pypsa import Network

# Import enhanced constraint porting system
from network.constraint_porting import port_plexos_constraints

logger = logging.getLogger(__name__)


def add_constraints(network: Network, db: PlexosDB) -> None:
    """Add constraints from the PLEXOS database to the PyPSA network.

    The function infers the possible kinds of constraints, based on the constraint name, properties, and memberships:

    1. Emissions Budgets:
        - Identified by the presence of Production Coefficient and RHS Custom.
        - These constraints limit emissions and can be mapped to pypsa.global_constraints.

    2. Line Flow Limits:
        - Identified by the presence of Flow Forward Coefficient or Flow Back Coefficient.
        - These constraints limit the flow on specific transmission lines.

    3. Generation Limits:
        - Identified by the presence of Generation Coefficient or Installed Capacity Coefficient.
        - These constraints limit the generation capacity of specific generators.

    4. Storage Constraints:
        - Identified by properties like Units Built in Year Coefficient or RHS Year.
        - These constraints limit storage capacity or usage.

    5. Other Constraints:
        - Constraints that don't fit into the above categories can be logged for manual review.

    The constraints are mapped to the PyPSA network as follows:
    - Emissions Budgets: Added as global constraints (network.global_constraints).
    - Line Flow Limits: Applied to specific lines (network.lines).
    - Generation Limits: Applied to specific generators (network.generators).
    - Storage Constraints: Applied to specific storage units (network.storage_units).

    For each constraint, the function checks for all RHS-related properties (RHS Custom, RHS Year, RHS Day, etc.).
    If multiple RHS values exist (e.g., for different time periods),
    the function aggregates or processes them appropriately (currently taking the average).

    Parameters
    ----------
    network : pysa.Network
        The PyPSA network to which constraints will be added.
    db : PlexosDB
        The PLEXOS database containing constraint data.

    Notes
    -----
    - The function retrieves all constraints from the database and their properties.
    - It infers the type of constraint based on the properties and memberships.
    - The function applies the constraints to the network based on their inferred type.
    - The function handles various types of constraints, including emissions budgets,
        line flow limits, generation limits, storage constraints, and capacity constraints.
    - The function logs the application of each constraint and any errors encountered.
    - The function uses the mean of RHS values when multiple are found.
    - The function handles exceptions and logs errors for constraints that fail to apply.
    - The function uses the logger to provide detailed information about the process.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> add_constraints(network, db)
    Added 5 constraints
    >>> # Check the applied constraints in the network
    >>> print(network.global_constraints)

    """
    constraints = db.list_objects_by_class(ClassEnum.Constraint)
    logger.info(f"Found {len(constraints)} constraints in db")

    for constraint in constraints:
        try:
            # Get constraint properties and memberships
            props = db.get_object_properties(ClassEnum.Constraint, constraint)
            memberships = db.get_memberships_system(
                constraint, object_class=ClassEnum.Constraint
            )

            # Extract relevant properties
            sense = next(
                (float(p["value"]) for p in props if p["property"] == "Sense"), None
            )

            # Collect all RHS values (e.g., RHS Custom, RHS Year, RHS Day)
            rhs_values = [
                float(p["value"])
                for p in props
                if p["property"] in {"RHS Custom", "RHS Year", "RHS Day", "RHS"}
            ]

            # Aggregate RHS values using the mean
            rhs = mean(rhs_values) if rhs_values else None

            # Extract coefficients
            production_coefficients = [
                float(p["value"])
                for p in props
                if p["property"] == "Production Coefficient"
            ]
            generation_coefficients = [
                float(p["value"])
                for p in props
                if p["property"] == "Generation Coefficient"
            ]
            flow_coefficients = [
                float(p["value"])
                for p in props
                if p["property"]
                in {
                    "Flow Forward Coefficient",
                    "Flow Back Coefficient",
                    "Flow Coefficient",
                }
            ]
            storage_coefficients = [
                float(p["value"])
                for p in props
                if p["property"] == "Units Built in Year Coefficient"
            ]
            capacity_coefficients = [
                float(p["value"])
                for p in props
                if p["property"]
                in {"Installed Capacity Coefficient", "Capacity Built Coefficient"}
            ]

            # Infer constraint type
            if production_coefficients and rhs is not None:
                inferred_type = "Emissions Budget"
            elif flow_coefficients:
                inferred_type = "Line Flow Limit"
            elif generation_coefficients:
                inferred_type = "Generation Limit"
            elif storage_coefficients:
                inferred_type = "Storage Constraint"
            elif capacity_coefficients:
                inferred_type = "Capacity Constraint"
            else:
                inferred_type = "Unknown"

            logger.debug(f"Constraint: {constraint}, Type: {inferred_type}, RHS: {rhs}")

            # Apply constraints based on inferred type
            if inferred_type == "Emissions Budget":
                # Apply emissions budget as a global constraint
                network.global_constraints.loc[constraint, "type"] = "primary_energy"
                network.global_constraints.loc[constraint, "sense"] = (
                    "<=" if sense == -1 else ">="
                )
                network.global_constraints.loc[constraint, "constant"] = rhs
                network.global_constraints.loc[constraint, "carrier_attribute"] = (
                    "emissions"
                )
                logger.info(
                    f"Applied emissions budget of {rhs} to constraint {constraint}"
                )

            elif inferred_type == "Line Flow Limit":
                # Apply line flow limits to specific lines
                for membership in memberships:
                    logger.debug(f"Membership: {membership}")
                    if (
                        membership["class"] == "Line"
                        and membership["name"] in network.lines.index
                    ):
                        network.lines.loc[membership["name"], "s_nom"] = rhs
                logger.info(
                    f"Applied line flow limit of {rhs} to line {membership['name']}"
                )

            elif inferred_type == "Generation Limit":
                # Apply generation limits to specific generators
                for membership in memberships:
                    logger.debug(f"Membership: {membership}")
                    if (
                        membership["class"] == "Generator"
                        and membership["name"] in network.generators.index
                    ):
                        # TODO: should this be something else?
                        network.generators.loc[membership["name"], "p_nom"] = rhs
                logger.info(
                    f"Applied generation limit of {rhs} to generator {membership['name']}"
                )

            elif inferred_type == "Storage Constraint":
                # Apply storage constraints to specific storage units
                for membership in memberships:
                    logger.debug(f"Membership: {membership}")
                    if (
                        membership["class"] == "Storage"
                        and membership["name"] in network.storage_units.index
                    ):
                        network.storage_units.loc[membership["name"], "max_hours"] = rhs
                logger.info(
                    f"Applied storage constraint of {rhs} hours to storage {membership['name']}"
                )

            elif inferred_type == "Capacity Constraint":
                # Apply capacity constraints to specific components
                for membership in memberships:
                    logger.debug(f"Membership: {membership}")
                    if (
                        membership["class"] == "Generator"
                        and membership["name"] in network.generators.index
                    ):
                        network.generators.loc[membership["name"], "p_nom_max"] = rhs
                logger.info(
                    f"Applied capacity constraint of {rhs} to generator {membership['name']}"
                )

            else:
                logger.warning(f"Unknown constraint type for {constraint}, skipping.")

        except Exception:
            logger.exception(f"Failed to apply constraint {constraint}")


def add_constraints_enhanced(
    network: Network, db: PlexosDB, verbose: bool = True
) -> dict:
    """Enhanced constraint porting using the comprehensive constraint analysis system.

    This function provides better constraint classification, implementation, and reporting
    compared to the original add_constraints function.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add constraints to
    db : PlexosDB
        The PLEXOS database containing constraint data
    verbose : bool, default True
        Whether to print detailed implementation messages and statistics

    Returns
    -------
    dict
        Summary dictionary with implementation statistics and warnings

    Examples
    --------
    >>> results = add_constraints_enhanced(network, db, verbose=True)
    >>> print(f"Implemented {results['implemented']} constraints")
    """
    return port_plexos_constraints(network, db, verbose=verbose)
