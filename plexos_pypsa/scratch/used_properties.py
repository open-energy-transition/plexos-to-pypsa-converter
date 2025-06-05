import glob
import os

import pandas as pd

from plexos_pypsa.db.plexosdb import PlexosDB


def get_collections_for_class(db, class_id):
    """
    Returns a DataFrame of all collections for a given class_id.
    """
    query = """
        SELECT collection_id, description
        FROM t_collection
        WHERE child_class_id = ?
    """
    rows = db.query(query, (class_id,))
    return pd.DataFrame(rows, columns=["collection_id", "collection_description"])


def get_used_properties_for_class(db, class_id):
    """
    Returns a DataFrame of all actually used property names for all objects of a given class (no data_id, no per-object info).
    """
    # Get all objects of this class
    query_obj = """
        SELECT object_id FROM t_object WHERE class_id = ?
    """
    obj_rows = db.query(query_obj, (class_id,))
    obj_ids = [row[0] for row in obj_rows]
    if not obj_ids:
        return pd.DataFrame(columns=["property"])
    # Get all memberships for these objects
    query_mem = f"""
        SELECT membership_id FROM t_membership WHERE child_object_id IN ({",".join(["?"] * len(obj_ids))})
    """
    mem_rows = db.query(query_mem, tuple(obj_ids))
    mem_ids = [row[0] for row in mem_rows]
    if not mem_ids:
        return pd.DataFrame(columns=["property"])
    # Get all data for these memberships
    query_data = f"""
        SELECT DISTINCT p.name as property
        FROM t_data d
        JOIN t_property p ON d.property_id = p.property_id
        WHERE d.membership_id IN ({",".join(["?"] * len(mem_ids))})
    """
    data_rows = db.query(query_data, tuple(mem_ids))
    return pd.DataFrame(data_rows, columns=["property"])


def get_possible_properties_for_collections(db, collection_ids):
    """
    Returns a DataFrame of all possible properties for the given collection_ids.
    """
    query = f"""
        SELECT property_id, collection_id, name as property, description as property_description
        FROM t_property
        WHERE collection_id IN ({",".join(["?"] * len(collection_ids))})
    """
    rows = db.query(query, tuple(collection_ids))
    return pd.DataFrame(
        rows,
        columns=["property_id", "collection_id", "property", "property_description"],
    )


def get_class_name_for_id(db, class_id):
    query = "SELECT name FROM t_class WHERE class_id = ?"
    rows = db.query(query, (class_id,))
    if rows:
        return rows[0][0]
    return None


def get_all_possible_and_used_properties(xml_file, class_id):
    db = PlexosDB.from_xml(xml_file)
    class_name = get_class_name_for_id(db, class_id)
    # 1. Find all collections for this class
    collections_df = get_collections_for_class(db, class_id)
    if collections_df.empty:
        print(f"No collections found for class_id {class_id}")
        return None, None
    # 2. Find all possible properties for these collections
    possible_props_df = get_possible_properties_for_collections(
        db, collections_df["collection_id"].tolist()
    )
    # Add collection_description to possible_props_df
    possible_props_df = pd.merge(
        possible_props_df, collections_df, on="collection_id", how="left"
    )
    # 3. Find all used properties for this class (across all objects)
    used_props_names_df = get_used_properties_for_class(db, class_id)
    # 4. Merge: enrich used properties with info from possible_props_df and collections_df
    used_props_df = pd.merge(
        used_props_names_df, possible_props_df, on="property", how="left"
    )
    # 5. Add class_id and class_name columns
    possible_props_df["class_id"] = class_id
    possible_props_df["class_name"] = class_name
    used_props_df["class_id"] = class_id
    used_props_df["class_name"] = class_name
    return possible_props_df[
        [
            "class_id",
            "class_name",
            "collection_id",
            "collection_description",
            "property_id",
            "property",
            "property_description",
        ]
    ], used_props_df[
        [
            "class_id",
            "class_name",
            "collection_id",
            "collection_description",
            "property_id",
            "property",
            "property_description",
        ]
    ]


def get_all_possible_and_used_properties_for_all_classes(xml_file, class_id_to_name):
    db = PlexosDB.from_xml(xml_file)
    all_possible = []
    all_used = []
    for class_id, class_name in class_id_to_name.items():
        collections_df = get_collections_for_class(db, class_id)
        if collections_df.empty:
            print(f"No collections found for class_id {class_id} ({class_name})")
            continue
        possible_props_df = get_possible_properties_for_collections(
            db, collections_df["collection_id"].tolist()
        )
        possible_props_df = pd.merge(
            possible_props_df, collections_df, on="collection_id", how="left"
        )
        used_props_names_df = get_used_properties_for_class(db, class_id)
        used_props_df = pd.merge(
            used_props_names_df, possible_props_df, on="property", how="left"
        )
        possible_props_df["class_id"] = class_id
        possible_props_df["class_name"] = class_name
        used_props_df["class_id"] = class_id
        used_props_df["class_name"] = class_name
        all_possible.append(
            possible_props_df[
                [
                    "class_id",
                    "class_name",
                    "collection_id",
                    "collection_description",
                    "property_id",
                    "property",
                    "property_description",
                ]
            ]
        )
        all_used.append(
            used_props_df[
                [
                    "class_id",
                    "class_name",
                    "collection_id",
                    "collection_description",
                    "property_id",
                    "property",
                    "property_description",
                ]
            ]
        )
    all_possible_df = (
        pd.concat(all_possible, ignore_index=True) if all_possible else pd.DataFrame()
    )
    all_used_df = pd.concat(all_used, ignore_index=True) if all_used else pd.DataFrame()
    return all_possible_df, all_used_df


def get_all_possible_and_used_properties_for_all_xmls(
    xml_files, class_id_to_name, save_path
):
    all_possible = []
    all_used = []
    for xml_file in xml_files:
        base = os.path.splitext(os.path.basename(xml_file))[0]
        possible_df, used_df = get_all_possible_and_used_properties_for_all_classes(
            xml_file, class_id_to_name
        )
        if possible_df is not None and not possible_df.empty:
            possible_df = possible_df.copy()
            possible_df["filename"] = base
            all_possible.append(possible_df)
            all_csv = f"{base}_all.csv"
            all_csv_path = os.path.join(save_path, all_csv)
            possible_df.to_csv(all_csv_path, index=False)
            print(f"Saved all possible properties to {all_csv_path}")
        if used_df is not None and not used_df.empty:
            used_df = used_df.copy()
            used_df["filename"] = base
            all_used.append(used_df)
            used_csv = f"{base}_used.csv"
            used_csv_path = os.path.join(save_path, used_csv)
            used_df.to_csv(used_csv_path, index=False)
            print(f"Saved used properties to {used_csv_path}")
    all_possible_df = (
        pd.concat(all_possible, ignore_index=True) if all_possible else pd.DataFrame()
    )
    all_used_df = pd.concat(all_used, ignore_index=True) if all_used else pd.DataFrame()
    return all_possible_df, all_used_df


def get_combined_properties_for_all_xmls(xml_files, class_id_to_name, save_path):
    for xml_file in xml_files:
        base = os.path.splitext(os.path.basename(xml_file))[0]
        # Get the subfolder from the first portion of the xml_file path (relative to COMMON_PREFIX)
        rel_path = os.path.relpath(xml_file, COMMON_PREFIX)
        subfolder = rel_path.split(os.sep)[0]
        out_dir = os.path.join(save_path, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        possible_df, used_df = get_all_possible_and_used_properties_for_all_classes(
            xml_file, class_id_to_name
        )
        if possible_df is not None and not possible_df.empty:
            possible_df = possible_df.copy()
            possible_df["filename"] = base
            used_set = (
                set(
                    tuple(row)
                    for row in used_df[
                        [
                            "class_id",
                            "class_name",
                            "collection_id",
                            "collection_description",
                            "property_id",
                            "property",
                            "property_description",
                        ]
                    ].itertuples(index=False, name=None)
                )
                if used_df is not None and not used_df.empty
                else set()
            )

            def is_used(row):
                return (
                    tuple(
                        row[
                            [
                                "class_id",
                                "class_name",
                                "collection_id",
                                "collection_description",
                                "property_id",
                                "property",
                                "property_description",
                            ]
                        ]
                    )
                    in used_set
                )

            possible_df["used"] = possible_df.apply(is_used, axis=1)
            # Save single CSV per file
            out_csv = f"{base}_properties.csv"
            out_csv_path = os.path.join(out_dir, out_csv)
            possible_df.to_csv(out_csv_path, index=False)
            print(f"Saved combined possible/used properties to {out_csv_path}")


def summarize_property_usage_across_models(properties_dir):
    """
    Analyze all *_properties.csv files in subfolders of properties_dir and output a DataFrame
    with a count of how many models use each property, and the total number of models.
    """
    import os

    import pandas as pd

    # Find all *_properties.csv files recursively
    pattern = os.path.join(properties_dir, "*", "*_properties.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"No property files found in {properties_dir}")
        return None
    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Only keep used properties
        used = df[df["used"]].copy()
        used["model"] = os.path.splitext(os.path.basename(f))[0].replace(
            "_properties", ""
        )
        all_dfs.append(used)
    if not all_dfs:
        print("No used properties found in any model.")
        return None
    concat = pd.concat(all_dfs, ignore_index=True)
    # Group by property identity columns
    group_cols = [
        "class_id",
        "class_name",
        "collection_id",
        "collection_description",
        "property_id",
        "property",
        "property_description",
    ]
    usage = (
        concat.groupby(group_cols)
        .agg(
            models_used=("model", "nunique"),
            models_list=("model", lambda x: list(sorted(set(x)))),
        )
        .reset_index()
    )
    total_models = len(files)
    usage["total_models"] = total_models
    usage["fraction"] = usage["models_used"] / total_models
    usage = usage.sort_values(["class_name", "property"])

    # reorder columns for better readability
    usage = usage[
        [
            "class_id",
            "class_name",
            "collection_id",
            "collection_description",
            "property_id",
            "property",
            "property_description",
            "models_used",
            "total_models",
            "fraction",
            "models_list",
        ]
    ]
    return usage


CLASS_ID_TO_NAME = {
    1: "System",
    2: "Generator",
    3: "Power Station",
    4: "Fuel",
    5: "Fuel Contract",
    6: "Power2X",
    7: "Battery",
    8: "Storage",
    9: "Waterway",
    10: "Emission",
    11: "Abatement",
    12: "Physical Contract",
    13: "Purchaser",
    14: "Reserve",
    15: "Reliability",
    16: "Financial Contract",
    17: "Cournot",
    18: "RSI",
    19: "Region",
    20: "Pool",
    21: "Zone",
    22: "Node",
    23: "Load",
    24: "Line",
    25: "MLF",
    26: "Transformer",
    27: "Flow Control",
    28: "Interface",
    29: "Contingency",
    30: "Hub",
    31: "Transmission Right",
    32: "Heat Plant",
    33: "Heat Node",
    34: "Heat Storage",
    35: "Gas Field",
    36: "Gas Plant",
    37: "Gas Pipeline",
    38: "Gas Node",
    39: "Gas Storage",
    40: "Gas Demand",
    41: "Gas DSM Program",
    42: "Gas Basin",
    43: "Gas Zone",
    44: "Gas Contract",
    45: "Gas Transport",
    46: "Gas Path",
    47: "Gas Capacity Release Offer",
    48: "Water Plant",
    49: "Water Pipeline",
    50: "Water Node",
    51: "Water Storage",
    52: "Water Demand",
    53: "Water Zone",
    54: "Water Pump Station",
    55: "Water Pump",
    56: "Vehicle",
    57: "Charging Station",
    58: "Fleet",
    59: "Company",
    60: "Commodity",
    61: "Process",
    62: "Facility",
    63: "Maintenance",
    64: "Flow Network",
    65: "Flow Node",
    66: "Flow Path",
    67: "Flow Storage",
    68: "Entity",
    69: "Market",
    70: "Constraint",
    71: "Objective",
    72: "Decision Variable",
    73: "Nonlinear Constraint",
    74: "Data File",
    75: "Variable",
    76: "Timeslice",
    77: "Global",
    78: "Scenario",
    79: "Weather Station",
    80: "Model",
    81: "Project",
    82: "Horizon",
    83: "Report",
    84: "Stochastic",
    85: "Preview",
    86: "LT Plan",
    87: "PASA",
    88: "MT Schedule",
    89: "ST Schedule",
    90: "Transmission",
    91: "Production",
    92: "Competition",
    93: "Performance",
    94: "Diagnostic",
    95: "List",
    96: "Layout",
}

# Common prefix for all xml_files
COMMON_PREFIX = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/Plexos Converter/Input Models"

xml_files = [
    "AEMO/2024 ISP/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml",
    "AEMO/2024 ISP/2024 ISP Green Energy Exports/2024 ISP Green Energy Exports Model.xml",
    "AEMO/2024 ISP/2024 ISP Step Change/2024 ISP Step Change Model.xml",
    "AEMO/2022 ISP/2022 Final ISP Model/2022 ISP Hydrogen Superpower/2022 Hydrogen Superpower ISP Model.xml",
    "AEMO/2022 ISP/2022 Final ISP Model/2022 ISP Progressive Change/2022 Progressive Change ISP Model.xml",
    "AEMO/2022 ISP/2022 Final ISP Model/2022 ISP Slow Change/2022 Slow Change ISP Model.xml",
    "AEMO/2022 ISP/2022 Final ISP Model/2022 ISP Step Change/2022 Step Change ISP Model.xml",
    "CAISO/IRP/IRP23 - 25MMT Stochastic models with CEC 2023 IEPR Load Forecast/caiso-irp23-stochastic-2024-0517/CAISOIRP23Stochastic 20240517.xml",
    "CAISO/IRP/IRP20 - 38MMT PSP Stochastic and Deterministic Models with CEC 2019 IEPR Load Forecast/caiso-integrated-resource-planning-38mmt-coreportfolio-plexos-deterministic-2026-2030/CAISOIRP21 0130.xml",
    "CAISO/IRP/IRP20 - 38MMT PSP Stochastic and Deterministic Models with CEC 2019 IEPR Load Forecast/caiso-integrated-resource-planning-38mmt-coreportfolio-plexos-stochastic-2026/CAISOIRP21 Stochastic2026 0130.xml",
    "CAISO/IRP/IRP20 - 38MMT PSP Stochastic and Deterministic Models with CEC 2019 IEPR Load Forecast/caiso-integrated-resource-planning-38mmt-coreportfolio-plexos-stochastic-2030/CAISOIRP21 Stochastic2030 0130.xml",
    "CAISO/Seasonal Assessments/2025-summer-loads-and-resources-assessment-public-stochastic-model/CAISOSA25 20250505.xml",
    "NREL/NREL-118/mti-118-mt-da-rt-reserves-all-generators.xml",
    "SEM/SEM 2024-2032/SEM PLEXOS Forecast Model 2024-2032( Public Version)/PUBLIC Validation 2024-2032 Model 2025-03-14.xml",
    "University College Cork/MaREI/EU Power & Gas Model/European Integrated Power & Gas Model.xml",
    "University College Cork/PLEXOS-World - Spatial Resolution Case Study/dataverse_files/PLEXOS-World Spatial Resolution Case Study (Second Journal Submission version).xml",
]

full_paths = [os.path.join(COMMON_PREFIX, rel_path) for rel_path in xml_files]

if __name__ == "__main__":
    save_path = "plexos_pypsa/data/properties/"
    get_combined_properties_for_all_xmls(full_paths, CLASS_ID_TO_NAME, save_path)
    summary = summarize_property_usage_across_models("plexos_pypsa/data/properties")
    summary.to_csv(
        "plexos_pypsa/data/properties/property_usage_summary.csv", index=False
    )
    # print(summary.head())
