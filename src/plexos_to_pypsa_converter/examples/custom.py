"""Example script for running a custom PLEXOS model outside the repo.

This demonstrates how to point the workflow at a model stored in an arbitrary
location (e.g., a USB drive or Dropbox folder) by providing a minimal model
descriptor. Here we reuse the AEMO Progressive Change workflow but tell the
converter to read the XML/CSV files under a CUSTOM_MODEL_DIR.
"""

from pathlib import Path

from plexos_to_pypsa_converter.workflow import run_model_workflow

CUSTOM_MODEL_DIR = Path("/Users/meas/Dropbox/downloads/converter/fake-model-path")
CUSTOM_XML_FILE = CUSTOM_MODEL_DIR / "input.xml"

WORKFLOW_STEPS = [
    {
        "name": "create_model",
        "params": {
            "use_csv": True,
            "xml_file": str(CUSTOM_XML_FILE),
            "model_dir": str(CUSTOM_MODEL_DIR),
        },
    },
    {
        "name": "load_vre_profiles",
        "params": {
            "csv_dir": None,
            "profiles_path": None,
            "property_name": "Rating",
            "target_property": "p_max_pu",
            "target_type": "generators_t",
            "apply_mode": "replace",
            "scenario": 1,
            "generator_filter": "all",
            "carrier_mapping": {"Wind": "wind", "Solar": "solar"},
            "value_scaling": 1.0,
            "manual_mappings": {},
        },
    },
    {
        "name": "add_storage_inflows",
        "params": {
            "csv_dir": None,
            "inflow_path": None,
        },
    },
    {
        "name": "apply_generator_units",
        "params": {"csv_dir": None},
    },
    {
        "name": "parse_outages",
        "params": {
            "csv_dir": None,
            "include_explicit": False,
            "include_forced": True,
            "include_maintenance": True,
            "generator_filter": "exclude_vre",
            "random_seed": 42,
        },
    },
    {
        "name": "save_network",
        "params": {},
    },
]

WORKFLOW_OVERRIDE = {
    "csv_dir_pattern": "csvs_from_xml/NEM",
    "steps": WORKFLOW_STEPS,
}


if __name__ == "__main__":
    NETWORK, SUMMARY = run_model_workflow(
        model_id="custom-model",
        workflow_overrides=WORKFLOW_OVERRIDE,
        model_dir_override=CUSTOM_MODEL_DIR,
        registry_model_id="aemo-2024-isp-progressive-change",
        solve=False,
    )

    print("Custom conversion finished.")
    print("Workflow summary keys:", list(SUMMARY.keys()))
