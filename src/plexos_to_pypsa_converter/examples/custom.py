"""Example: run conversion on a custom PLEXOS model directory with auto-detection.

Point this script at your model directory (containing the XML).
"""

from pathlib import Path

from plexos_to_pypsa_converter.workflow import run_model_workflow

CUSTOM_MODEL_DIR = Path("/path/to/your/plexos/model_directory")

network, summary = run_model_workflow(
    model_id="custom-model",
    model_dir_override=CUSTOM_MODEL_DIR,
    auto_workflow=True,
    enable_outages=False,
    solve=False,
)
