#!/usr/bin/env python3
"""Download models for CI using recipe system.

This script ensures models are complete by checking for required files
and downloading via recipes if necessary.
"""

import argparse
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plexos_to_pypsa_converter.db.registry import MODEL_REGISTRY
from plexos_to_pypsa_converter.utils.model_paths import get_model_directory
from plexos_to_pypsa_converter.utils.recipe_executor import RecipeExecutor


def is_model_complete(model_id: str) -> bool:
    """Check if model passes its recipe validation checks.

    This function parses the validation checks from the model's recipe
    and verifies that all required files and directories exist.

    Args:
        model_id: Model identifier

    Returns:
        True if model is complete, False otherwise

    """
    model_dir = get_model_directory(model_id)
    if not model_dir.exists():
        print(f"    Model directory not found: {model_dir}")
        return False

    config = MODEL_REGISTRY[model_id]

    # Find validation step in recipe
    if "recipe" not in config:
        # No recipe - just check if XML exists
        xml_file = model_dir / config["xml_filename"]
        if not xml_file.exists():
            print(f"    XML file not found: {xml_file}")
            return False
        return True

    validate_step = None
    for step in config["recipe"]:
        if step.get("step") == "validate":
            validate_step = step
            break

    if not validate_step:
        # No validation step - just check if XML exists
        xml_file = model_dir / config["xml_filename"]
        if not xml_file.exists():
            print(f"    XML file not found: {xml_file}")
            return False
        return True

    # Run the validation checks
    checks = validate_step.get("checks", [])
    for check in checks:
        if check == "xml_exists":
            xml_file = model_dir / config["xml_filename"]
            if not xml_file.exists():
                print(f"    Missing: {xml_file.name}")
                return False

        elif check.startswith("required_dir:"):
            dir_name = check.split(":", 1)[1]
            required_dir = model_dir / dir_name
            if not required_dir.exists():
                print(f"    Missing: {dir_name}/")
                return False

        elif check.startswith("required_file:"):
            file_name = check.split(":", 1)[1]
            required_file = model_dir / file_name
            if not required_file.exists():
                print(f"    Missing: {file_name}")
                return False

        elif check.startswith("min_files:"):
            min_count = int(check.split(":", 1)[1])
            file_count = len(list(model_dir.glob("*")))
            if file_count < min_count:
                print(f"    Too few files: {file_count} < {min_count}")
                return False

    return True


def download_model(model_id: str) -> bool:
    """Download model using recipe if incomplete.

    Args:
        model_id: Model identifier

    Returns:
        True if successful, False otherwise

    """
    print(f"\nChecking model: {model_id}")

    if model_id not in MODEL_REGISTRY:
        print(f"    Unknown model: {model_id}")
        return False

    # Check if model is complete
    if is_model_complete(model_id):
        print("Model complete, skipping download")
        return True

    # Model incomplete or missing - download using recipe
    print("    Model incomplete, downloading using recipe...")

    config = MODEL_REGISTRY[model_id]
    if "recipe" not in config:
        print(f"No recipe available for {model_id}")
        return False

    model_dir = get_model_directory(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)

    executor = RecipeExecutor(model_id, model_dir, verbose=True)

    try:
        executor.execute_recipe(config["recipe"])
    except Exception as e:
        print(f"Download failed: {e}")
        traceback.print_exc()
        return False
    else:
        print("Download complete")
        return True


def main() -> int:
    """Run model download and validation script."""
    parser = argparse.ArgumentParser(
        description="Download and validate PLEXOS models for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download SEM and AEMO models
    python scripts/download_models.py --models sem-2024-2032 aemo-2024-isp-progressive-change

    # Download single model
    python scripts/download_models.py --models sem-2024-2032
        """,
    )
    parser.add_argument(
        "--models", nargs="+", required=True, help="Model IDs to download/validate"
    )
    args = parser.parse_args()

    print(f"Ensuring models are complete: {', '.join(args.models)}")
    print("=" * 60)

    results = [download_model(model_id) for model_id in args.models]

    print("\n" + "=" * 60)
    if all(results):
        print("All models ready")
        return 0
    print("Some models failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
