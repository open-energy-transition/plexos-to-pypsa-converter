"""Central registry of PLEXOS model metadata.

This module contains the MODEL_REGISTRY dictionary which stores metadata for all
supported PLEXOS models. It's extracted into a separate module to avoid circular
import dependencies.

Each model entry includes:
- name: Human-readable model name
- source: Organization that created the model
- xml_filename: Name of the PLEXOS XML file
- model_type: Type of model (electricity, multi_sector_gas_electric, multi_sector_flow)
- default_config: Default configuration parameters for the model
- recipe (optional): Auto-download instructions for the model

See models.py for functions that use this registry.
"""

# Model metadata registry
MODEL_REGISTRY = {
    "aemo-2024-green": {
        "name": "AEMO 2024 ISP - Green Energy Exports",
        "source": "AEMO",
        "xml_filename": "2024 ISP Green Energy Exports Model.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
    },
    "aemo-2024-isp-progressive": {
        "name": "AEMO 2024 ISP - Progressive Change",
        "source": "AEMO",
        "xml_filename": "2024 ISP Progressive Change Model.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
    },
    "aemo-2024-isp-step": {
        "name": "AEMO 2024 ISP - Step Change",
        "source": "AEMO",
        "xml_filename": "2024 ISP Step Change Model.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
    },
    "caiso-irp23": {
        "name": "CAISO IRP 2023 Stochastic",
        "source": "CAISO",
        "xml_filename": "CAISOIRP23Stochastic 20240517.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "aggregate_node",
            "aggregate_node_name": "CAISO_Load_Aggregate",
            "model_name": "M01Y2024 PSP23_25MMT",
            "cross_model_dependencies": {
                "vre_profiles_model_id": "aemo-2024-isp-progressive",
            },
        },
        "recipe": [
            {
                "step": "download",
                "url": "https://www.caiso.com/documents/caiso-irp23-stochastic-2024-0517.zip",
                "target": "caiso-irp23.zip",
                "description": "Downloading CAISO IRP23 Stochastic model",
            },
            {
                "step": "extract",
                "source": "caiso-irp23.zip",
                "target": ".",
                "description": "Extracting model files",
            },
            {
                "step": "delete",
                "pattern": "caiso-irp23.zip",
                "description": "Removing archive",
            },
            {
                "step": "validate",
                "checks": ["xml_exists", "required_dir:LoadProfile"],
                "description": "Validating installation",
            },
        ],
    },
    "caiso-sa25": {
        "name": "CAISO 2025 Summer Assessment",
        "source": "CAISO",
        "xml_filename": "CAISOSA25 20250505.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
    },
    "nrel-118": {
        "name": "NREL Extended IEEE 118-bus",
        "source": "NREL",
        "xml_filename": "mti-118-mt-da-rt-reserves-all-generators.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
        # TODO: Add recipe when download URL is available
    },
    "sem-2024-2032": {
        "name": "SEM 2024-2032 Validation Model",
        "source": "SEM",
        "xml_filename": "PUBLIC Validation 2024-2032 Model 2025-03-14.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "target_node",
            "target_node": "SEM",
            "model_name": "Opt A 24-32 (Avail, Uplift, Wheeling)--MIP 25/26",
            "cross_model_dependencies": {
                "vre_profiles_model_id": "aemo-2024-isp-progressive",
            },
        },
    },
    "marei-eu": {
        "name": "European Power & Gas Model",
        "source": "UCC",
        "xml_filename": "European Integrated Power & Gas Model.xml",
        "model_type": "multi_sector_gas_electric",
        "default_config": {
            "use_csv_integration": False,
            "infrastructure_scenario": "PCI",
            "pricing_scheme": "Production",
            "generators_as_links": False,
            "testing_mode": False,
        },
    },
    "plexos-world-2015": {
        "name": "PLEXOS-World 2015 Gold V1.1",
        "source": "UCC",
        "xml_filename": "PLEXOS-World 2015 Gold V1.1.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
    },
    "plexos-world-spatial": {
        "name": "PLEXOS-World Spatial Resolution",
        "source": "UCC",
        "xml_filename": "PLEXOS-World Spatial Resolution Case Study (Second Journal Submission version).xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
    },
    "plexos-message": {
        "name": "MESSAGEix-GLOBIOM",
        "source": "UCC",
        "xml_filename": "H2_Global_MESSAGEix_EN_NPi2020_500.xml",
        "model_type": "multi_sector_flow",
        "default_config": {
            "testing_mode": False,
            "use_csv_integration": True,
        },
    },
}
