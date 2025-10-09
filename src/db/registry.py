"""Central registry of example PLEXOS models' metadata.

This module contains the MODEL_REGISTRY dictionary which stores metadata for all
supported example PLEXOS models.

Each model can optionally include a "recipe" - a list of instructions for automatically
downloading, extracting, and organizing model files. Recipes are executed when a model
is requested but not found locally (if auto_download is enabled).

Each model entry includes:
- name: Human-readable model name
- source: Organization that created the model
- xml_filename: Name of the PLEXOS XML file
- model_type: Type of model (electricity, multi_sector_gas_electric, multi_sector_flow)
- default_config: Default configuration parameters for the model
- recipe (optional): Auto-download instructions for the model

Recipe Instruction Types:
- download: Download file from URL
- extract: Extract archive to target directory
- move: Move files/directories (supports glob patterns)
- copy: Copy files/directories
- rename: Rename file/directory
- delete: Delete files/directories
- create_dir: Create directory
- flatten: Flatten nested directory structure
- validate: Validate model installation
- manual: Display manual download instructions
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
            "demand_file": "CSV Files (1 of 2)/AI Demand_2023-2033_5base years.csv",
            "cross_model_dependencies": {
                "vre_profiles_model_id": "aemo-2024-isp-progressive",
            },
        },
        "recipe": [
            # Main model ZIP (contains XML file)
            {
                "step": "download",
                "url": "https://www.semcommittee.com/files/semcommittee/2025-03/SEM%20PLEXOS%20Forecast%20Model%202024-2032%28%20Public%20Version%29.zip",
                "target": "sem-main.zip",
                "description": "Downloading main model ZIP",
            },
            {
                "step": "extract",
                "source": "sem-main.zip",
                "target": ".",
                "description": "Extracting main model files",
            },
            {
                "step": "flatten",
                "source": "SEM PLEXOS Forecast Model 2024-2032( Public Version)",
                "levels": 1,
                "description": "Moving XML to root directory",
            },
            {
                "step": "delete",
                "pattern": "sem-main.zip",
                "description": "Removing main ZIP file",
            },
            # CSV Files 1 of 2
            {
                "step": "download",
                "url": "https://www.semcommittee.com/files/semcommittee/2025-03/SEM%20PLEXOS%20Forecast%20Model%202024-2032%20%28CSV%20Files%201%20of%202%29.zip",
                "target": "csv1.zip",
                "description": "Downloading CSV Files 1 of 2",
            },
            {
                "step": "extract",
                "source": "csv1.zip",
                "target": ".",
                "description": "Extracting CSV Files 1 of 2",
            },
            {
                "step": "rename",
                "source": "SEM PLEXOS Forecast Model 2024-2032 (CSV Files 1 of 2)/CSV Files (1 of 2)",
                "target": "CSV Files (1 of 2)",
                "description": "Moving CSV Files (1 of 2) to root",
            },
            {
                "step": "delete",
                "pattern": "SEM PLEXOS Forecast Model 2024-2032 (CSV Files 1 of 2)",
                "recursive": True,
                "description": "Removing wrapper folder",
            },
            {
                "step": "delete",
                "pattern": "csv1.zip",
                "description": "Removing CSV 1 ZIP file",
            },
            # CSV Files 2 of 2
            {
                "step": "download",
                "url": "https://www.semcommittee.com/files/semcommittee/2025-03/SEM%20PLEXOS%20Forecast%20Model%202024-2032%20%28CSV%20Files%202%20of%202%29.zip",
                "target": "csv2.zip",
                "description": "Downloading CSV Files 2 of 2",
            },
            {
                "step": "extract",
                "source": "csv2.zip",
                "target": ".",
                "description": "Extracting CSV Files 2 of 2",
            },
            {
                "step": "rename",
                "source": "SEM PLEXOS Forecast Model 2024-2032 (CSV Files 2 of 2)/CSV Files (2 of 2)",
                "target": "CSV Files (2 of 2)",
                "description": "Moving CSV Files (2 of 2) to root",
            },
            {
                "step": "delete",
                "pattern": "SEM PLEXOS Forecast Model 2024-2032 (CSV Files 2 of 2)",
                "recursive": True,
                "description": "Removing wrapper folder",
            },
            {
                "step": "delete",
                "pattern": "csv2.zip",
                "description": "Removing CSV 2 ZIP file",
            },
            # Additional Input Files (Excel files)
            {
                "step": "download",
                "url": "https://www.semcommittee.com/files/semcommittee/2025-03/SEM%20PLEXOS%20Forecast%20Model%202024-2032%28%20Additional%20Input%20Files%29.zip",
                "target": "additional.zip",
                "description": "Downloading Additional Input Files",
            },
            {
                "step": "extract",
                "source": "additional.zip",
                "target": ".",
                "description": "Extracting Additional Input Files",
            },
            {
                "step": "move",
                "source": "SEM PLEXOS Forecast Model 2024-2032( Additional Input Files)/*.xlsx",
                "target": ".",
                "description": "Moving Excel files to root",
            },
            {
                "step": "delete",
                "pattern": "SEM PLEXOS Forecast Model 2024-2032( Additional Input Files)",
                "recursive": True,
                "description": "Removing wrapper folder",
            },
            {
                "step": "delete",
                "pattern": "additional.zip",
                "description": "Removing Additional Input Files ZIP",
            },
            # Create demand folder with copy of AI Demand CSV
            {
                "step": "create_dir",
                "path": "demand",
                "description": "Creating demand directory",
            },
            {
                "step": "copy",
                "source": "CSV Files (1 of 2)/AI Demand_2023-2033_5base years.csv",
                "target": "demand/AI Demand_2023-2033_5base years.csv",
                "description": "Copying AI Demand CSV to demand folder",
            },
            # Validate installation
            {
                "step": "validate",
                "checks": [
                    "xml_exists",
                    "required_dir:CSV Files (1 of 2)",
                    "required_dir:CSV Files (2 of 2)",
                    "required_dir:demand",
                ],
                "description": "Validating installation",
            },
        ],
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
