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
    "aemo-2024-green-energy-exports": {
        "name": "AEMO 2024 ISP - Green Energy Exports",
        "source": "AEMO",
        "xml_filename": "2024 ISP Green Energy Exports Model.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
        "recipe": [
            # Download and extract main model ZIP (contains all 3 models)
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-model.zip",
                "target": "aemo-models.zip",
                "description": "Downloading AEMO 2024 ISP models",
            },
            {
                "step": "extract",
                "source": "aemo-models.zip",
                "target": ".",
                "description": "Extracting model files",
            },
            # Download solar traces and move into nested model folder
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-solar-traces.zip",
                "target": "solar-traces.zip",
                "description": "Downloading solar traces",
            },
            {
                "step": "extract",
                "source": "solar-traces.zip",
                "target": ".",
                "description": "Extracting solar traces",
            },
            {
                "step": "move",
                "source": "solar",
                "target": "2024 ISP Model/2024 ISP Green Energy Exports/Traces",
                "description": "Moving solar traces into model folder",
            },
            {
                "step": "delete",
                "pattern": "solar-traces.zip",
                "description": "Removing solar archive",
            },
            # Download timeslice traces and move into nested model folder
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-timeslice-traces.zip",
                "target": "timeslice-traces.zip",
                "description": "Downloading timeslice traces",
            },
            {
                "step": "extract",
                "source": "timeslice-traces.zip",
                "target": ".",
                "description": "Extracting timeslice traces",
            },
            {
                "step": "move",
                "source": "timeslice",
                "target": "2024 ISP Model/2024 ISP Green Energy Exports/Traces",
                "description": "Moving timeslice traces into model folder",
            },
            {
                "step": "delete",
                "pattern": "timeslice-traces.zip",
                "description": "Removing timeslice archive",
            },
            # Download wind traces and move into nested model folder
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-wind-traces.zip",
                "target": "wind-traces.zip",
                "description": "Downloading wind traces",
            },
            {
                "step": "extract",
                "source": "wind-traces.zip",
                "target": ".",
                "description": "Extracting wind traces",
            },
            {
                "step": "move",
                "source": "wind",
                "target": "2024 ISP Model/2024 ISP Green Energy Exports/Traces",
                "description": "Moving wind traces into model folder",
            },
            {
                "step": "delete",
                "pattern": "wind-traces.zip",
                "description": "Removing wind archive",
            },
            # Move assembled model contents to root
            {
                "step": "delete",
                "pattern": "aemo-models.zip",
                "description": "Removing main model archive",
            },
            {
                "step": "move",
                "source": "2024 ISP Model/2024 ISP Green Energy Exports/*",
                "target": ".",
                "description": "Moving Green Energy Exports model contents to root",
            },
            {
                "step": "delete",
                "pattern": "2024 ISP Model",
                "recursive": True,
                "description": "Removing wrapper folder and other models",
            },
            # Validate installation
            {
                "step": "validate",
                "checks": [
                    "xml_exists",
                    "required_dir:Traces/solar",
                    "required_dir:Traces/timeslice",
                    "required_dir:Traces/wind",
                ],
                "description": "Validating installation",
            },
        ],
    },
    "aemo-2024-isp-progressive-change": {
        "name": "AEMO 2024 ISP - Progressive Change",
        "source": "AEMO",
        "xml_filename": "2024 ISP Progressive Change Model.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
        "recipe": [
            # Download and extract main model ZIP (contains all 3 models)
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-model.zip",
                "target": "aemo-models.zip",
                "description": "Downloading AEMO 2024 ISP models",
            },
            {
                "step": "extract",
                "source": "aemo-models.zip",
                "target": ".",
                "description": "Extracting model files",
            },
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-solar-traces.zip",
                "target": "solar-traces.zip",
                "description": "Downloading solar traces",
            },
            {
                "step": "extract",
                "source": "solar-traces.zip",
                "target": ".",
                "description": "Extracting solar traces",
            },
            {
                "step": "move",
                "source": "solar",
                "target": "2024 ISP Model/2024 ISP Progressive Change/Traces",
                "description": "Moving solar traces into model folder",
            },
            {
                "step": "delete",
                "pattern": "solar-traces.zip",
                "description": "Removing solar archive",
            },
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-timeslice-traces.zip",
                "target": "timeslice-traces.zip",
                "description": "Downloading timeslice traces",
            },
            {
                "step": "extract",
                "source": "timeslice-traces.zip",
                "target": ".",
                "description": "Extracting timeslice traces",
            },
            {
                "step": "move",
                "source": "timeslice",
                "target": "2024 ISP Model/2024 ISP Progressive Change/Traces",
                "description": "Moving timeslice traces into model folder",
            },
            {
                "step": "delete",
                "pattern": "timeslice-traces.zip",
                "description": "Removing timeslice archive",
            },
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-wind-traces.zip",
                "target": "wind-traces.zip",
                "description": "Downloading wind traces",
            },
            {
                "step": "extract",
                "source": "wind-traces.zip",
                "target": ".",
                "description": "Extracting wind traces",
            },
            {
                "step": "move",
                "source": "wind",
                "target": "2024 ISP Model/2024 ISP Progressive Change/Traces",
                "description": "Moving wind traces into model folder",
            },
            {
                "step": "delete",
                "pattern": "wind-traces.zip",
                "description": "Removing wind archive",
            },
            {
                "step": "delete",
                "pattern": "aemo-models.zip",
                "description": "Removing main model archive",
            },
            {
                "step": "move",
                "source": "2024 ISP Model/2024 ISP Progressive Change/*",
                "target": ".",
                "description": "Moving Progressive Change model contents to root",
            },
            {
                "step": "delete",
                "pattern": "2024 ISP Model",
                "recursive": True,
                "description": "Removing wrapper folder and other models",
            },
            {
                "step": "validate",
                "checks": [
                    "xml_exists",
                    "required_dir:Traces/solar",
                    "required_dir:Traces/timeslice",
                    "required_dir:Traces/wind",
                ],
                "description": "Validating installation",
            },
        ],
    },
    "aemo-2024-isp-step-change": {
        "name": "AEMO 2024 ISP - Step Change",
        "source": "AEMO",
        "xml_filename": "2024 ISP Step Change Model.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
        "recipe": [
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-model.zip",
                "target": "aemo-models.zip",
                "description": "Downloading AEMO 2024 ISP models",
            },
            {
                "step": "extract",
                "source": "aemo-models.zip",
                "target": ".",
                "description": "Extracting model files",
            },
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-solar-traces.zip",
                "target": "solar-traces.zip",
                "description": "Downloading solar traces",
            },
            {
                "step": "extract",
                "source": "solar-traces.zip",
                "target": ".",
                "description": "Extracting solar traces",
            },
            {
                "step": "move",
                "source": "solar",
                "target": "2024 ISP Model/2024 ISP Step Change/Traces",
                "description": "Moving solar traces into model folder",
            },
            {
                "step": "delete",
                "pattern": "solar-traces.zip",
                "description": "Removing solar archive",
            },
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-timeslice-traces.zip",
                "target": "timeslice-traces.zip",
                "description": "Downloading timeslice traces",
            },
            {
                "step": "extract",
                "source": "timeslice-traces.zip",
                "target": ".",
                "description": "Extracting timeslice traces",
            },
            {
                "step": "move",
                "source": "timeslice",
                "target": "2024 ISP Model/2024 ISP Step Change/Traces",
                "description": "Moving timeslice traces into model folder",
            },
            {
                "step": "delete",
                "pattern": "timeslice-traces.zip",
                "description": "Removing timeslice archive",
            },
            {
                "step": "download",
                "url": "https://www.aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-wind-traces.zip",
                "target": "wind-traces.zip",
                "description": "Downloading wind traces",
            },
            {
                "step": "extract",
                "source": "wind-traces.zip",
                "target": ".",
                "description": "Extracting wind traces",
            },
            {
                "step": "move",
                "source": "wind",
                "target": "2024 ISP Model/2024 ISP Step Change/Traces",
                "description": "Moving wind traces into model folder",
            },
            {
                "step": "delete",
                "pattern": "wind-traces.zip",
                "description": "Removing wind archive",
            },
            {
                "step": "delete",
                "pattern": "aemo-models.zip",
                "description": "Removing main model archive",
            },
            {
                "step": "move",
                "source": "2024 ISP Model/2024 ISP Step Change/*",
                "target": ".",
                "description": "Moving Step Change model contents to root",
            },
            {
                "step": "delete",
                "pattern": "2024 ISP Model",
                "recursive": True,
                "description": "Removing wrapper folder and other models",
            },
            {
                "step": "validate",
                "checks": [
                    "xml_exists",
                    "required_dir:Traces/solar",
                    "required_dir:Traces/timeslice",
                    "required_dir:Traces/wind",
                ],
                "description": "Validating installation",
            },
        ],
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
                "vre_profiles_model_id": "aemo-2024-isp-progressive-change",
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
        "recipe": [
            {
                "step": "download",
                "url": "https://www.caiso.com/documents/2025-summer-loads-and-resources-assessment-public-stochastic-model.zip",
                "target": "caiso-sa25.zip",
                "description": "Downloading CAISO 2025 Summer Assessment model",
            },
            {
                "step": "extract",
                "source": "caiso-sa25.zip",
                "target": ".",
                "description": "Extracting model files",
            },
            {
                "step": "delete",
                "pattern": "caiso-sa25.zip",
                "description": "Removing archive",
            },
            {
                "step": "validate",
                "checks": ["xml_exists"],
                "description": "Validating installation",
            },
        ],
    },
    "nrel-118": {
        "name": "NREL Extended IEEE 118-bus",
        "source": "NREL",
        "xml_filename": "mti-118-mt-da-rt-reserves-all-generators.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
        "recipe": [
            {
                "step": "manual",
                "instructions": "Download 'mti-118-mt-da-rt-reserves-all-generators.xml' from https://db.bettergrids.org/bettergrids/handle/1001/120 and place it in this folder.",
                "description": "Manual download required for NREL 118 XML file.",
            },
            {
                "step": "manual",
                "instructions": "Download 'Input-files.zip' from https://db.bettergrids.org/bettergrids/handle/1001/120, extract it, and place the resulting 'Input files' folder in this directory.",
                "description": "Manual download and extraction required for NREL 118 input files.",
            },
            {
                "step": "manual",
                "instructions": "Download 'additional-files-mti-118.zip' from https://db.bettergrids.org/bettergrids/handle/1001/120, extract it, and place the resulting 'Additional Files MTI 118' folder in this directory.",
                "description": "Manual download and extraction required for NREL 118 additional files.",
            },
            {
                "step": "validate",
                "checks": [
                    "xml_exists",
                    "required_dir:Input files",
                    "required_dir:Additional Files MTI 118",
                ],
                "description": "Validating installation",
            },
        ],
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
                "vre_profiles_model_id": "aemo-2024-isp-progressive-change",
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
        "recipe": [
            {
                "step": "download",
                "url": "https://www.dropbox.com/scl/fi/biv5n52x8s5pxeh06u2b1/EU-Power-Gas-Model.zip?rlkey=hmscke4vsknxbj6w18vosfyxb&e=1&dl=1",
                "target": "marei-eu.zip",
                "description": "Downloading European Power & Gas Model",
            },
            {
                "step": "extract",
                "source": "marei-eu.zip",
                "target": ".",
                "description": "Extracting model files",
            },
            {
                "step": "flatten",
                "source": "EU Power & Gas Model",
                "levels": 1,
                "description": "Moving contents to root directory",
            },
            {
                "step": "delete",
                "pattern": "marei-eu.zip",
                "description": "Removing archive",
            },
            {
                "step": "validate",
                "checks": ["xml_exists", "required_dir:CSV Files"],
                "description": "Validating installation",
            },
        ],
    },
    "plexos-world-2015": {
        "name": "PLEXOS-World 2015 Gold V1.1",
        "source": "UCC",
        "xml_filename": "PLEXOS-World 2015 Gold V1.1.xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
        "recipe": [
            # Download main XML file (124 MB)
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/4214857",
                "target": "PLEXOS-World 2015 Gold V1.1.xml",
                "description": "Downloading main XML file",
            },
            # Download supporting data files
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/3985039",
                "target": "All Demand UTC 2015.tab",
                "description": "Downloading demand data",
            },
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/3737270",
                "target": "Hydro_Monthly_Profiles (15 year average).tab",
                "description": "Downloading hydro profiles (15yr avg)",
            },
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/3985040",
                "target": "Hydro_Monthly_Profiles (2015).tab",
                "description": "Downloading hydro profiles (2015)",
            },
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/4300175",
                "target": "renewables.ninja.Csp.output.full.adjusted.tab",
                "description": "Downloading CSP profiles (adjusted)",
            },
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/4300174",
                "target": "renewables.ninja.Csp.output.full.tab",
                "description": "Downloading CSP profiles",
            },
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/3985046",
                "target": "renewables.ninja.Solar.farms.output.full.adjusted.csv",
                "description": "Downloading solar profiles (adjusted)",
            },
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/3985041",
                "target": "renewables.ninja.Solar.farms.output.full.csv",
                "description": "Downloading solar profiles",
            },
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/3985047",
                "target": "Renewables.ninja.wind.output.Full.adjusted.csv",
                "description": "Downloading wind profiles (adjusted)",
            },
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/3985042",
                "target": "Renewables.ninja.wind.output.Full.csv",
                "description": "Downloading wind profiles",
            },
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/4008393",
                "target": "PLEXOS-World 2015 Gold V1.1.tab",
                "description": "Downloading model metadata",
            },
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/3985043",
                "target": "Journal paper supplementary material for manuscript 'Building and Calibrating a Country-Level Detailed Global Electricity Model Based on Public Data'.docx",
                "description": "Downloading documentation",
            },
            # Validate installation
            {
                "step": "validate",
                "checks": ["xml_exists", "min_files:12"],
                "description": "Validating installation",
            },
        ],
    },
    "plexos-world-spatial": {
        "name": "PLEXOS-World Spatial Resolution",
        "source": "UCC",
        "xml_filename": "PLEXOS-World Spatial Resolution Case Study (Second Journal Submission version).xml",
        "model_type": "electricity",
        "default_config": {
            "demand_assignment_strategy": "per_node",
        },
        "recipe": [
            # Download main XML file (234 MB)
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/7882637",
                "target": "PLEXOS-World Spatial Resolution Case Study (Second Journal Submission version).xml",
                "description": "Downloading main XML file",
            },
            # Download and extract input timeseries RAR (365 MB)
            {
                "step": "download",
                "url": "https://dataverse.harvard.edu/api/access/datafile/7882636",
                "target": "Input timeseries.rar",
                "description": "Downloading input timeseries",
            },
            {
                "step": "extract",
                "source": "Input timeseries.rar",
                "target": ".",
                "description": "Extracting input timeseries",
            },
            {
                "step": "delete",
                "pattern": "Input timeseries.rar",
                "description": "Removing input timeseries archive",
            },
            # Validate installation
            {
                "step": "validate",
                "checks": [
                    "xml_exists",
                    "required_dir:Base Profiles",
                    "required_dir:Bin Profiles",
                    "required_dir:Forecasted Profiles",
                    "required_dir:Load Files",
                ],
                "description": "Validating installation",
            },
        ],
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
        "recipe": [
            {
                "step": "download",
                "url": "https://github.com/DuncanDotPY/MESSAGEix-GLOBIOM-EN-NPi2020-500-Soft-Link/archive/refs/heads/main.zip",
                "target": "plexos-message.zip",
                "description": "Downloading MESSAGEix-GLOBIOM model from GitHub",
            },
            {
                "step": "extract",
                "source": "plexos-message.zip",
                "target": ".",
                "description": "Extracting model files",
            },
            {
                "step": "move",
                "source": "MESSAGEix-GLOBIOM-EN-NPi2020-500-Soft-Link-main/*",
                "target": ".",
                "description": "Moving model contents to root directory",
            },
            {
                "step": "delete",
                "pattern": "MESSAGEix-GLOBIOM-EN-NPi2020-500-Soft-Link-main",
                "recursive": True,
                "description": "Removing GitHub folder wrapper",
            },
            {
                "step": "delete",
                "pattern": "plexos-message.zip",
                "description": "Removing archive",
            },
            {
                "step": "validate",
                "checks": ["xml_exists"],
                "description": "Validating installation",
            },
        ],
    },
}
