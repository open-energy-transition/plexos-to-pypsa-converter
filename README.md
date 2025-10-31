# PLEXOS-to-PyPSA Converter

Convert PLEXOS input models to PyPSA networks for optimization and analysis.

This tool converts PLEXOS input XML files and their associated data into PyPSA networks that you can solve, analyze, and visualize.

![Milestones chart](doc/visualization/image/milestones.png)

## Key Features

- **Automated workflow** - One-line conversion from PLEXOS XML to solved PyPSA network
- **Flexible demand strategies** - Per-node, aggregate, or target node assignment
- **Rich analysis tools** - Built-in `NetworkAnalyzer` with generation, capacity, and energy balance plots
- **Interactive notebooks** - Example Jupyter notebooks

## Requirements

- **Python**: 3.11 or 3.12
- **PyPSA**: Version <1.0 (specifically >=0.34, <1.0)

## Installation

### Install from PyPI

```bash
pip install plexos-to-pypsa-converter
```

### Development Install

```bash
git clone https://github.com/open-energy-transition/plexos-to-pypsa-converter.git
cd plexos-to-pypsa-converter
pip install -e .
```

### Optional: System Dependencies

Some models use RAR archives and require `unrar`:

```bash
# macOS
brew install rar
```

This is only needed for RAR-based models like `plexos-world-spatial`.

## Usage

### Quick Start with Example Models

The easiest way to get started is with the high-level workflow API:

```python
from workflow import run_model_workflow

# Run a complete workflow: load → convert → optimize → save
network, setup_summary = run_model_workflow("caiso-irp23")

# Solve for specific year (multi-period models)
network, setup_summary = run_model_workflow(
    "aemo-2024-isp-progressive-change",
    optimize__year=2025
)
```

This automatically:
1. Downloads the model data (if the data does not exist locally)
2. Converts PLEXOS XML to CSV files (using `plexos-coad`)
3. Creates PyPSA network with proper topology and components
4. Adds other features, such as renewable profiles, retirements, generator outages, and slack generators
5. Solves the optimization problem
6. Saves results to NetCDF format

### Interactive Notebooks

For interactive analysis and visualization, check out the example notebooks:

- **CAISO IRP23**:
  - `src/examples/caiso_irp23/caiso_solve.ipynb` - Model conversion and solve
  - `src/examples/caiso_irp23/caiso_analysis.ipynb` - Analysis and plots

- **AEMO 2024 Progressive Change**:
  - `src/examples/aemo_2024_prog/aemo_solve.ipynb` - Model conversion and solve
  - `src/examples/aemo_2024_prog/aemo_analysis.ipynb` - Analysis and plots

These notebooks show how to:
- Convert and solve PLEXOS models
- Generate network statistics
- Create plots for simple analysis and validation
- Export results for further analysis

Note that you have to run the `{}_solve.ipynb` notebooks before you are able to run the `{}_analysis.ipynb` notebooks.

### Network Analysis

Analyze solved networks using the `NetworkAnalyzer`:

```python
from analysis.core import NetworkAnalyzer

# Load a solved network
analyzer = NetworkAnalyzer.from_netcdf("solved_network.nc")

# Generate plots
analyzer.plot_generation_by_carrier()
analyzer.plot_installed_capacity()
analyzer.plot_capacity_factors()
analyzer.plot_energy_balance()

# Create comprehensive dashboard
analyzer.plot_dashboard()

# Get network statistics
stats = analyzer.info()
print(f"Buses: {stats['buses']}, Generators: {stats['generators']}")
```

### Advanced: Custom Workflows

The default workflow steps for each model is specified in `src/db/registry.py`.
However, you can customize the workflow steps to skip some steps. For example, to skip saving the network and to save under a custom filename:

```python
from db.registry import MODEL_REGISTRY
from workflow import run_model_workflow

# Modify workflow (e.g., skip automatic save)
default_workflow = MODEL_REGISTRY["caiso-irp23"]["processing_workflow"]
custom_workflow = default_workflow.copy()
custom_workflow["steps"] = [
    step for step in default_workflow["steps"]
    if step["name"] != "save_network"
]

# Run with custom workflow
network, setup_summary = run_model_workflow(
    "caiso-irp23",
    workflow_overrides=custom_workflow,
    demand_assignment_strategy="aggregate_node"
)

# Save with custom filename
network.export_to_netcdf("my_custom_output.nc")
```

## Supported Models

The following PLEXOS models are supported or in development:

| Model Name | Source | Status | Download |
|------------|--------|--------|----------|
| AEMO 2024 ISP - Green Energy Exports | AEMO | 🔴 Not yet converted | [Download](https://aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-model.zip) |
| AEMO 2024 ISP - Progressive Change | AEMO | 🟡 In-progress | [Download](https://aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-model.zip) |
| AEMO 2024 ISP - Step Change | AEMO | 🔴 Not yet converted | [Download](https://aemo.com.au/-/media/files/major-publications/isp/2024/supporting-materials/2024-isp-model.zip) |
| CAISO IRP 2023 Stochastic (25 MMT) | CAISO | 🟡 In-progress | [Download](https://www.caiso.com/documents/caiso-irp23-stochastic-2024-0517.zip) |
| CAISO 2025 Summer Assessment | CAISO | 🔴 Not yet converted | [Download](https://www.caiso.com/documents/2025-summer-loads-and-resources-assessment-public-stochastic-model.zip) |
| NREL Extended IEEE 118-bus | NREL | 🔴 Not yet converted | [Download](https://db.bettergrids.org/bettergrids/handle/1001/120) |
| SEM 2024-2032 Validation Model | SEM | 🟡 In-progress | [Download](https://www.semcommittee.com/publications/sem-25-010-sem-plexos-model-validation-2024-2032-and-backcast-report) |
| European Power & Gas Model | UCC | 🟡 In-progress | [Download](https://www.dropbox.com/scl/fi/biv5n52x8s5pxeh06u2b1/EU-Power-Gas-Model.zip) |
| PLEXOS-World 2015 Gold V1.1 | UCC | 🔴 Not yet converted | [Download](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CBYXBY) |
| PLEXOS-World Spatial Resolution | UCC | 🔴 Not yet converted | [Download](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NY1QW0) |
| MESSAGEix-GLOBIOM-EN-NPi2020-500-Soft-Link | UCC | 🟡 In-progress | [Download](https://github.com/DuncanDotPY/MESSAGEix-GLOBIOM-EN-NPi2020-500-Soft-Link) |

**Status Legend:**
- 🟢 **Converted** - Model successfully converted to PyPSA network
- 🟡 **In-progress** - Conversion currently underway
- 🔴 **Not yet converted** - Planned for future conversion

### Model × Feature Coverage

![Model Coverage Heatmap](doc/visualization/image/coverage_heatmap.png)

The heatmap shows conversion status of different features across all models.

## What Gets Converted

The converter handles these PLEXOS components:

**Core Components:**
- Buses/nodes and network topology
- Generators (thermal, hydro, renewables)
- Loads with flexible assignment strategies
- Transmission lines and links
- Storage units (batteries, pumped hydro)

**Advanced Features:**
- Forced retirements and capacity additions
- Generator outages and maintenance schedules

## Architecture

The converter uses a CSV-based approach to take the PLEXOS model properties and turn them into PyPSA networks:

1. **Download** - Automatically fetch model data from source URLs
2. **CSV Export** - Convert PLEXOS XML to structured CSV files using COAD
3. **Network Creation** - Build PyPSA network from CSV data; Add outages, add slack generators, validate constraints
4. **Optimization** - Solve using PyPSA's optimization
5. **Analysis** - Generate plots and statistics with NetworkAnalyzer

## Contribute

Report bugs on [GitHub Issues](https://github.com/open-energy-transition/plexos-to-pypsa-converter/issues).
