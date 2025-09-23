# PLEXOS-to-PyPSA converter

A Python tool for converting PLEXOS input XML files and their associated data to create equivalent PyPSA networks.

## Installation

```bash
# Clone the repository
git clone https://github.com/open-energy-transition/plexos-to-pypsa-converter.git
cd plexos-to-pypsa-converter

# Install the package
pip install -e .
```

## Converted Features

The converter ports the following PLEXOS components to PyPSA:

### Core Network Components
- **Buses/Nodes** - Network topology and zonal structure
- **Snapshots** - Time index for optimization periods
- **Carriers** - Energy types (electricity, gas, hydrogen, etc.)
- **Loads** - Demand assignment with flexible strategies

### Generation Assets
- **Conventional Generators** - Thermal, nuclear, hydro plants
  - Capacity ratings and availability
  - Efficiency curves and heat rates
  - Capital and marginal costs
  - Fuel constraints and emissions
- **Renewable Generators** - Solar, wind, other VRE
  - Capacity factors from VRE profiles
  - Technology-specific parameters

### Network Infrastructure
- **Transmission Lines** - AC lines with flow limits
- **Links** - DC connections and cross-sector links
- **Transformers** - Voltage level connections

### Storage Systems
- **Battery Storage** - Electrochemical storage
- **Pumped Hydro Storage** - Pumped storage plants
- **Generic Storage** - Other storage technologies
  - Charging/discharging efficiencies
  - Energy and power constraints

### Multi-Sector Components
- **Gas Network** - Gas buses, pipelines, and storage
- **Hydrogen Systems** - H2 production, storage, and demand
- **Transport Sector** - Electric vehicle integration

### System Constraints
- **Generation Constraints** - Unit commitment constraints
- **Transmission Constraints** - Flow and capacity limits
- **Policy Constraints** - Renewable targets, emissions limits

## Input Files & Data Structure

### Required Files

**PLEXOS XML File**
- Main model structure containing system topology, generators, buses, lines
- Contains object definitions and property mappings

**CSV Data Files**

Different data sources provide input CSV files in various formats, so we have built this converter to try and detect as many possible types/formats as possible, but note some customization might be needed.

The converter supports multiple CSV formats for time-series data:

#### Demand Data
1. **Directory Format** (AEMO-style):
   ```
   demand/
   ├── Bus_001.csv
   ├── Bus_002.csv
   └── Bus_003.csv
   ```
   Each file contains time-series demand for one bus.

2. **Single CSV Format** (CAISO/SEM-style):
   ```
   demand.csv:
   Datetime,1,2,3,Iteration
   2024-01-01 00:00,100,150,200,1
   2024-01-01 01:00,110,160,210,1
   ```
   Single file with columns for each zone/bus, supports iterations for stochastic modeling.

#### VRE Profiles
```
vre_profiles/
├── solar_zone1.csv
├── wind_zone1.csv
└── wind_offshore.csv
```
Renewable generation capacity factors (0-1) by technology and location.

#### Timeslice Data
```
timeslice.csv:
Timeslice,Property,Value
1,Peak_Hour,1
2,Off_Peak,0.8
```
Maps time-dependent properties for generators and other assets.


## Quick Start

### Basic XML Conversion

```python
from src.network.electricity_sector import create_model_from_xml

# Convert a PLEXOS model with automatic data discovery
network = create_model_from_xml(
    xml_file_path="path/to/your/model.xml",
    demand_assignment_strategy="per_node"  # or "target_node", "aggregate_node"
)

# Save the PyPSA network
network.export_to_netcdf("output_model.nc")
```