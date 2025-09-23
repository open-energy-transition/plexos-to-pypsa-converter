# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**plexos-to-pypsa-converter** is a Python tool that converts PLEXOS energy system models to PyPSA format. It reads PLEXOS XML files and creates equivalent PyPSA networks for optimization and analysis.

Key features:
- Converts PLEXOS models (generators, loads, buses, links) to PyPSA components
- Supports multiple load assignment strategies (per-node, aggregated, target node)
- Handles time series data parsing from various formats
- Includes pre-configured models for AEMO, CAISO, SEM, and NREL systems

## Development Commands

This is a Python package without explicit build/test scripts. Common development tasks:

```bash
# Run specific model examples
python -m src.model.create_model  # AEMO model
python -m src.model.caiso_irp23    # CAISO IRP23 model
python -m src.model.sem_2024       # SEM 2024 model

# Interactive development with specific models
python src/model/create_model.py
python src/model/caiso_irp23.py
python src/model/sem_2024.py
```

## Project Structure

### Core Architecture

The project follows a layered architecture:

1. **Database Layer** (`db/`): 
   - `plexosdb.py` - Main API for interacting with PLEXOS database schema using SQLite
   - `parse.py` - Functions to extract and parse properties from PLEXOS database
   - `models.py` - Contains path mappings to XML input models for different systems

2. **Network Layer** (`network/`):
   - `core.py` - Core PyPSA network setup functions (buses, snapshots, carriers, loads)  
   - `generators.py` - Generator conversion from PLEXOS to PyPSA
   - `links.py` - Link/line conversion between nodes
   - `carriers.py` - Fuel/carrier management
   - `storage.py` - Storage unit conversion (TODO)

3. **Model Layer** (`model/`):
   - Pre-configured model setups for specific systems:
   - `create_model.py` - AEMO model setup with traditional per-bus loads
   - `caiso_irp23.py` - CAISO IRP23 with demand aggregation strategy  
   - `sem_2024.py` - SEM 2024 with target node assignment strategy

### Key Components

**PlexosDB**: Main interface to PLEXOS XML data via SQLite database. Supports querying objects by class (Generator, Node, Fuel, etc.) and extracting properties.

**Network Conversion Functions**:
- `port_core_network()` - Sets up buses, snapshots, carriers, loads with flexible demand parsing
- `port_generators()` - Converts PLEXOS generators to PyPSA generators with properties
- `port_links()` - Converts PLEXOS lines to PyPSA links

**Demand Processing Strategies**:
1. **Per-node assignment** - Each demand file maps to a network bus
2. **Target node assignment** - All demand assigned to specific existing node (e.g. "SEM")  
3. **Aggregate node assignment** - Creates new node and assigns all demand to it (e.g. "CAISO_Load_Aggregate")

**Time Series Handling**: Supports multiple demand data formats including:
- Directory with individual CSV files per bus
- Single CSV with Period columns (CAISO/SEM format) 
- Iteration-based stochastic demand data

### Input Data Structure

Models expect PLEXOS XML files plus associated time series data in specific directory structures:
- Demand data: Individual CSV files per bus OR single CSV with multiple zones/iterations
- VRE profiles: Directory with renewable generation profiles
- Timeslice data: CSV mapping for time-dependent properties

The `db/models.py` file contains hardcoded paths to various input models on Google Drive.

## Model-Specific Notes

**AEMO Models**: Use traditional per-bus load assignment where each CSV file corresponds to a network bus.

**CAISO IRP23**: Uses demand aggregation - creates single aggregate node and reassigns all generators/links to it. Handles iteration-based stochastic demand format.

**SEM 2024**: Uses target node assignment - all demand assigned to existing "SEM" node while maintaining original generator/link assignments.