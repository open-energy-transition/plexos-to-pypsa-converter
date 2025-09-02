#!/usr/bin/env python3
"""
Enhanced PLEXOS Constraint Porting System for PyPSA Networks

This module provides comprehensive constraint extraction, analysis, and implementation
functionality for converting PLEXOS constraints to PyPSA networks. It implements
the constraint mapping analysis developed in the constraint analysis work.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

import pypsa  # type: ignore
from plexosdb import PlexosDB, ClassEnum  # type: ignore

logger = logging.getLogger(__name__)


class ConstraintAnalyzer:
    """
    Analyzes PLEXOS constraints and classifies them for PyPSA implementation.
    """
    
    def __init__(self, db: PlexosDB):
        self.db = db
        self.constraints_data = []
        self.implementation_stats = defaultdict(int)
    
    def extract_constraints(self) -> List[Dict[str, Any]]:
        """
        Extract all constraints from PLEXOS database with detailed properties.
        
        Returns:
            List of constraint dictionaries with properties and classification
        """
        try:
            constraint_names = self.db.list_objects_by_class(ClassEnum.Constraint)
            logger.info(f"Found {len(constraint_names)} constraints in PLEXOS database")
            
            constraints = []
            for constraint_name in constraint_names:
                try:
                    # Get constraint properties
                    properties = self.db.get_object_properties(ClassEnum.Constraint, constraint_name)
                    
                    # Get constraint memberships (what objects it applies to)
                    memberships = self.db.get_memberships_system(
                        constraint_name, object_class=ClassEnum.Constraint
                    )
                    
                    # Parse properties into structured format
                    parsed_props = self._parse_properties(properties)
                    
                    # Create constraint record
                    constraint_record = {
                        'name': constraint_name,
                        'properties': parsed_props,
                        'memberships': memberships,
                        'raw_properties': properties
                    }
                    
                    constraints.append(constraint_record)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract constraint {constraint_name}: {e}")
                    continue
            
            self.constraints_data = constraints
            return constraints
            
        except AssertionError:
            logger.warning("No constraints found in PLEXOS database")
            return []
        except Exception as e:
            logger.error(f"Failed to extract constraints: {e}")
            return []
    
    def _parse_properties(self, properties: List[Dict]) -> Dict[str, Any]:
        """Parse raw constraint properties into structured format."""
        parsed = {
            'sense': None,
            'rhs_value': None,
            'coefficients': {},
            'other_properties': {}
        }
        
        for prop in properties:
            prop_name = prop.get('property', '')
            prop_value = prop.get('value')
            
            # Parse sense (-1 for <=, 0 for =, 1 for >=)
            if prop_name == 'Sense':
                parsed['sense'] = float(prop_value) if prop_value is not None else None
            
            # Parse RHS values
            elif prop_name in ['RHS', 'RHS Custom', 'RHS Day', 'RHS Year', 'RHS Hour']:
                if prop_value is not None and parsed['rhs_value'] is None:
                    parsed['rhs_value'] = float(prop_value)
            
            # Parse coefficient types
            elif 'Coefficient' in prop_name:
                parsed['coefficients'][prop_name] = float(prop_value) if prop_value is not None else 0.0
            
            # Store other properties
            else:
                parsed['other_properties'][prop_name] = prop_value
        
        return parsed
    
    def classify_constraint(self, constraint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a constraint for PyPSA implementation based on analysis framework.
        
        Returns:
            Classification with implementation approach, difficulty, and details
        """
        name = constraint['name']
        props = constraint['properties']
        memberships = constraint['memberships']
        coefficients = props.get('coefficients', {})
        
        # Determine constraint domain and mathematical type
        domain = self._classify_domain(name, coefficients, memberships)
        math_type = self._classify_mathematical_type(props.get('sense'))
        system_level = self._classify_system_level(memberships)
        
        # Determine PyPSA implementation approach
        implementation = self._classify_implementation(domain, coefficients, props, memberships)
        
        return {
            'name': name,
            'domain': domain,
            'mathematical_type': math_type,
            'system_level': system_level,
            'rhs_value': props.get('rhs_value'),
            'sense': props.get('sense'),
            'coefficients': coefficients,
            'memberships': memberships,
            'implementation': implementation
        }
    
    def _classify_domain(self, name: str, coefficients: Dict, memberships: List) -> str:
        """Classify constraint domain based on name patterns and coefficients."""
        name_lower = name.lower()
        
        # Emissions constraints
        if any(term in name_lower for term in ['emission', 'co2', 'carbon', 'budget']):
            return 'Emissions'
        
        # Generation constraints
        if any(term in name_lower for term in ['gen', 'generator', 'capacity']) or 'Generation Coefficient' in coefficients:
            return 'Generation'
        
        # Transmission constraints
        if (any(term in name_lower for term in ['line', 'flow', 'export', 'import', 'marinus', 'basslink']) or
            any(coef in coefficients for coef in ['Flow Coefficient', 'Flow Forward Coefficient', 'Flow Back Coefficient'])):
            return 'Transmission'
        
        # Storage constraints
        if any(term in name_lower for term in ['storage', 'battery', 'hydro', 'pump']):
            return 'Storage'
        
        # Reserves constraints
        if any(term in name_lower for term in ['reserve', 'spinning', 'ancillary']):
            return 'Reserves'
        
        # Fuel constraints
        if any(term in name_lower for term in ['fuel', 'gas', 'coal']):
            return 'Fuel'
        
        # Demand constraints
        if any(term in name_lower for term in ['load', 'demand']):
            return 'Demand'
        
        return 'Unknown'
    
    def _classify_mathematical_type(self, sense: Optional[float]) -> str:
        """Classify mathematical constraint type based on sense."""
        if sense is None:
            return 'Unknown'
        elif sense == -1.0:
            return 'Inequality (<=)'
        elif sense == 0.0:
            return 'Equality'
        elif sense == 1.0:
            return 'Inequality (>=)'
        else:
            return 'Unknown'
    
    def _classify_system_level(self, memberships: List) -> str:
        """Classify system level based on memberships."""
        if len(memberships) == 0:
            return 'System-wide'
        elif len(memberships) == 1:
            return 'Component-level'
        else:
            return 'Regional'
    
    def _classify_implementation(self, domain: str, coefficients: Dict, props: Dict, memberships: List) -> Dict[str, Any]:
        """Classify implementation approach for PyPSA."""
        
        rhs_value = props.get('rhs_value')
        
        # Easy implementations (direct PyPSA support)
        if domain == 'Emissions' and rhs_value is not None and rhs_value > 0:
            return {
                'method': 'Global Constraint',
                'difficulty': 'Easy',
                'component': 'network.add_global_constraint()',
                'notes': 'Direct emissions budget constraint implementation'
            }
        
        elif domain == 'Generation' and len(memberships) == 1 and rhs_value is not None:
            return {
                'method': 'Generator Property',
                'difficulty': 'Easy',
                'component': 'generators.p_nom_max',
                'notes': 'Direct generator capacity constraint'
            }
        
        elif domain == 'Transmission' and 'Flow Coefficient' in coefficients and len(memberships) <= 1 and rhs_value is not None:
            return {
                'method': 'Link Property',
                'difficulty': 'Easy',
                'component': 'links.p_nom',
                'notes': 'Direct transmission capacity constraint'
            }
        
        elif domain == 'Storage' and rhs_value is not None:
            return {
                'method': 'Storage Unit Property',
                'difficulty': 'Easy',
                'component': 'storage_units.p_nom',
                'notes': 'Direct storage capacity constraint'
            }
        
        # Custom implementations (medium difficulty)
        elif (domain in ['Generation', 'Transmission'] and len(memberships) > 1) or len(coefficients) > 1:
            return {
                'method': 'Custom Constraint',
                'difficulty': 'Medium',
                'component': 'extra_functionality',
                'notes': 'Multi-asset constraint requiring custom aggregation'
            }
        
        # Hard implementations
        elif domain == 'Reserves':
            return {
                'method': 'Custom Constraint',
                'difficulty': 'Hard',
                'component': 'extra_functionality',
                'notes': 'Reserve constraints not natively supported in PyPSA'
            }
        
        elif domain == 'Fuel':
            return {
                'method': 'Custom Constraint',
                'difficulty': 'Hard',
                'component': 'extra_functionality',
                'notes': 'Fuel supply constraints require custom tracking'
            }
        
        # Impossible implementations
        elif 'Capacity Built Coefficient' in coefficients or 'Units Built in Year Coefficient' in coefficients:
            return {
                'method': 'Not Implementable',
                'difficulty': 'Impossible',
                'component': 'N/A',
                'notes': 'Discrete investment decisions not supported in PyPSA'
            }
        
        elif rhs_value is None or (rhs_value == 0 and props.get('sense') == 0.0):
            return {
                'method': 'Not Implementable',
                'difficulty': 'Impossible',
                'component': 'N/A',
                'notes': 'Constraint has no clear physical meaning or missing parameters'
            }
        
        # Default: requires analysis
        else:
            return {
                'method': 'Requires Analysis',
                'difficulty': 'Medium',
                'component': 'TBD',
                'notes': f'Unknown constraint pattern requiring detailed analysis'
            }


class ConstraintPorter:
    """
    Ports implementable PLEXOS constraints to PyPSA networks.
    """
    
    def __init__(self, network: pypsa.Network, db: PlexosDB):
        self.network = network
        self.db = db
        self.analyzer = ConstraintAnalyzer(db)
        self.implementation_stats = defaultdict(int)
        self.warnings = []
    
    def port_constraints(self) -> Dict[str, Any]:
        """
        Main function to port all implementable PLEXOS constraints to PyPSA network.
        
        Returns:
            Summary statistics of constraint porting results
        """
        logger.info("Starting PLEXOS constraint porting process")
        
        # Extract and classify constraints
        constraints = self.analyzer.extract_constraints()
        classified_constraints = [self.analyzer.classify_constraint(c) for c in constraints]
        
        # Group constraints by implementation approach
        constraint_groups = defaultdict(list)
        for constraint in classified_constraints:
            method = constraint['implementation']['method']
            difficulty = constraint['implementation']['difficulty']
            constraint_groups[f"{method}_{difficulty}"].append(constraint)
        
        # Implement constraints by category
        results = {
            'total_constraints': len(classified_constraints),
            'implemented': 0,
            'skipped': 0,
            'failed': 0,
            'warnings': [],
            'by_category': {}
        }
        
        # Easy implementations
        results['implemented'] += self._implement_emissions_constraints(
            [c for c in classified_constraints if c['implementation']['method'] == 'Global Constraint']
        )
        
        results['implemented'] += self._implement_generator_constraints(
            [c for c in classified_constraints if c['implementation']['method'] == 'Generator Property']
        )
        
        results['implemented'] += self._implement_transmission_constraints(
            [c for c in classified_constraints if c['implementation']['method'] == 'Link Property']
        )
        
        results['implemented'] += self._implement_storage_constraints(
            [c for c in classified_constraints if c['implementation']['method'] == 'Storage Unit Property']
        )
        
        # Medium/Hard implementations
        custom_constraints = [c for c in classified_constraints if c['implementation']['method'] == 'Custom Constraint']
        if custom_constraints:
            logger.info(f"Found {len(custom_constraints)} custom constraints - implementing basic patterns")
            results['implemented'] += self._implement_custom_constraints(custom_constraints)
        
        # Record skipped/impossible constraints
        skipped_methods = ['Not Implementable', 'Requires Analysis']
        for constraint in classified_constraints:
            method = constraint['implementation']['method']
            if method in skipped_methods:
                results['skipped'] += 1
                self._add_warning(constraint, f"Skipped: {constraint['implementation']['notes']}")
        
        # Update results
        results['warnings'] = self.warnings
        results['by_category'] = dict(self.implementation_stats)
        
        logger.info(f"Constraint porting completed: {results['implemented']} implemented, {results['skipped']} skipped")
        return results
    
    def _implement_emissions_constraints(self, constraints: List[Dict]) -> int:
        """Implement emissions budget constraints as global constraints."""
        implemented = 0
        
        for constraint in constraints:
            try:
                name = constraint['name']
                rhs_value = constraint['rhs_value']
                
                if rhs_value is not None and rhs_value > 0:
                    # Add as global constraint with emissions attribute
                    # Use direct DataFrame assignment for compatibility
                    self.network.global_constraints.loc[name] = {
                        'type': 'primary_energy',
                        'sense': '<=',
                        'constant': rhs_value,
                        'carrier_attribute': 'co2_emissions'
                    }
                    
                    logger.info(f"✓ Implemented emissions constraint '{name}': {rhs_value} tonnes CO2")
                    implemented += 1
                    self.implementation_stats['emissions_global'] += 1
                else:
                    self._add_warning(constraint, "Emissions constraint missing valid RHS value")
                    
            except Exception as e:
                logger.error(f"Failed to implement emissions constraint '{constraint['name']}': {e}")
                self._add_warning(constraint, f"Implementation error: {e}")
        
        return implemented
    
    def _implement_generator_constraints(self, constraints: List[Dict]) -> int:
        """Implement generator capacity constraints as generator properties."""
        implemented = 0
        
        for constraint in constraints:
            try:
                name = constraint['name']
                rhs_value = constraint['rhs_value']
                memberships = constraint['memberships']
                
                if rhs_value is not None and rhs_value > 0:
                    # Apply to generator members
                    for membership in memberships:
                        if membership.get('class') == 'Generator':
                            gen_name = membership.get('name')
                            if gen_name in self.network.generators.index:
                                self.network.generators.loc[gen_name, 'p_nom_max'] = rhs_value
                                logger.info(f"✓ Applied generator constraint '{name}' to {gen_name}: {rhs_value} MW")
                                implemented += 1
                                self.implementation_stats['generator_capacity'] += 1
                            else:
                                self._add_warning(constraint, f"Generator {gen_name} not found in network")
                else:
                    self._add_warning(constraint, "Generator constraint missing valid RHS value")
                    
            except Exception as e:
                logger.error(f"Failed to implement generator constraint '{constraint['name']}': {e}")
                self._add_warning(constraint, f"Implementation error: {e}")
        
        return implemented
    
    def _implement_transmission_constraints(self, constraints: List[Dict]) -> int:
        """Implement transmission flow constraints as link properties."""
        implemented = 0
        
        for constraint in constraints:
            try:
                name = constraint['name']
                rhs_value = constraint['rhs_value']
                memberships = constraint['memberships']
                
                if rhs_value is not None and rhs_value > 0:
                    # Apply to line/link members
                    for membership in memberships:
                        member_class = membership.get('class')
                        member_name = membership.get('name')
                        
                        if member_class == 'Line' and member_name in self.network.lines.index:
                            self.network.lines.loc[member_name, 's_nom'] = rhs_value
                            logger.info(f"✓ Applied line constraint '{name}' to {member_name}: {rhs_value} MW")
                            implemented += 1
                            self.implementation_stats['transmission_line'] += 1
                            
                        elif member_name in self.network.links.index:
                            self.network.links.loc[member_name, 'p_nom'] = rhs_value
                            logger.info(f"✓ Applied link constraint '{name}' to {member_name}: {rhs_value} MW")
                            implemented += 1
                            self.implementation_stats['transmission_link'] += 1
                        else:
                            self._add_warning(constraint, f"Transmission asset {member_name} not found in network")
                else:
                    self._add_warning(constraint, "Transmission constraint missing valid RHS value")
                    
            except Exception as e:
                logger.error(f"Failed to implement transmission constraint '{constraint['name']}': {e}")
                self._add_warning(constraint, f"Implementation error: {e}")
        
        return implemented
    
    def _implement_storage_constraints(self, constraints: List[Dict]) -> int:
        """Implement storage constraints as storage unit properties."""
        implemented = 0
        
        for constraint in constraints:
            try:
                name = constraint['name']
                rhs_value = constraint['rhs_value']
                memberships = constraint['memberships']
                
                if rhs_value is not None:
                    if rhs_value == 0:
                        # Energy balance constraint
                        for membership in memberships:
                            if membership.get('class') == 'Storage':
                                storage_name = membership.get('name')
                                if storage_name in self.network.storage_units.index:
                                    self.network.storage_units.loc[storage_name, 'cyclic_state_of_charge'] = True
                                    logger.info(f"✓ Applied storage balance constraint '{name}' to {storage_name}")
                                    implemented += 1
                                    self.implementation_stats['storage_balance'] += 1
                    else:
                        # Capacity constraint
                        for membership in memberships:
                            if membership.get('class') == 'Storage':
                                storage_name = membership.get('name')
                                if storage_name in self.network.storage_units.index:
                                    self.network.storage_units.loc[storage_name, 'p_nom'] = rhs_value
                                    logger.info(f"✓ Applied storage capacity constraint '{name}' to {storage_name}: {rhs_value} MW")
                                    implemented += 1
                                    self.implementation_stats['storage_capacity'] += 1
                else:
                    self._add_warning(constraint, "Storage constraint missing RHS value")
                    
            except Exception as e:
                logger.error(f"Failed to implement storage constraint '{constraint['name']}': {e}")
                self._add_warning(constraint, f"Implementation error: {e}")
        
        return implemented
    
    def _implement_custom_constraints(self, constraints: List[Dict]) -> int:
        """Handle custom constraints that require extra_functionality (basic patterns only)."""
        implemented = 0
        
        # For now, just log custom constraints that would need extra_functionality
        for constraint in constraints:
            difficulty = constraint['implementation']['difficulty']
            if difficulty == 'Medium':
                # These could be implemented with extra_functionality
                self._add_warning(constraint, 
                    f"Custom constraint requires extra_functionality implementation: {constraint['implementation']['notes']}")
                self.implementation_stats['custom_medium'] += 1
            elif difficulty == 'Hard':
                # These are more complex
                self._add_warning(constraint, 
                    f"Hard constraint not implemented: {constraint['implementation']['notes']}")
                self.implementation_stats['custom_hard'] += 1
        
        return implemented
    
    def _add_warning(self, constraint: Dict, message: str):
        """Add warning message for constraint implementation."""
        warning = {
            'constraint_name': constraint['name'],
            'domain': constraint['domain'],
            'message': message
        }
        self.warnings.append(warning)
        logger.warning(f"⚠️  {constraint['name']}: {message}")


def port_plexos_constraints(network: pypsa.Network, db: PlexosDB, verbose: bool = True) -> Dict[str, Any]:
    """
    Main interface function to port PLEXOS constraints to PyPSA network.
    
    Parameters:
    -----------
    network : pypsa.Network
        Target PyPSA network to add constraints to
    db : PlexosDB
        Source PLEXOS database containing constraints
    verbose : bool
        Whether to print detailed implementation messages
        
    Returns:
    --------
    dict
        Summary of constraint porting results including statistics and warnings
    """
    
    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    porter = ConstraintPorter(network, db)
    results = porter.port_constraints()
    
    # Print summary
    if verbose:
        print(f"\n📊 PLEXOS CONSTRAINT PORTING SUMMARY:")
        print(f"Total constraints analyzed: {results['total_constraints']}")
        print(f"Successfully implemented: {results['implemented']}")
        print(f"Skipped (impossible/analysis needed): {results['skipped']}")
        print(f"Implementation failed: {results['failed']}")
        
        if results['by_category']:
            print(f"\nBy Category:")
            for category, count in results['by_category'].items():
                print(f"  {category}: {count}")
        
        if results['warnings']:
            print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
            for warning in results['warnings'][:10]:  # Show first 10 warnings
                print(f"  {warning['constraint_name']}: {warning['message']}")
            if len(results['warnings']) > 10:
                print(f"  ... and {len(results['warnings']) - 10} more warnings")
    
    return results


# For backward compatibility with existing constraints.py
def add_constraints(network: pypsa.Network, db: PlexosDB) -> None:
    """
    Legacy function to maintain compatibility with existing code.
    This now uses the enhanced constraint porting system.
    """
    logger.info("Using enhanced constraint porting system (legacy interface)")
    results = port_plexos_constraints(network, db, verbose=True)
    
    if results['implemented'] > 0:
        logger.info(f"Added {results['implemented']} constraints to network")
    else:
        logger.warning("No constraints were successfully implemented")