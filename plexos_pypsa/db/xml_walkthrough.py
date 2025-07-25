from plexosdb.enums import ClassEnum, CollectionEnum  # type: ignore

from plexos_pypsa.db.models import INPUT_XMLS
from plexos_pypsa.db.plexosdb import PlexosDB  # type: ignore
from plexos_pypsa.db.read import (
    check_valid_properties,
    list_and_print_objects,
    print_membership_data_entries,
    print_memberships,
    print_properties,
    save_properties,
)

file_xml = INPUT_XMLS["caiso-irp23"]

# load PlexosDB from XML file
mod_db = PlexosDB.from_xml(file_xml)

# print classes
mod_classes = mod_db.list_classes()
mod_classes.sort()
print("\nAvailable classes:")
for cls in mod_classes:
    print(f"  - {cls}")

# going through classes
# mod_db.list_objects_by_class(ClassEnum.System)
# mod_db.list_objects_by_class(ClassEnum.Generator)
# mod_db.list_objects_by_class(ClassEnum.Storage)
# mod_db.list_objects_by_class(ClassEnum.Node)
# mod_db.list_objects_by_class(ClassEnum.Line)
# mod_db.list_objects_by_class(ClassEnum.Constraint)
# mod_db.list_objects_by_class(ClassEnum.Transmission)
# mod_db.list_objects_by_class(ClassEnum.Emission)
# mod_db.list_objects_by_class(ClassEnum.Fuel)
# mod_db.list_objects_by_class(ClassEnum.Scenario)
# mod_db.list_objects_by_class(ClassEnum.Load)

mod_db.list_objects_by_class(ClassEnum.STSchedule)
mod_db.list_objects_by_class(ClassEnum.MTSchedule)
mod_db.list_objects_by_class(ClassEnum.Horizon)
mod_db.list_objects_by_class(ClassEnum.Scenario)
mod_db.list_objects_by_class(ClassEnum.Region)
mod_db.list_objects_by_class(ClassEnum.Node)

mem_node = mod_db.get_memberships_system("CIPB", object_class=ClassEnum.Node)
print_memberships(mem_node)

# list and print objects for various classes
mod_nodes = list_and_print_objects(mod_db, ClassEnum.Node, "node")
mod_generators = list_and_print_objects(mod_db, ClassEnum.Generator, "generator")
mod_storage = list_and_print_objects(mod_db, ClassEnum.Storage, "storage")
mod_lines = list_and_print_objects(mod_db, ClassEnum.Line, "line")
mod_constraints = list_and_print_objects(mod_db, ClassEnum.Constraint, "constraint")
mod_emissions = list_and_print_objects(mod_db, ClassEnum.Emission, "emission")
mod_fuels = list_and_print_objects(mod_db, ClassEnum.Fuel, "fuel")
list_and_print_objects(mod_db, ClassEnum.Scenario, "scenario")
print_properties(mod_db, ClassEnum.Scenario, "Apply GB+FR HR Decline", detailed=False)

mem_scenario = mod_db.get_memberships_system(
    "Apply GB+FR HR Decline", object_class=ClassEnum.Scenario
)

print_memberships(mem_scenario)


# print properties for specific objects
print_properties(mod_db, ClassEnum.Node, "CIPB")
print_properties(mod_db, ClassEnum.Generator, "YABULU2", detailed=False)
print_properties(mod_db, ClassEnum.Generator, "CQ CCGT", detailed=False)
print_properties(mod_db, ClassEnum.Generator, "CQ CCGT", detailed=True)
print_properties(mod_db, ClassEnum.Generator, "BOGONG2", detailed=False)

print_properties(mod_db, ClassEnum.Generator, "YSWF1", detailed=False)
print_properties(mod_db, ClassEnum.Fuel, "New OCGT NSW", detailed=False)
print_properties(mod_db, ClassEnum.Fuel, "New OCGT NSW", detailed=True)

print_properties(mod_db, ClassEnum.Storage, "Kidston Lower", detailed=False)
print_properties(mod_db, ClassEnum.Line, "SESA-CSA", detailed=False)
save_properties(
    mod_db,
    ClassEnum.Line,
    "CNSW-NNSW",
    "plexos_pypsa/data/scratch/properties_line_CNSW-NNSW.csv",
)

mod_db.get_object_properties(ClassEnum.Storage, "Anthony Pieman")

print_properties(mod_db, ClassEnum.Constraint, "Basslink Daily Energy Limit")
print_properties(mod_db, ClassEnum.Storage, "Anthony Pieman")
print_properties(mod_db, ClassEnum.Generator, "SESA Biomass", detailed=False)
print_properties(mod_db, ClassEnum.Emission, "Comb Co2 NSW")
print_properties(mod_db, ClassEnum.Fuel, "ROI Oil")
print_properties(mod_db, ClassEnum.Scenario, "Three-State Start")
print_properties(mod_db, ClassEnum.Constraint, "2030 Emissions Budget 1")
print_properties(mod_db, ClassEnum.Constraint, "Snowy 2.0 - Gen_2")
print_properties(mod_db, ClassEnum.Constraint, "REZLimit_N1_North West NSW")


# Use the function for different collections and classes
check_valid_properties(
    mod_db,
    CollectionEnum.Generators,
    ClassEnum.System,
    ClassEnum.Generator,
    "generator",
)
check_valid_properties(
    mod_db,
    CollectionEnum.Nodes,
    ClassEnum.System,
    ClassEnum.Node,
    "generator",
)

check_valid_properties(
    mod_db, CollectionEnum.Emissions, ClassEnum.System, ClassEnum.Emission, "emission"
)
check_valid_properties(
    mod_db, CollectionEnum.Fuels, ClassEnum.System, ClassEnum.Fuel, "fuel"
)
check_valid_properties(
    mod_db, CollectionEnum.Nodes, ClassEnum.System, ClassEnum.Node, "node"
)


# check memberships of one generator
mem_gen = mod_db.get_memberships_system(
    "Aramara Solar Farm", object_class=ClassEnum.Generator
)
print_memberships(mem_gen)

# check memberships of all generators
mem_gen_all = mod_db.get_memberships_system(
    mod_generators, object_class=ClassEnum.Generator
)
print_memberships(mem_gen_all)


# check memberships for generator "CQ CCGT"
mem_gen = mod_db.get_memberships_system("CQ CCGT", object_class=ClassEnum.Generator)
print_memberships(mem_gen)

# check memberships of fuel Hallett
print_memberships(mod_db.get_memberships_system("Hallett", object_class=ClassEnum.Fuel))

# get properties of fuel Hallett
print_properties(mod_db, ClassEnum.Fuel, "Hallett")

# check memberships of node CNSW
mem_node = mod_db.get_memberships_system("CIPB", object_class=ClassEnum.Node)
print_memberships(mem_node)

mem_region = mod_db.get_memberships_system("SDGE", object_class=ClassEnum.Region)
print_memberships(mem_region)

print_memberships(
    mod_db.get_memberships_system("New OCGT NSW", object_class=ClassEnum.Fuel)
)

# check memberships of constraint
mem_constraint = mod_db.get_memberships_system(
    "2050 Emissions Budget 4", object_class=ClassEnum.Constraint
)
print_memberships(mem_constraint)

# check memberships of storage
mem_storage = mod_db.get_memberships_system(
    "Tantangara", object_class=ClassEnum.Storage
)
print_memberships(mem_storage)

# check membership of constraint CNSW-SNW North
mem_constraint = mod_db.get_memberships_system(
    "REZLimit_N1_North West NSW", object_class=ClassEnum.Constraint
)
print_memberships(mem_constraint)


# check memberships of emission
mem_emission = mod_db.get_memberships_system("CO2", object_class=ClassEnum.Emission)
print_memberships(mem_emission)

# check memberships for all fuels
mem_fuels = mod_db.get_memberships_system(mod_fuels, object_class=ClassEnum.Fuel)
print_memberships(mem_fuels)

# check memberships for a line
mem_line = mod_db.get_memberships_system("CNSW-NNSW", object_class=ClassEnum.Line)
print_memberships(mem_line)

# print data entries for a specific membership
mem_data = mod_db.search_membership_id(1255)
print_membership_data_entries(mem_data)

# check memberships of storage unit "Anthony Pieman"
print_memberships(
    mod_db.get_memberships_system("Anthony Pieman", object_class=ClassEnum.Storage)
)

# get properties of generator BASTYAN
print_properties(mod_db, ClassEnum.Generator, "BASTYAN", detailed=False)

# explore constraints --------
mod_constraints = list_and_print_objects(mod_db, ClassEnum.Constraint, "constraint")
print_properties(mod_db, ClassEnum.Constraint, "CNSW-SNW North")

print_properties(mod_db, ClassEnum.Constraint, "2030 Emissions Budget 1", detailed=True)
print_memberships(mod_db.search_child_object_id(1600))
print_properties(mod_db, ClassEnum.Emission, "Comb Co2 VIC", detailed=True)
print_memberships(mod_db.search_child_object_id(1165))  # Comb Co2 VIC
print_memberships(mod_db.search_child_object_id(16))

print_properties(
    mod_db, ClassEnum.Constraint, "2050 Emissions Budget NSW 1", detailed=True
)
print_memberships(mod_db.search_child_object_id(1325))

print_properties(mod_db, ClassEnum.Constraint, "Annual GPG Limit", detailed=True)
print_memberships(mod_db.search_child_object_id(1725))
print_properties(mod_db, ClassEnum.Generator, "TAS OCGT Large", detailed=True)

print_properties(mod_db, ClassEnum.Constraint, "Barron Gorge Constraint", detailed=True)
print_memberships(mod_db.search_child_object_id(1337))

print_properties(mod_db, ClassEnum.Constraint, "Group_MN1", detailed=True)
print_memberships(mod_db.search_child_object_id(1349))

print_properties(mod_db, ClassEnum.Constraint, "ResourceLimit_N5_Solar", detailed=True)
print_memberships(mod_db.search_child_object_id(1408))

print_properties(mod_db, ClassEnum.Constraint, "Snowy 2.0 - Gen_1", detailed=True)
print_memberships(mod_db.search_child_object_id(1509))

# check all memberships associated with child_object_id
obj_mem = mod_db.search_child_object_id(1340)
print_memberships(obj_mem)

# search for all memberships of a specific class 66
scenario_mem = mod_db.search_child_class_id(66)
print_memberships(scenario_mem)

# search for all memberships with parent_object_id 33399
mod_mem = mod_db.search_parent_object_id(33399)
print_memberships(mod_mem)
