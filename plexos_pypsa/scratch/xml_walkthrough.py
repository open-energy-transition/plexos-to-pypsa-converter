from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum, CollectionEnum  # type: ignore

from plexos_pypsa.db.read import (
    check_valid_properties,
    list_and_print_objects,
    print_memberships,
    print_properties,
)

# list XML file
file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
# file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/sem/2024-2032/SEM PLEXOS Forecast Model 2024-2032( Public Version)/PUBLIC Validation 2024-2032 Model 2025-03-14.xml"

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

# list and print objects for various classes
mod_generators = list_and_print_objects(mod_db, ClassEnum.Generator, "generator")
mod_nodes = list_and_print_objects(mod_db, ClassEnum.Node, "node")
mod_constraints = list_and_print_objects(mod_db, ClassEnum.Constraint, "constraint")
mod_storage = list_and_print_objects(mod_db, ClassEnum.Storage, "storage")
mod_fuels = list_and_print_objects(mod_db, ClassEnum.Fuel, "fuel")
mod_emissions = list_and_print_objects(mod_db, ClassEnum.Emission, "emission")

# print properties for specific objects
print_properties(mod_db, ClassEnum.Generator, "WOOLGSF1")
print_properties(mod_db, ClassEnum.Generator, "AGLHAL05")
print_properties(mod_db, ClassEnum.Node, "CNSW")
print_properties(mod_db, ClassEnum.Constraint, "Basslink Daily Energy Limit")
print_properties(mod_db, ClassEnum.Storage, "Kidston Lower")
print_properties(mod_db, ClassEnum.Emission, "Comb Co2 NSW")
print_properties(mod_db, ClassEnum.Fuel, "ROI Oil")
print_properties(mod_db, ClassEnum.Scenario, "Three-State Start")

# Use the function for different collections and classes
check_valid_properties(
    mod_db,
    CollectionEnum.Generators,
    ClassEnum.System,
    ClassEnum.Generator,
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
mem_gen = mod_db.get_memberships_system("AGLHAL05", object_class=ClassEnum.Generator)
print_memberships(mem_gen)

# check memberships of all generators
mem_gen_all = mod_db.get_memberships_system(
    mod_generators, object_class=ClassEnum.Generator
)
print_memberships(mem_gen_all)

# check memberships of fuel Hallett
print_memberships(mod_db.get_memberships_system("Hallett", object_class=ClassEnum.Fuel))

# get properties of fuel Hallett
print_properties(mod_db, ClassEnum.Fuel, "Hallett")

# check memberships of node CNSW
mem_node = mod_db.get_memberships_system("CNSW", object_class=ClassEnum.Node)
print_memberships(mem_node)

# check memberships of constraint
mem_constraint = mod_db.get_memberships_system(
    "2050 Emissions Budget 4", object_class=ClassEnum.Constraint
)
print_memberships(mem_constraint)

# check memberships of storage
mem_storage = mod_db.get_memberships_system(
    "Kidston Lower", object_class=ClassEnum.Storage
)
print_memberships(mem_storage)

# check memberships of emission
mem_emission = mod_db.get_memberships_system("CO2", object_class=ClassEnum.Emission)
print_memberships(mem_emission)

# check memberships for all fuels
mem_fuels = mod_db.get_memberships_system(mod_fuels, object_class=ClassEnum.Fuel)
print_memberships(mem_fuels)
