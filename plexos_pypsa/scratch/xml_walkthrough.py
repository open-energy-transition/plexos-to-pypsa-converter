from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum, CollectionEnum  # type: ignore

# Create a database from an XML file
file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
prog_db = PlexosDB.from_xml(file_xml)


# print all generators in the model alphabetically in a nice format
def print_objects_alphabetically(objects, object_type="object"):
    objects.sort()
    print(f"\n{object_type.capitalize()}s in prog_db:")
    for obj in objects:
        print(f"  - {obj}")


# print classes
prog_classes = prog_db.list_classes()
print(f"Available classes: {prog_classes}")

# print prog_classes alphabetically in a nice format
prog_classes = prog_db.list_classes()
prog_classes.sort()
print("\nAvailable classes in prog_db:")
for cls in prog_classes:
    print(f"  - {cls}")


# list all generators in the model
prog_generators = prog_db.list_objects_by_class(ClassEnum.Generator)
print(f"Found {len(prog_generators)} generators")
print_objects_alphabetically(prog_generators, object_type="generator")

# list all nodes in the model
prog_nodes = prog_db.list_objects_by_class(ClassEnum.Node)
print(f"Found {len(prog_nodes)} nodes")
print_objects_alphabetically(prog_nodes, object_type="node")

# list all constraints in the model
prog_constraints = prog_db.list_objects_by_class(ClassEnum.Constraint)
print(f"Found {len(prog_constraints)} constraints")
print_objects_alphabetically(prog_constraints, object_type="constraint")

# list all storage in the model
prog_storage = prog_db.list_objects_by_class(ClassEnum.Storage)
print(f"Found {len(prog_storage)} storage units")
print_objects_alphabetically(prog_storage, object_type="storage")


# function: given a generator, node, etc,
# print all properties of the generator
def print_properties(db, class_enum, name):
    properties = db.get_object_properties(class_enum, name)
    print(f"Properties of {name} ({class_enum.name}):")
    for prop in properties:
        print(f"  - {prop['property']}: {prop['value']} {prop['unit'] or ''}")
        if prop["scenario"]:
            print(f"    Scenario: {prop['scenario']}")


print_properties(prog_db, ClassEnum.Generator, "COOPGWF1")
print_properties(prog_db, ClassEnum.Node, "CNSW")
print_properties(prog_db, ClassEnum.Constraint, "Basslink Daily Energy Limit")
print_properties(prog_db, ClassEnum.Storage, "Kidston Lower")

# check valid properties for generators
gen_props = prog_db.list_valid_properties(
    CollectionEnum.Generators,
    parent_class_enum=ClassEnum.System,
    child_class_enum=ClassEnum.Generator,
)
print(f"Valid generator properties: {gen_props}")
for p in gen_props:
    print(f"  - {p}")


# check valid properties for nodes
node_props = prog_db.list_valid_properties(
    CollectionEnum.Nodes,
    parent_class_enum=ClassEnum.System,
    child_class_enum=ClassEnum.Node,
)
print(f"Valid node properties: {node_props}")
for n in node_props:
    print(f"  - {n}")


# Function to print memberships in a more readable format
def print_memberships(memberships):
    print("\nMemberships:")
    for membership in memberships:
        membership_id = membership.get("membership_id", "N/A")
        child_class_id = membership.get("child_class_id", "N/A")
        name = membership.get("name", "Unknown")
        class_name = membership.get("class", "Unknown")
        collection_id = membership.get("collection_id", "N/A")
        collection_name = membership.get("collection_name", "Unknown")
        print(f"  - Membership ID: {membership_id}")
        print(f"    Name: {name}")
        print(f"    Child Class ID: {child_class_id}")
        print(f"    Class Name: {class_name}")
        print(f"    Collection ID: {collection_id}")
        print(f"    Collection Name: {collection_name}")
        print()


# check memberships of generator COOPGWF1
mem_gen = prog_db.get_memberships_system("COOPGWF1", object_class=ClassEnum.Generator)
print_memberships(mem_gen)

# check memberships of node CNSW
mem_node = prog_db.get_memberships_system("CNSW", object_class=ClassEnum.Node)
print_memberships(mem_node)

# check memberships of constraint
mem_constraint = prog_db.get_memberships_system(
    "2050 Emissions Budget 4", object_class=ClassEnum.Constraint
)
print_memberships(mem_constraint)

# check memberships of storage
mem_storage = prog_db.get_memberships_system(
    "Kidston Lower", object_class=ClassEnum.Storage
)
print_memberships(mem_storage)


# non-comprehensive list of all classes
prog_db.list_objects_by_class(ClassEnum.System)
prog_db.list_objects_by_class(ClassEnum.Generator)
prog_db.list_objects_by_class(ClassEnum.Storage)
prog_db.list_objects_by_class(ClassEnum.Node)
prog_db.list_objects_by_class(ClassEnum.Line)
prog_db.list_objects_by_class(ClassEnum.Constraint)
prog_db.list_objects_by_class(ClassEnum.Transmission)
prog_db.list_objects_by_class(ClassEnum.MLF)

# for every constraint, list the properties and the memberships
# write this into a txt file


def write_constraints_to_file(db, filename):
    with open(filename, "w") as f:
        for constraint in db.list_objects_by_class(ClassEnum.Constraint):
            f.write(f"Constraint: {constraint}\n")
            properties = db.get_object_properties(ClassEnum.Constraint, constraint)
            for prop in properties:
                f.write(
                    f"  - {prop['property']}: {prop['value']} {prop['unit'] or ''}\n"
                )
                if prop["scenario"]:
                    f.write(f"    Scenario: {prop['scenario']}\n")
            memberships = db.get_memberships_system(
                constraint, object_class=ClassEnum.Constraint
            )
            f.write("Memberships:\n")
            for membership in memberships:
                f.write(
                    f"  - Membership ID: {membership.get('membership_id', 'N/A')}\n"
                )
                f.write(f"    Name: {membership.get('name', 'Unknown')}\n")
                f.write(
                    f"    Child Class ID: {membership.get('child_class_id', 'N/A')}\n"
                )
                f.write(f"    Class Name: {membership.get('class', 'Unknown')}\n")
                f.write(
                    f"    Collection ID: {membership.get('collection_id', 'N/A')}\n"
                )
                f.write(
                    f"    Collection Name: {membership.get('collection_name', 'Unknown')}\n"
                )
            f.write("\n")


write_constraints_to_file(prog_db, "constraints.txt")
