from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore

# Create a database from an XML file
# file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/sem/2024-2032/SEM PLEXOS Forecast Model 2024-2032( Public Version)/PUBLIC Validation 2024-2032 Model 2025-03-14.xml"

mod_db = PlexosDB.from_xml(file_xml)


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


write_constraints_to_file(mod_db, "constraints.txt")

# for all emissions, list the properties and the memberships
# write this into a txt file


def write_emissions_to_file(db, filename):
    with open(filename, "w") as f:
        for emission in db.list_objects_by_class(ClassEnum.Emission):
            f.write(f"Emission: {emission}\n")
            properties = db.get_object_properties(ClassEnum.Emission, emission)
            for prop in properties:
                f.write(
                    f"  - {prop['property']}: {prop['value']} {prop['unit'] or ''}\n"
                )
                if prop["scenario"]:
                    f.write(f"    Scenario: {prop['scenario']}\n")
            memberships = db.get_memberships_system(
                emission, object_class=ClassEnum.Emission
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


write_emissions_to_file(mod_db, "emissions.txt")
