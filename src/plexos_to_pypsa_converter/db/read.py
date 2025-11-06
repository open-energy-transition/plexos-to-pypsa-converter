import pandas as pd
import plexosdb as PlexosDB
from plexosdb.enums import ClassEnum


def print_objects_alphabetically(
    objects: list[str], object_type: str = "object"
) -> None:
    objects.sort()
    print(f"\n{object_type.capitalize()}s in mod_db:")
    for obj in objects:
        print(f"  - {obj}")


def list_and_print_objects(
    db: PlexosDB, class_enum: ClassEnum, object_type: str | None = None
) -> list[str]:
    objects = db.list_objects_by_class(class_enum)
    if object_type is None:
        object_type = "object"
    print(f"Found {len(objects)} {object_type}s")
    print_objects_alphabetically(objects, object_type=object_type)
    return objects


def print_properties(
    db: PlexosDB, class_enum: ClassEnum, name: str, detailed: bool = True
) -> None:
    properties = db.get_object_properties(class_enum, name)
    print(f"Properties of {name} ({class_enum.name}):")
    for prop in properties:
        print(f"  - {prop['property']}: {prop['value']} {prop['unit'] or ''}")
        if detailed:
            print(f"    Data ID: {prop['data_id']}")
            print(f"    Parent Object ID: {prop['parent_object_id']}")
            print(f"    Property ID: {prop['property_id']}")
            print(f"    Membership ID: {prop['membership_id']}")
            print(f"    Child Object ID: {prop['child_object_id']}")
            if prop["scenario"]:
                print(f"    Scenario: {prop['scenario']}")
            if prop["scenario_category"]:
                print(f"    Scenario Category: {prop['scenario_category']}")
            if prop["texts"]:
                print(f"    Texts: {prop['texts']}")
            if prop["tags"]:
                print(f"    Tags: {prop['tags']}")
            if prop["bands"]:
                print("    Bands:")
                if isinstance(prop["bands"], str):
                    print(f"        Band: {prop['bands']}")
                else:
                    for band in prop["bands"]:
                        print(
                            f"        {band['name']}: {band['value']} {band['unit'] or ''}"
                        )


def save_properties(
    db: PlexosDB, class_enum: ClassEnum, name: str, file_path: str
) -> None:
    properties = db.get_object_properties(class_enum, name)
    if not properties:
        print(f"No properties found for {name} ({class_enum.name}).")
        return

    # Prepare data for DataFrame
    data = [
        {
            "property": prop["property"],
            "value": prop["value"],
            "unit": prop["unit"] or "",
            "scenario": prop["scenario"],
            "scenario_category": prop["scenario_category"],
            "texts": prop["texts"],
            "tags": prop["tags"],
            "data_id": prop["data_id"],
            "parent_object_id": prop["parent_object_id"],
            "property_id": prop["property_id"],
            "membership_id": prop["membership_id"],
            "child_object_id": prop["child_object_id"],
            "bands": prop["bands"]
            if isinstance(prop["bands"], str)
            else "; ".join(
                [
                    f"{band['name']}: {band['value']} {band['unit'] or ''}"
                    for band in prop["bands"]
                ]
            ),
        }
        for prop in properties
    ]

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(file_path, index=False, encoding="utf-8")

    print(f"Properties of {name} ({class_enum.name}) saved to {file_path}.")


def check_valid_properties(
    db: PlexosDB,
    collection_enum: ClassEnum,
    parent_class_enum: ClassEnum,
    child_class_enum: ClassEnum,
    label: str,
) -> None:
    props = db.list_valid_properties(
        collection_enum,
        parent_class_enum=parent_class_enum,
        child_class_enum=child_class_enum,
    )
    print(f"Valid {label} properties: {props}")
    props.sort()
    for p in props:
        print(f"  - {p}")

    # Check for specific keywords in the properties
    for p in props:
        if "Rate" in p or "Emission" in p:
            print(f"  - {p} (contains 'Rate' or 'Emission')")


def print_memberships(memberships: list[dict]) -> None:
    print("\nMemberships:")
    for membership in memberships:
        membership_id = membership.get("membership_id", "N/A")
        parent_class_id = membership.get("parent_class_id", "N/A")
        parent_class_name = membership.get("parent_class_name", "Unknown")
        parent_object_id = membership.get("parent_object_id", "N/A")
        parent_object_name = membership.get("parent_object_name", "Unknown")
        collection_id = membership.get("collection_id", "N/A")
        collection_name = membership.get("collection_name", "Unknown")
        child_class_id = membership.get("child_class_id", "N/A")
        child_class_name = membership.get("child_class_name", "Unknown")
        child_object_id = membership.get("child_object_id", "N/A")
        child_object_name = membership.get("child_object_name", "Unknown")

        print(f"  - Membership ID: {membership_id}")
        print(f"    Parent Class ID: {parent_class_id}")
        print(f"    Parent Class Name: {parent_class_name}")
        print(f"    Parent Object ID: {parent_object_id}")
        print(f"    Parent Object Name: {parent_object_name}")
        print(f"    Collection ID: {collection_id}")
        print(f"    Collection Name: {collection_name}")
        print(f"    Child Class ID: {child_class_id}")
        print(f"    Child Class Name: {child_class_name}")
        print(f"    Child Object ID: {child_object_id}")
        print(f"    Child Object Name: {child_object_name}")
        print()


def format_t_data_entries(entries: list[tuple]) -> list[dict]:
    """Format <t_data> entries neatly for printing or further processing."""
    formatted_entries = []
    for entry in entries:
        formatted_entry = {
            "data_id": entry[0],
            "membership_id": entry[1],
            "property_id": entry[2],
            "value": entry[3],
            "property_name": entry[4],
            "collection_id": entry[5],
            "collection_name": entry[6],
            "property_group_id": entry[7],
            "property_group_name": entry[8],
        }
        formatted_entries.append(formatted_entry)
    return formatted_entries


def print_membership_data_entries(entries: list[tuple]) -> None:
    """Print <t_data> entries in a more readable format."""
    l_entries = format_t_data_entries(entries)

    print("\n<t_data> Entries:")
    for entry in l_entries:
        print("  - Entry:")
        print(f"    Data ID: {entry['data_id']}")
        print(f"    Membership ID: {entry['membership_id']}")
        print(f"    Property ID: {entry['property_id']}")
        print(f"    Property Name: {entry['property_name']}")
        print(f"    Collection ID: {entry['collection_id']}")
        print(f"    Collection Name: {entry['collection_name']}")
        print(f"    Property Group ID: {entry['property_group_id']}")
        print(f"    Property Group Name: {entry['property_group_name']}")
        print(f"    Value: {entry['value']}")
        print()
