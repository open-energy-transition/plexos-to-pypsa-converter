def print_objects_alphabetically(objects, object_type="object"):
    objects.sort()
    print(f"\n{object_type.capitalize()}s in mod_db:")
    for obj in objects:
        print(f"  - {obj}")


def list_and_print_objects(db, class_enum, object_type):
    objects = db.list_objects_by_class(class_enum)
    print(f"Found {len(objects)} {object_type}s")
    print_objects_alphabetically(objects, object_type=object_type)
    return objects


def print_properties(db, class_enum, name):
    properties = db.get_object_properties(class_enum, name)
    print(f"Properties of {name} ({class_enum.name}):")
    for prop in properties:
        print(f"  - {prop['property']}: {prop['value']} {prop['unit'] or ''}")
        if prop["scenario"]:
            print(f"    Scenario: {prop['scenario']}")
        if "bands" in prop and prop["bands"]:
            print("    Bands:")
            if isinstance(prop["bands"], str):
                print(f"      - Band: {prop['bands']}")
            else:
                for band in prop["bands"]:
                    print(
                        f"      - {band['name']}: {band['value']} {band['unit'] or ''}"
                    )


def check_valid_properties(
    db, collection_enum, parent_class_enum, child_class_enum, label
):
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


def print_memberships(memberships):
    print("\nMemberships:")
    for membership in memberships:
        membership_id = membership.get("membership_id", "N/A")
        child_class_id = membership.get("child_class_id", "N/A")
        name = membership.get("name", "Unknown")
        collection_id = membership.get("collection_id", "N/A")
        class_name = membership.get("class", "Unknown")
        collection_name = membership.get("collection_name", "Unknown")
        print(f"  - Membership ID: {membership_id}")
        print(f"    Name: {name}")
        print(f"    Child Class ID: {child_class_id}")
        print(f"    Class Name: {class_name}")
        print(f"    Collection ID: {collection_id}")
        print(f"    Collection Name: {collection_name}")
        print()
