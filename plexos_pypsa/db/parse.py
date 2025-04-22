def find_bus_for_object(db, object_name, object_class):
    """
    Finds the associated bus (Node) for a given object (e.g., Generator, Storage).

    Parameters:
        db: PlexosDB instance
        object_name: Name of the object (e.g., generator name)
        object_class: ClassEnum for the object (e.g., ClassEnum.Generator)

    Returns:
        The name of the associated Node (bus), or None if not found.
    """
    try:
        memberships = db.get_memberships_system(object_name, object_class=object_class)
    except Exception as e:
        print(f"Error finding memberships for {object_name}: {e}")
        return None

    # First pass: direct relationship to Node
    for m in memberships:
        if m.get("class") == "Node":
            return m.get("name")

    # Second pass: indirect match via collection name (common for Storage)
    for m in memberships:
        if "node" in m.get("collection_name", "").lower():
            return m.get("name")

    # Third pass: check child memberships if available (less common)
    try:
        child_memberships = db.get_child_memberships(
            object_name, object_class=object_class
        )
        for cm in child_memberships:
            if cm.get("class") == "Node":
                return cm.get("name")
    except Exception as e:
        print(f"Error finding child memberships for {object_name}: {e}")

    print(f"No associated bus found for {object_name}")
    return None
