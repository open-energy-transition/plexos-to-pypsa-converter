def find_bus_for_generator(db, object_name, object_class):
    """
    Finds the associated bus (Node) for a given object (e.g., Generator).

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

    for m in memberships:
        if m.get("class") == "Node":
            return m.get("name")

    # Try fallback: search memberships' collection name just in case
    for m in memberships:
        if "node" in m.get("collection_name", "").lower():
            return m.get("name")

    return None
