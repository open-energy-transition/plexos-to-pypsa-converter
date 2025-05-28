from plexosdb.enums import ClassEnum  # type: ignore


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


def get_emission_rates(db, generator_name):
    """
    Extracts emission rate properties for a specific generator.

    Parameters:
        db: PlexosDB instance
        generator_name: name of the generator (str)

    Returns:
        Dictionary of emission properties {emission_type: rate}
    """
    emission_props = {}
    try:
        properties = db.get_object_properties(ClassEnum.Generator, generator_name)
    except Exception as e:
        print(f"Error retrieving properties for {generator_name}: {e}")
        return emission_props

    for prop in properties:
        tag = prop.get("tag", "").lower()
        if "emission" in tag or any(
            pollutant in tag for pollutant in ["co2", "so2", "nox", "Co2"]
        ):
            emission_type = prop.get("property").lower()
            emission_value = prop.get("value")
            unit = prop.get("unit")
            emission_props[emission_type] = (emission_value, unit)

    return emission_props


def find_fuel_for_generator(db, generator_name):
    """
    Finds the associated fuel for a given generator by searching its memberships.

    Parameters:
        db: PlexosDB instance
        generator_name: Name of the generator (str)

    Returns:
        The name of the associated Fuel, or None if not found.
    """
    try:
        memberships = db.get_memberships_system(
            generator_name, object_class=ClassEnum.Generator
        )
    except Exception as e:
        print(f"Error finding memberships for {generator_name}: {e}")
        return None

    for m in memberships:
        if m.get("class") == "Fuel":
            return m.get("name")
    # print(f"No associated fuel found for {generator_name}")
    return None


# gen_name = "MURRAY10"
# emissions = get_emission_rates(prog_db, gen_name)
# print(f"Emission rates for {gen_name}:")
# for k, v in emissions.items():
#     print(f"  {k}: {v[0]} {v[1]}")
