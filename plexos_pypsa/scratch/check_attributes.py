import datetime

from plexos_pypsa.db.models import INPUT_XMLS
from plexos_pypsa.db.plexosdb import PlexosDB  # type: ignore


def parse_excel_date(serial_date):
    """
    Convert Excel serial date to Python datetime.
    Excel serial dates count days since January 1, 1900.
    Note: Excel incorrectly treats 1900 as a leap year, so we need to adjust.
    """
    try:
        # Excel epoch is January 1, 1900 (but Excel has a bug treating 1900 as leap year)
        # So we use January 1, 1900 and subtract 2 days to account for the bug
        excel_epoch = datetime.datetime(1899, 12, 30)  # Adjusted epoch
        return excel_epoch + datetime.timedelta(days=float(serial_date))
    except (ValueError, TypeError):
        return f"Invalid date: {serial_date}"


file_xml = INPUT_XMLS["plexos-world-2015"]

# load PlexosDB from XML file
mod_db = PlexosDB.from_xml(file_xml)


def analyze_plexos_attributes(mod_db, target_class_names):
    print(f"Target class names: {target_class_names}")

    # Step 0: Find the actual class IDs for these class names
    print("\nStep 0: Finding class IDs for target class names...")
    class_names_str = "', '".join(target_class_names)
    query_class_ids = f"""
        SELECT class_id, name
        FROM t_class
        WHERE name IN ('{class_names_str}')
        ORDER BY name
    """

    print(f"Query: {query_class_ids}")
    class_rows = mod_db.query(query_class_ids)

    target_classes = {}
    found_class_names = []
    for row in class_rows:
        class_id, class_name = row
        target_classes[class_id] = class_name
        found_class_names.append(class_name)
        print(f"  - Found: {class_name} -> class_id {class_id}")

    # Check for missing classes
    missing_classes = set(target_class_names) - set(found_class_names)
    if missing_classes:
        print(f"\nWarning: The following class names were not found: {missing_classes}")

    if not target_classes:
        print("Error: No target classes found in the database!")
        return

    print(f"\nFinal target classes mapping: {target_classes}")

    # Step 1: Search for all attribute_ids with the specified class_ids
    print("\nStep 1: Searching for attributes with target class IDs...")
    class_ids_str = ", ".join(str(cid) for cid in target_classes.keys())

    query_attributes = f"""
        SELECT a.attribute_id, a.class_id, a.name as attribute_name, c.name as class_name
        FROM t_attribute a
        JOIN t_class c ON a.class_id = c.class_id
        WHERE a.class_id IN ({class_ids_str})
        ORDER BY a.class_id, a.name
    """

    print(f"Query: {query_attributes}")
    attribute_rows = mod_db.query(query_attributes)

    print(f"\nFound {len(attribute_rows)} attributes:")
    attribute_ids = []
    for row in attribute_rows:
        attribute_id, class_id, attr_name, class_name = row
        attribute_ids.append(attribute_id)
        print(
            f"  - Attribute ID {attribute_id}: {class_name}.{attr_name} (class_id: {class_id})"
        )

    print(f"\nTotal unique attribute IDs: {len(set(attribute_ids))}")

    # Step 2: Search for all attribute data for these attribute_ids
    if attribute_ids:
        print("\nStep 2: Searching for attribute data...")
        attribute_ids_str = ", ".join(str(aid) for aid in set(attribute_ids))

        query_attribute_data = f"""
            SELECT ad.attribute_id, ad.object_id, ad.value,
                   a.name as attribute_name, c.name as class_name, o.name as object_name
            FROM t_attribute_data ad
            JOIN t_attribute a ON ad.attribute_id = a.attribute_id
            JOIN t_class c ON a.class_id = c.class_id
            JOIN t_object o ON ad.object_id = o.object_id
            WHERE ad.attribute_id IN ({attribute_ids_str})
            ORDER BY ad.attribute_id, ad.object_id
        """

        print(f"Query: {query_attribute_data}")
        attribute_data_rows = mod_db.query(query_attribute_data)

        print(f"\nFound {len(attribute_data_rows)} attribute data entries:")
        for i, row in enumerate(attribute_data_rows):
            (
                attr_id,
                obj_id,
                value,
                attr_name,
                class_name,
                obj_name,
            ) = row

            # Special handling for date fields
            display_value = value
            if (
                attr_name in ["Date From", "Date To", "Chrono Date From"]
                and value
                and str(value).replace(".", "").isdigit()
            ):
                try:
                    parsed_date = parse_excel_date(float(value))
                    display_value = f"{value} ({parsed_date})"
                except Exception:
                    display_value = f"{value} (date parse failed)"

            print(
                f"  {i + 1:3d}. {attr_id} | {class_name}.{attr_name} | Object: {obj_name} | Value: {display_value}"
            )

        # Get count of total attribute data entries
        query_count = f"""
            SELECT COUNT(*) as total_count
            FROM t_attribute_data
            WHERE attribute_id IN ({attribute_ids_str})
        """
        count_result = mod_db.query(query_count)
        total_count = count_result[0][0] if count_result else 0
        print(f"\nTotal attribute data entries: {total_count}")

        # Group by class for summary
        print("\nSummary by class:")
        for class_id, class_name in target_classes.items():
            class_attrs = [aid for aid, cid, _, _ in attribute_rows if cid == class_id]
            if class_attrs:
                class_attrs_str = ", ".join(str(aid) for aid in class_attrs)
                query_class_count = f"""
                    SELECT COUNT(*) as class_count
                    FROM t_attribute_data
                    WHERE attribute_id IN ({class_attrs_str})
                """
                class_count_result = mod_db.query(query_class_count)
                class_count = class_count_result[0][0] if class_count_result else 0
                print(
                    f"  - {class_name} (ID {class_id}): {len(class_attrs)} attributes, {class_count} data entries"
                )
            else:
                print(f"  - {class_name} (ID {class_id}): No attributes found")

    else:
        print("No attributes found for the specified class IDs")


# Example usage:
target_class_names = [
    "Horizon",
    "Report",
    "Stochastic",
    "LT Plan",
    "PASA",
    "MT Schedule",
    "ST Schedule",
    "Transmission",
    "Production",
    "Competition",
    "Performance",
    "Scenario",
]
analyze_plexos_attributes(mod_db, target_class_names)

# Additional analysis: Show all available classes for reference
print("\nAll available classes in this XML file:")
query_all_classes = """
    SELECT class_id, name
    FROM t_class
    ORDER BY class_id
"""
all_classes = mod_db.query(query_all_classes)
for class_id, class_name in all_classes:
    found_indicator = "✓" if class_id in target_class_names else " "
    print(f"  {found_indicator} {class_id:3d}: {class_name}")


# search for all properties (in the property table) with "participation factor" (any case) in the name
query_properties = """
    SELECT property_id, name
    FROM t_property
    WHERE LOWER(name) LIKE '%participation factor%'
    ORDER BY name
"""
print("\nSearching for properties with 'participation factor' in the name:")
participation_rows = mod_db.query(query_properties)
if participation_rows:
    for property_id, property_name in participation_rows:
        print(f"  - Property ID {property_id}: {property_name}")
else:
    print("  No properties found with 'participation factor' in the name.")

if participation_rows:
    participation_ids = [row[0] for row in participation_rows]
    participation_ids_str = ", ".join(str(pid) for pid in participation_ids)

    query_data = """
        SELECT d.data_id, d.property_id, d.value, p.name as property_name
        FROM t_data d
        JOIN t_property p ON d.property_id = p.property_id
        WHERE d.property_id IN (participation_ids)
        ORDER BY d.property_id, d.data_id
    """
else:
    query_data = None

data_rows = mod_db.query(query_data)
data_rows

# query t_data for all rows where the property_id is in the property_id in the participation_rows
if participation_rows:
    participation_ids = [row[0] for row in participation_rows]
    participation_ids_str = ", ".join(str(pid) for pid in participation_ids)

    query_data = f"""
        SELECT d.data_id, d.property_id, d.value, p.name as property_name
        FROM t_data d
        JOIN t_property p ON d.property_id = p.property_id
        WHERE d.property_id IN ({participation_ids_str})
        ORDER BY d.property_id, d.data_id
    """

    print("\nSearching for data entries with 'participation factor' properties:")
    data_rows = mod_db.query(query_data)
    if data_rows:
        for data_id, property_id, value, property_name in data_rows:
            print(
                f"  - Data ID {data_id}: {property_name} (ID {property_id}) | Value: {value}"
            )
    else:
        print("  No data entries found for 'participation factor' properties.")
