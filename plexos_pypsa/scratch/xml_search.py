def search_membership_id(db, membership_id):
    """Search for all <t_data> entries with the specified <membership_id> and include property details."""
    query = """
    SELECT
        d.data_id,
        d.membership_id,
        d.property_id,
        d.value,
        p.name AS property_name,
        c.collection_id,
        c.name AS collection_name,
        pg.property_group_id,
        pg.name AS property_group_name
    FROM t_data d
    JOIN t_property p ON d.property_id = p.property_id
    JOIN t_collection c ON p.collection_id = c.collection_id
    LEFT JOIN t_property_group pg ON p.property_group_id = pg.property_group_id
    WHERE d.membership_id = ?
    """
    results = db.query(query, (membership_id,))
    return results


def format_t_data_entries(entries):
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


def print_t_data_entries(entries):
    """Print <t_data> entries in a more readable format."""
    print("\n<t_data> Entries:")
    for entry in entries:
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


# if __name__ == "__main__":
#     db_path = "/path/to/your/plexos.db"  # Replace with the actual database path
#     membership_id = 1255  # Replace with the desired membership ID

#     db = PlexosDB(db_path)
#     entries = search_membership_id(db, membership_id)
#     formatted_entries = format_t_data_entries(entries)
#     print_t_data_entries(formatted_entries)
