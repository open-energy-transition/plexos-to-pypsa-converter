import csv
import xml.etree.ElementTree as ET

import pandas as pd
from plexosdb.enums import ClassEnum  # type: ignore

from plexos_pypsa.db.plexosdb import PlexosDB  # type: ignore
from plexos_pypsa.db.read import list_and_print_objects


def find_collection_ids_for_class(xml_file, target_class_id):
    """
    Finds all collection_ids associated with a given class_id or child_class_id.

    Parameters
    ----------
    xml_file : str
        Path to the PLEXOS XML file.
    target_class_id : str
        The class_id or child_class_id to search for.

    Returns
    -------
    collection_ids : dict
        A dictionary mapping collection_ids to their descriptions.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the namespace from the root element
    namespace = {"ns": root.tag.split("}")[0].strip("{")}

    collection_ids = {}

    # Iterate through all <t_collection> elements with the namespace
    for collection in root.findall(".//ns:t_collection", namespace):
        child_class_id = collection.find("ns:child_class_id", namespace)
        if child_class_id is not None and child_class_id.text == target_class_id:
            collection_id = collection.find("ns:collection_id", namespace)
            description = collection.find("ns:description", namespace)
            if collection_id is not None:
                collection_ids[collection_id.text] = (
                    description.text if description is not None else ""
                )

    return collection_ids


def extract_properties_for_collections(xml_file, collection_ids):
    """
    Extracts all unique properties for the given collection_ids.

    Parameters
    ----------
    xml_file : str
        Path to the PLEXOS XML file.
    collection_ids : dict
        A dictionary mapping collection_ids to their descriptions.

    Returns
    -------
    properties : list of dict
        A list of dictionaries containing property details.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the namespace from the root element
    namespace = {"ns": root.tag.split("}")[0].strip("{")}

    properties = []

    # Iterate through all <t_property> elements with the namespace
    for prop in root.findall(".//ns:t_property", namespace):
        collection_id = prop.find("ns:collection_id", namespace)
        if collection_id is not None and collection_id.text in collection_ids:
            property_id = prop.find("ns:property_id", namespace)
            name = prop.find("ns:name", namespace)
            property_description = prop.find("ns:description", namespace)
            properties.append(
                {
                    "property_id": property_id.text if property_id is not None else "",
                    "collection_id": collection_id.text,
                    "name": name.text if name is not None else "",
                    "collection_description": collection_ids[collection_id.text],
                    "property_description": property_description.text
                    if property_description is not None
                    else "",
                }
            )

    return properties


# write function to convert properties to a dataframe, with columns:
def properties_to_dataframe(properties):
    df = pd.DataFrame(properties)
    return df


def save_properties_to_csv(properties, output_csv):
    """
    Saves the properties to a CSV file.

    Parameters
    ----------
    properties : list of dict
        A list of dictionaries containing property details.
    output_csv : str
        Path to the output CSV file.
    """
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "property_id",
                "collection_id",
                "name",
                "collection_description",
                "property_description",
            ],
        )
        writer.writeheader()
        writer.writerows(properties)


def unique_generator_properties(file_xml):
    # load PlexosDB from XML file
    plexosdb = PlexosDB.from_xml(file_xml)

    # Get all generators
    mod_generators = list_and_print_objects(plexosdb, ClassEnum.Generator, "generator")

    # Initialize an empty list to store individual generator DataFrames
    generator_dataframes = []

    # Iterate through all generators
    for generator in mod_generators:
        # Get all properties of the generator
        try:
            generator_properties = plexosdb.get_object_properties(
                ClassEnum.Generator, generator
            )
        except Exception as e:
            print(f"Skipping generator {generator} due to error: {e}")
            continue

        # Skip the generator if it has no properties
        if not generator_properties:
            print(f"Skipping generator {generator} as it has no properties.")
            continue

        # Extract unique property names
        unique_properties = {prop["property"] for prop in generator_properties}

        # Create a DataFrame for the current generator
        generator_df = pd.DataFrame(
            {
                "generator_name": [generator] * len(unique_properties),
                "property": list(unique_properties),
            }
        )

        # Append the DataFrame to the list
        generator_dataframes.append(generator_df)

    # Combine all individual DataFrames into one
    all_generators_df = pd.concat(generator_dataframes, ignore_index=True)

    # Create dataframe of just unique properties
    unique_properties_df = (
        all_generators_df[["property"]].drop_duplicates().reset_index(drop=True)
    ).sort_values(by="property")

    return all_generators_df, unique_properties_df
