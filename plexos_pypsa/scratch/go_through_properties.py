import pandas as pd
from plexosdb.enums import ClassEnum  # type: ignore

from plexos_pypsa.db.plexosdb import PlexosDB  # type: ignore
from plexos_pypsa.db.read import list_and_print_objects


def extract_generator_properties(file_xml):
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
