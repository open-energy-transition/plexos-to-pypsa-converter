import pandas as pd

from plexos_pypsa.scratch.prop import (
    extract_properties_for_collections,
    find_collection_ids_for_class,
    save_properties_to_csv,
    unique_generator_properties,
)

# Example usage
xml_file = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
output_csv = (
    "/Users/meas/oet/plexos-pypsa/plexos_pypsa/data/scratch/generator_properties.csv"
)

# Find all collection_ids for the class (class_id = 2 for Generator)
target_class_id = "2"  # Desired class
collection_ids = find_collection_ids_for_class(xml_file, target_class_id)

# Extract all unique properties for these collection_ids
properties = extract_properties_for_collections(xml_file, collection_ids)


# Step 3: Save the properties to a CSV file
save_properties_to_csv(properties, output_csv)

print(f"Generator properties saved to {output_csv}")

# Get unique properties for generators
all_gen_df, un_gen_df = unique_generator_properties(xml_file)

# Merge un_gen_df with properties
un_gen_properties = pd.merge(
    un_gen_df,
    pd.DataFrame(properties),
    left_on="property",
    right_on="name",
    how="left",
)

# Save the merged DataFrame to a CSV file
output_csv_merged = "/Users/meas/oet/plexos-pypsa/plexos_pypsa/data/scratch/generator_properties_unique.csv"
un_gen_properties.to_csv(output_csv_merged, index=False)


# keep generators in all_gen_df that has either both Max Capacity and Firm Capacity OR both Max Capacity and Rating

all_gen_df = all_gen_df[
    (all_gen_df["property"].isin(["Max Capacity", "Firm Capacity"]))
    | (all_gen_df["property"].isin(["Max Capacity", "Rating"]))
]
