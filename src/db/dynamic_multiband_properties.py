import xml.etree.ElementTree as ET

import pandas as pd


def extract_properties(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    namespace = {
        "ns": "http://tempuri.org/MasterDataSet.xsd"
    }  # Replace with actual namespace
    properties = []

    # Build a mapping of collection_id to collection_name
    collections = {
        collection.find("ns:collection_id", namespace).text: collection.find(
            "ns:name", namespace
        ).text
        for collection in root.findall("ns:t_collection", namespace)
        if collection.find("ns:collection_id", namespace) is not None
        and collection.find("ns:name", namespace) is not None
    }

    for prop in root.findall("ns:t_property", namespace):
        # Extract property_id
        property_id = prop.find("ns:property_id", namespace)
        name = prop.find("ns:name", namespace)
        is_multi_band = prop.find("ns:is_multi_band", namespace)
        is_dynamic = prop.find("ns:is_dynamic", namespace)
        collection_id = prop.find("ns:collection_id", namespace)

        collection_id_text = collection_id.text if collection_id is not None else None

        # Match collection_id to name
        collection_name = collections.get(collection_id_text)

        properties.append(
            {
                "collection_id": collection_id_text,
                "collection_name": collection_name,
                "property_id": property_id.text if property_id is not None else None,
                "property_name": name.text if name is not None else None,
                "is_multi_band": is_multi_band.text == "true"
                if is_multi_band is not None
                else None,
                "is_dynamic": is_dynamic.text == "true"
                if is_dynamic is not None
                else None,
            }
        )

    df = pd.DataFrame(properties)
    df = df.drop_duplicates()

    df = df[
        [
            "collection_id",
            "collection_name",
            "property_id",
            "property_name",
            "is_multi_band",
            "is_dynamic",
        ]
    ]
    return df


if __name__ == "__main__":
    xml_file = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/Plexos Converter/AEMO/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
    df = extract_properties(xml_file)

    print(df)

# save df as dynamic_multiband_properties.csv
df.to_csv("src/data/scratch/dynamic_multiband_properties.csv", index=False)
