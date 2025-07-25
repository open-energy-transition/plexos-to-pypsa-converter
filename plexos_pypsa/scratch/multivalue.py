import csv
import xml.etree.ElementTree as ET


def extract_properties(xml_file):
    """
    Extracts all properties from the PLEXOS XML file.

    Parameters
    ----------
    xml_file : str
        Path to the PLEXOS XML file.

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
        property_details = {
            "name": prop.find("ns:name", namespace).text
            if prop.find("ns:name", namespace) is not None
            else None,
            "is_dynamic": prop.find("ns:is_dynamic", namespace).text
            if prop.find("ns:is_dynamic", namespace) is not None
            else "false",
            "is_multi_band": prop.find("ns:is_multi_band", namespace).text
            if prop.find("ns:is_multi_band", namespace) is not None
            else "false",
            "collection_id": prop.find("ns:collection_id", namespace).text
            if prop.find("ns:collection_id", namespace) is not None
            else None,
        }
        properties.append(property_details)

    return properties


def identify_multi_valued_properties(properties):
    """
    Identifies multi-valued properties from the list of properties.

    Parameters
    ----------
    properties : list of dict
        A list of dictionaries containing property details.

    Returns
    -------
    properties_with_multi_valued : list of dict
        The same list of properties with an additional "multi_valued" key.
    """
    for prop in properties:
        prop["multi_valued"] = (
            prop["is_dynamic"] == "true"
            or prop["is_multi_band"] == "true"
            or prop["collection_id"] is not None
        )
    return properties


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
                "name",
                "is_dynamic",
                "is_multi_band",
                "collection_id",
                "multi_valued",
            ],
        )
        writer.writeheader()
        writer.writerows(properties)


# Example usage
xml_file = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
output_csv = "/Users/meas/oet/plexos-pypsa/plexos_pypsa/data/scratch/properties.csv"

# Extract all properties
properties = extract_properties(xml_file)

# Identify multi-valued properties
properties_with_multi_valued = identify_multi_valued_properties(properties)

# Save to CSV
save_properties_to_csv(properties_with_multi_valued, output_csv)

print(f"Properties saved to {output_csv}")
