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
    properties : dict
        A dictionary mapping class IDs to their properties.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the namespace from the root element
    namespace = {"ns": root.tag.split("}")[0].strip("{")}

    properties = {}

    # Iterate through all <t_attribute> elements with the namespace
    for attribute in root.findall(".//ns:t_attribute", namespace):
        class_id = attribute.find("ns:class_id", namespace)
        name = attribute.find("ns:name", namespace)

        # Ensure both class_id and name exist
        if class_id is not None and name is not None:
            class_id = class_id.text
            name = name.text

            if class_id not in properties:
                properties[class_id] = []
            properties[class_id].append(name)

    return properties


from collections import defaultdict


def inspect_plexos_xml(xml_path):
    print(f"📄 Inspecting XML file: {xml_path}\n")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"❌ Error parsing XML: {e}")
        return

    print(f"🔹 Root tag: <{root.tag}>")
    if root.attrib:
        print(f"   Attributes: {root.attrib}")
    print()

    # Count top-level tag occurrences
    tag_counts = defaultdict(int)
    samples = defaultdict(list)

    for elem in root.iter():
        tag_counts[elem.tag] += 1
        if len(samples[elem.tag]) < 3:
            samples[elem.tag].append(elem)

    print("🔍 Tag counts:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  - <{tag}>: {count}")


# Example usage
xml_file = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
properties = extract_properties(xml_file)

# Print all properties grouped by class ID
for class_id, props in properties.items():
    print(f"Class ID {class_id}:")
    for prop in props:
        print(f"  - {prop}")

inspect_plexos_xml(xml_file)


def extract_multi_valued_properties(xml_file):
    """
    Extracts all properties that can appear multiple times for a single object.

    Parameters
    ----------
    xml_file : str
        Path to the XML file.

    Returns
    -------
    multi_valued_properties : list
        A list of multi-valued properties with their details.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    namespace = {"ns": root.tag.split("}")[0].strip("{")}
    multi_valued_properties = []

    for prop in root.findall(".//ns:t_property", namespace):
        name = prop.find("ns:name", namespace).text
        is_dynamic = prop.find("ns:is_dynamic", namespace)
        is_multi_band = prop.find("ns:is_multi_band", namespace)
        collection_id = prop.find("ns:collection_id", namespace)

        if (
            (is_dynamic is not None and is_dynamic.text == "true")
            or (is_multi_band is not None and is_multi_band.text == "true")
            or (collection_id is not None)
        ):
            multi_valued_properties.append(name)

    return multi_valued_properties


# Example usage
xml_file = "/path/to/2024 ISP Progressive Change Model.xml"
multi_valued_properties = extract_multi_valued_properties(xml_file)

print("Multi-Valued Properties:")
for prop in multi_valued_properties:
    print(f"  - {prop}")
