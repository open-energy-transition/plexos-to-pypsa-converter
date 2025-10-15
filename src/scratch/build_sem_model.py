from collections import defaultdict

import pandas as pd
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum, CollectionEnum

from db.read import (
    check_valid_properties,
    list_and_print_objects,
    print_membership_data_entries,
    print_memberships,
    print_properties,
    save_properties,
)
from network.conversion import create_model
from utils.model_paths import get_model_xml_path

MODEL_ID = "sem-2024-2032"
SNAPSHOTS_PER_YEAR = 60

network, setup_summary = create_model(MODEL_ID)

file_xml = get_model_xml_path("sem-2024-2032")
mod_db = PlexosDB.from_xml(file_xml)


# check nodes
list_and_print_objects(mod_db, ClassEnum.Node)
network.buses

# check generators
db_gen = list_and_print_objects(mod_db, ClassEnum.Generator)
network.generators

# check length of db_gen vs network.generators
len(db_gen), len(network.generators)
