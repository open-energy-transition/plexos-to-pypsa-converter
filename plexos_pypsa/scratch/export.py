import os

from coad.COAD import COAD
from coad.export_plexos_model import get_all_objects, write_object_report

from plexos_pypsa.db.models import INPUT_XMLS

sel_model = "aemo-2024-green"
file_xml = INPUT_XMLS[sel_model]
save_folder = "plexos_pypsa/data/models/coad/" + sel_model

c = COAD(file_xml)

# Get all system names
system_names = c.list("System")

for name_of_system in system_names:
    coad_obj = c["System"][name_of_system]
    all_objs = get_all_objects(coad_obj.coad)
    system_save_folder = os.path.join(save_folder, name_of_system)
    write_object_report(
        coad_obj,
        interesting_objs=all_objs,
        folder=system_save_folder,
    )
