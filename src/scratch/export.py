from pathlib import Path

from coad.COAD import COAD
from coad.export_plexos_model import get_all_objects, write_object_report

from src.db.models import INPUT_XMLS

sel_model = "plexos-world-spatial"
file_xml = INPUT_XMLS[sel_model]
save_folder = "src/data/models/coad/" + sel_model

c = COAD(file_xml)

# Get all system names
system_names = c.list("System")


def write_readme(system_save_folder, name_of_system, sel_model, file_xml):
    readme_path = Path(system_save_folder) / "README.md"
    Path(system_save_folder).mkdir(parents=True, exist_ok=True)
    with readme_path.open("w") as f:
        f.write(f"""# COAD Export for System: {name_of_system}

This folder contains data exported from the COAD/PLEXOS XML model using the following script:

- Model: {sel_model}
- System: {name_of_system}
- XML file path: {Path(file_xml).resolve()}
- Export script: src/scratch/export.py

## How this data was generated

1. The COAD library was used to parse the PLEXOS XML file.
2. All objects for the system '{name_of_system}' were extracted using `get_all_objects`.
3. The data was saved using `write_object_report` to this folder.

This export was generated on {__import__("datetime").datetime.now().isoformat()}.
""")


for name_of_system in system_names:
    coad_obj = c["System"][name_of_system]
    all_objs = get_all_objects(coad_obj.coad)
    system_save_folder = Path(save_folder) / name_of_system
    write_object_report(
        coad_obj,
        interesting_objs=all_objs,
        folder=system_save_folder,
    )
    write_readme(system_save_folder, name_of_system, sel_model, file_xml)
