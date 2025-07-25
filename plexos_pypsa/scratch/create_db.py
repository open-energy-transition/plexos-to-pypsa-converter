from coad.COAD import COAD # type: ignore
from plexos_pypsa.db.models import INPUT_XMLS
from plexosdb.enums import ClassEnum  # type: ignore
from plexos_pypsa.db.plexosdb import PlexosDB  # type: ignore

file_xml = INPUT_XMLS["plexos-world-2015"]

# this automatically creates a sqlite database from the xml file in the same directory
coad = COAD(file_xml)
coad.list("Model")
coad["Model"]["PLEXOS World 2015 Asia"].dump()


# load PlexosDB from XML file
mod_db = PlexosDB.from_xml(file_xml)

mod_db.list_objects_by_class(ClassEnum.Model)
mod_db.list_objects_by_class(ClassEnum.STSchedule)
mod_db.list_objects_by_class(ClassEnum.MTSchedule)
mod_db.list_objects_by_class(ClassEnum.Horizon)
mod_db.list_objects_by_class(ClassEnum.Scenario)
mod_db.list_objects_by_class(ClassEnum.Scenario)

[
    "Long Term",
    "Nodal",
    "PLEXOS World 2015 Africa",
    "PLEXOS World 2015 Asia",
    "PLEXOS World 2015 Europe",
    "PLEXOS World 2015 Global",
    "PLEXOS World 2015 North America",
    "PLEXOS World 2015 Oceania",
    "PLEXOS World 2015 South America",
    "Regional",
    "Reliability",
]
]
