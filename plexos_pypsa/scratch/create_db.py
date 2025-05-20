from coad.COAD import COAD

file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"

# this automatically creates a sqlite database from the xml file in the same directory
coad = COAD(file_xml)
