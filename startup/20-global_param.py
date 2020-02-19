import numpy as np

ZONE_PLATE = {}
ZONE_PLATE["OUT_ZONE_WIDTH"] = 30
ZONE_PLATE["ZONE_DIAMETER"] = 244
ZONE_PLATE["MANUFACTURE"] = "NanoTools"
ZONE_PLATE["EFFICIENCY"] = "Not determined"

# ZONE_DIAMETER = 200 # 200 um  ### xridia zone plate
OUT_ZONE_WIDTH = ZONE_PLATE["OUT_ZONE_WIDTH"]  # 30 nm
ZONE_DIAMETER = ZONE_PLATE["ZONE_DIAMETER"]  # new commercial zone plate
zp.wait_for_connection()
GLOBAL_VLM_MAG = 10  # vlm magnification
GLOBAL_MAG = np.round((DetU.z.position / zp.z.position - 1) * GLOBAL_VLM_MAG, 2)
CURRENT_MAG_1 = GLOBAL_MAG
CURRENT_MAG_2 = GLOBAL_MAG
CURRENT_MAG = {}
CALIBER_FLAG = 1

CALIBER = {}
