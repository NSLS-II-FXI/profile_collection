# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:26:18 2018

Accelerator BPM
@author: xf18id
"""

bpm17_1x = EpicsSignalRO("SR:C17-BI{BPM:1}Pos:X-I", name="bpm17_1x")
bpm17_1y = EpicsSignalRO("SR:C17-BI{BPM:1}Pos:Y-I", name="bpm17_1y")

bpm17_2x = EpicsSignalRO("SR:C17-BI{BPM:2}Pos:X-I", name="bpm17_2x")
bpm17_2y = EpicsSignalRO("SR:C17-BI{BPM:2}Pos:Y-I", name="bpm17_2y")

bpm17_3x = EpicsSignalRO("SR:C17-BI{BPM:3}Pos:X-I", name="bpm17_3x")
bpm17_3y = EpicsSignalRO("SR:C17-BI{BPM:3}Pos:Y-I", name="bpm17_3y")

bpm17_4x = EpicsSignalRO("SR:C17-BI{BPM:4}Pos:X-I", name="bpm17_4x")
bpm17_4y = EpicsSignalRO("SR:C17-BI{BPM:4}Pos:Y-I", name="bpm17_4y")

bpm17_5x = EpicsSignalRO("SR:C17-BI{BPM:5}Pos:X-I", name="bpm17_5x")
bpm17_5y = EpicsSignalRO("SR:C17-BI{BPM:5}Pos:Y-I", name="bpm17_5y")

bpm17_6x = EpicsSignalRO("SR:C17-BI{BPM:6}Pos:X-I", name="bpm17_6x")
bpm17_6y = EpicsSignalRO("SR:C17-BI{BPM:6}Pos:Y-I", name="bpm17_6y")

bpm18_7x = EpicsSignalRO("SR:C18-BI{BPM:7}Pos:X-I", name="bpm18_7x")
bpm18_7y = EpicsSignalRO("SR:C18-BI{BPM:7}Pos:Y-I", name="bpm18_7y")

bpm18_8x = EpicsSignalRO("SR:C18-BI{BPM:8}Pos:X-I", name="bpm18_8x")
bpm18_8y = EpicsSignalRO("SR:C18-BI{BPM:8}Pos:Y-I", name="bpm18_8y")

bpm18_1x = EpicsSignalRO("SR:C18-BI{BPM:1}Pos:X-I", name="bpm18_1x")
bpm18_1y = EpicsSignalRO("SR:C18-BI{BPM:1}Pos:Y-I", name="bpm18_1y")

bpm18_2x = EpicsSignalRO("SR:C18-BI{BPM:2}Pos:X-I", name="bpm18_2x")
bpm18_2y = EpicsSignalRO("SR:C18-BI{BPM:2}Pos:Y-I", name="bpm18_2y")

bpm18_3x = EpicsSignalRO("SR:C18-BI{BPM:3}Pos:X-I", name="bpm18_3x")
bpm18_3y = EpicsSignalRO("SR:C18-BI{BPM:3}Pos:Y-I", name="bpm18_3y")

bpm18_4x = EpicsSignalRO("SR:C18-BI{BPM:4}Pos:X-I", name="bpm18_4x")
bpm18_4y = EpicsSignalRO("SR:C18-BI{BPM:4}Pos:Y-I", name="bpm18_4y")

bpm18_5x = EpicsSignalRO("SR:C18-BI{BPM:5}Pos:X-I", name="bpm18_5x")
bpm18_5y = EpicsSignalRO("SR:C18-BI{BPM:5}Pos:Y-I", name="bpm18_5y")

bpm18_6x = EpicsSignalRO("SR:C18-BI{BPM:6}Pos:X-I", name="bpm18_6x")
bpm18_6y = EpicsSignalRO("SR:C18-BI{BPM:6}Pos:Y-I", name="bpm18_6y")

bpm_17x = [bpm17_1x, bpm17_2x, bpm17_3x, bpm17_4x, bpm17_5x, bpm17_6x]
bpm_17y = [bpm17_1y, bpm17_2y, bpm17_3y, bpm17_4y, bpm17_5y, bpm17_6y]
bpm_18x = [
    bpm18_1x,
    bpm18_2x,
    bpm18_3x,
    bpm18_4x,
    bpm18_5x,
    bpm18_6x,
    bpm18_7x,
    bpm18_8x,
]
bpm_18y = [
    bpm18_1y,
    bpm18_2y,
    bpm18_3y,
    bpm18_4y,
    bpm18_5y,
    bpm18_6y,
    bpm18_7y,
    bpm18_8y,
]

bpm_17 = bpm_17x + bpm_17y
bpm_18 = bpm_18x + bpm_18y


# bpm_17 = [bpm17_1x,bpm17_1y, bpm17_2x,bpm17_2y,bpm17_3x,bpm17_3y, bpm17_4x,bpm17_4y, bpm17_5x,bpm17_5y, bpm17_6x,bpm17_6y]
# bpm_18 = [bpm18_1x,bpm18_1y, bpm18_2x,bpm18_2y,bpm18_3x,bpm18_3y, bpm18_4x,bpm18_4y, bpm18_5x,bpm18_5y, bpm18_6x,bpm18_6y, bpm18_7x,bpm18_7y, bpm18_8x,bpm18_8y]
