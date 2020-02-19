# -*- coding: utf-8 -*-
"""
DCM pump Valve
"""
V4 = EpicsSignalRO("XF:18IDA-UT{Cryo:1-V4}Cmd:Opn-Sts", name="V4")
V5 = EpicsSignalRO("XF:18IDA-UT{Cryo:1-V5}Cmd:Opn-Sts", name="V5")

L3 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}PS-I", name="L3")
P5 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}P:PS-I", name="P5")

T4 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}T:LN2Exhaust-I", name="T4")
L2 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}Acc-I", name="L2")
P3 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}P:Acc-I", name="P3")

T2 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}T:LN2Rtrn-I", name="T2")
P2 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}P:LN2Rtrn-I", name="P2")

T3 = EpicsSignalRO("XF:18IDA-UT{Cryo:1-HP1}T-I", name="T3")

LNF = EpicsSignalRO("XF:18IDA-UT{Cryo:1}F:LN2-I", name="LNF")
LNT = EpicsSignalRO("XF:18IDA-UT{Cryo:1}T:LN2FM-I", name="LNT")

P4 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}P:LN2Bath-I", name="P4")
L1 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}LN2Bath-I", name="L1")

P1 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}P:LN2Sply-I", name="P1")
T1 = EpicsSignalRO("XF:18IDA-UT{Cryo:1}T:LN2Sply-I", name="T1")

BeamCurrent = EpicsSignalRO("SR:OPS-BI{DCCT:1}I:Real-I", name="BeamCurrent")
