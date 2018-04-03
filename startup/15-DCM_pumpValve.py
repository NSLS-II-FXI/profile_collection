# -*- coding: utf-8 -*-
"""
DCM pump Valve
"""
V4=EpicsSignalRO('XF:18IDA-UT{Cryo:1-V4}Cmd:Opn-Sts', name='V4')
V5=EpicsSignalRO('XF:18IDA-UT{Cryo:1-V5}Cmd:Opn-Sts', name='V5')

BeamCurrent = EpicsSignalRO('SR:OPS-BI{DCCT:1}I:Real-I', name='BeamCurrent')