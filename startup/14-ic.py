from ophyd import EpicsSignal

ic1=EpicsSignal('XF:18IDB-BI{i404:1}I:R1-I', name='ic1')
ic2=EpicsSignal('XF:18IDB-BI{i404:1}I:R2-I', name='ic2')
ic3=EpicsSignal('XF:18IDB-BI{i404:1}I:R3-I', name='ic3')
ic4=EpicsSignal('XF:18IDB-BI{i404:1}I:R4-I', name='ic4')

ic_rate=EpicsSignal('XF:18IDB-BI{i404:1}Cmd:GetCS-Cmd.SCAN',name='ic_rate')

Vout1 = EpicsSignal('XF:18IDB-UT{Voltage-input:1}Val', name='Vout1')
Vout2 = EpicsSignal('XF:18IDB-UT{Voltage-input:2}Val', name='Vout2')

