from ophyd import EpicsSignalRO

sr_current = EpicsSignalRO('SR:OPS-BI{DCCT:1}I:Real-I', name='sr_current')