
from collections import deque

from ophyd import (EpicsMotor, PVPositioner, PVPositionerPC,
                   EpicsSignal, EpicsSignalRO, Device)
from ophyd import Component as Cpt
from ophyd import FormattedComponent as FmtCpt
from ophyd import DynamicDeviceComponent as DDC
from ophyd import DeviceStatus, OrderedDict


class Lakeshore336Setpoint(PVPositioner):
    readback = Cpt(EpicsSignalRO, 'T-RB')
    setpoint = Cpt(EpicsSignal, 'T-SP')
    done = Cpt(EpicsSignalRO, 'Sts:Ramp-Sts')
    ramp_enabled = Cpt(EpicsSignal, 'Enbl:Ramp-Sel')
    done_value = 0


class Lakeshore336Channel(Device):
    T = Cpt(EpicsSignalRO, 'T-I')
    V = Cpt(EpicsSignalRO, 'Val:Sens-I')
    status = Cpt(EpicsSignalRO, 'T-Sts')


def _temp_fields(chans, **kwargs):
    defn = OrderedDict()
    for c in chans:
        suffix = '-Chan:{}}}'.format(c)
        defn[c] = (Lakeshore336Channel, suffix, kwargs)

    return defn


class Lakeshore336(Device):
    temp = DDC(_temp_fields(['A','B','C','D']))
    out1 = Cpt(Lakeshore336Setpoint, '-Out:1}')
    out2 = Cpt(Lakeshore336Setpoint, '-Out:2}')
    out3 = Cpt(Lakeshore336Setpoint, '-Out:3}')
    out4 = Cpt(Lakeshore336Setpoint, '-Out:4}')


lakeshore336 = Lakeshore336('XF:18ID-ES{Env:01' , name='lakeshore336')
