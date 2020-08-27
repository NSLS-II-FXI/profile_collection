from collections import deque

from ophyd import (
    EpicsMotor,
    PVPositioner,
    PVPositionerPC,
    EpicsSignal,
    EpicsSignalRO,
    Device,
)
from ophyd import Component as Cpt
from ophyd import FormattedComponent as FmtCpt
from ophyd import DynamicDeviceComponent as DDC
from ophyd import DeviceStatus, OrderedDict


class Lakeshore336Setpoint(PVPositioner):
    readback = Cpt(EpicsSignalRO, "T-RB")
    setpoint = Cpt(EpicsSignal, "T-SP")
    done = Cpt(EpicsSignalRO, "Sts:Ramp-Sts")
    ramp_enabled = Cpt(EpicsSignal, "Enbl:Ramp-Sel")
    ramp_rate = Cpt(EpicsSignal, "Val:Ramp-SP")
    p_gain = Cpt(EpicsSignal, "Gain:P-RB")
    i_gain = Cpt(EpicsSignal, "Gain:I-RB")
    d_gain = Cpt(EpicsSignal, "Gain:D-RB")
    done_value = 0


class Lakeshore336Channel(Device):
    T = Cpt(EpicsSignalRO, "T-I")
    V = Cpt(EpicsSignalRO, "Val:Sens-I")
    status = Cpt(EpicsSignalRO, "T-Sts")


def _temp_fields(chans, **kwargs):
    defn = OrderedDict()
    for c in chans:
        suffix = "-Chan:{}}}".format(c)
        defn[c] = (Lakeshore336Channel, suffix, kwargs)

    return defn


class Lakeshore336(Device):
    temp = DDC(_temp_fields(["A", "B", "C", "D"]))
    out1 = Cpt(Lakeshore336Setpoint, "-Out:1}")
    out2 = Cpt(Lakeshore336Setpoint, "-Out:2}")
    out3 = Cpt(Lakeshore336Setpoint, "-Out:3}")
    out4 = Cpt(Lakeshore336Setpoint, "-Out:4}")
    ChanA = Cpt(Lakeshore336Channel, "-Chan:A}")
    ChanB = Cpt(Lakeshore336Channel, "-Chan:B}")
    ChanC = Cpt(Lakeshore336Channel, "-Chan:C}")
    ChanD = Cpt(Lakeshore336Channel, "-Chan:D}")

'''
lakeshore336 = Lakeshore336("XF:18ID-ES{Env:01", name="lakeshore336")
lakeshore336.ChanA.wait_for_connection()
lakeshore336.ChanB.wait_for_connection()
lakeshore336.ChanC.wait_for_connection()
lakeshore336.ChanD.wait_for_connection()
motor_lakeshore = [
    lakeshore336.ChanC.T,
    lakeshore336.out3.setpoint,
    lakeshore336.out3.readback,
    lakeshore336.out3.p_gain,
    lakeshore336.out3.i_gain,
    lakeshore336.out3.d_gain,
    lakeshore336.out3.ramp_rate,
    lakeshore336.out3.ramp_enabled,
]
'''
motor_lakeshore = []
