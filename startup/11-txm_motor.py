from ophyd import EpicsMotor, EpicsSignalRO, Device, Component as Cpt
from ophyd import EpicsSignal


class TwoButtonShutter(Device):
    # vendored from nslsii.devices to extend timeouts/retries
    RETRY_PERIOD = 0.5  # seconds
    MAX_ATTEMPTS = 120
    # TODO: this needs to be fixed in EPICS as these names make no sense
    # the value coming out of the PV does not match what is shown in CSS
    open_cmd = Cpt(EpicsSignal, 'Cmd:Opn-Cmd', string=True)
    open_val = 'Open'

    close_cmd = Cpt(EpicsSignal, 'Cmd:Cls-Cmd', string=True)
    close_val = 'Not Open'

    status = Cpt(EpicsSignalRO, 'Pos-Sts', string=True)
    fail_to_close = Cpt(EpicsSignalRO, 'Sts:FailCls-Sts', string=True)
    fail_to_open = Cpt(EpicsSignalRO, 'Sts:FailOpn-Sts', string=True)
    enabled_status = Cpt(EpicsSignalRO, 'Enbl-Sts', string=True)

    # user facing commands
    open_str = 'Open'
    close_str = 'Close'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_st = None

    def set(self, val):
        if self._set_st is not None:
            raise RuntimeError(f'trying to set {self.name}'
                               ' while a set is in progress')

        cmd_map = {self.open_str: self.open_cmd,
                   self.close_str: self.close_cmd}
        target_map = {self.open_str: self.open_val,
                      self.close_str: self.close_val}

        cmd_sig = cmd_map[val]
        target_val = target_map[val]

        st = DeviceStatus(self)
        if self.status.get() == target_val:
            st._finished()
            return st

        self._set_st = st
        print(self.name, val, id(st))
        enums = self.status.enum_strs

        def shutter_cb(value, timestamp, **kwargs):
            value = enums[int(value)]
            if value == target_val:
                self._set_st = None
                self.status.clear_sub(shutter_cb)
                st._finished()

        cmd_enums = cmd_sig.enum_strs
        count = 0

        def cmd_retry_cb(value, timestamp, **kwargs):
            nonlocal count
            value = cmd_enums[int(value)]
            count += 1
            if count > self.MAX_ATTEMPTS:
                cmd_sig.clear_sub(cmd_retry_cb)
                self._set_st = None
                self.status.clear_sub(shutter_cb)
                st._finished(success=False)
            if value == 'None':
                if not st.done:
                    time.sleep(self.RETRY_PERIOD)
                    cmd_sig.set(1)

                    ts = datetime.datetime.fromtimestamp(timestamp) \
                        .strftime(_time_fmtstr)
                    if count > 2:
                        msg = '** ({}) Had to reactuate shutter while {}ing'
                        print(msg.format(ts, val if val != 'Close'
                                         else val[:-1]))
                else:
                    cmd_sig.clear_sub(cmd_retry_cb)

        cmd_sig.subscribe(cmd_retry_cb, run=False)
        self.status.subscribe(shutter_cb)
        cmd_sig.set(1)

        return st


class MyBaseMotor(EpicsMotor):
    dial_readback = Cpt(EpicsSignalRO, ".DRBV")
    dial_counts = Cpt(EpicsSignalRO, ".RRBV")
    motor_res = Cpt(EpicsSignalRO, ".MRES")
    encoder_res = Cpt(EpicsSignalRO, ".ERES")
    motor_stat = Cpt(EpicsSignalRO, ".STAT")


class MyEpicsMotor(MyBaseMotor):
    def stop(self, success=False):
        self.user_setpoint.set(self.user_readback.get())
        Device.stop(self, success=success)


class Condenser(Device):
    x = Cpt(MyEpicsMotor, "{CLens:1-Ax:X}Mtr")
    y1 = Cpt(MyEpicsMotor, "{CLens:1-Ax:Y1}Mtr")
    y2 = Cpt(MyEpicsMotor, "{CLens:1-Ax:Y2}Mtr")
    z1 = Cpt(MyEpicsMotor, "{CLens:1-Ax:Z1}Mtr")
    z2 = Cpt(MyEpicsMotor, "{CLens:1-Ax:Z2}Mtr")
    p = Cpt(MyEpicsMotor, "{CLens:1-Ax:P}Mtr")


class Zoneplate(Device):
    x = Cpt(MyEpicsMotor, "{ZP:1-Ax:X}Mtr")
    y = Cpt(MyEpicsMotor, "{ZP:1-Ax:Y}Mtr")
    z = Cpt(MyBaseMotor, "{TXM-ZP:1-Ax:Z}Mtr")


class Aperture(Device):
    x = Cpt(MyEpicsMotor, "{Aper:1-Ax:X}Mtr")
    y = Cpt(MyEpicsMotor, "{Aper:1-Ax:Y}Mtr")
    z = Cpt(MyBaseMotor, "{TXM-Aper:1-Ax:Z}Mtr")


class PhaseRing(Device):
    x = Cpt(MyEpicsMotor, "{PR:1-Ax:X}Mtr")
    y = Cpt(MyEpicsMotor, "{PR:1-Ax:Y}Mtr")
    z = Cpt(MyBaseMotor, "{TXM-PH:1-Ax:Z}Mtr")


class BetrandLens(Device):
    x = Cpt(MyEpicsMotor, "{BLens:1-Ax:X}Mtr")
    y = Cpt(MyEpicsMotor, "{BLens:1-Ax:Y}Mtr")
    z = Cpt(MyBaseMotor, "{BLens:1-Ax:Z}Mtr")


class TXMSampleStage(Device):
    sx = Cpt(MyEpicsMotor, "{Env:1-Ax:Xl}Mtr")
    sy = Cpt(MyEpicsMotor, "{Env:1-Ax:Yl}Mtr")
    sz = Cpt(MyEpicsMotor, "{Env:1-Ax:Zl}Mtr")
    pi_x = Cpt(MyBaseMotor, "{TXM:1-Ax:X}Mtr")
    pi_r = Cpt(MyEpicsMotor, "{TXM:1-Ax:R}Mtr")


class DetSupport(Device):
    x = Cpt(MyEpicsMotor, "-Ax:X}Mtr")
    y = Cpt(MyEpicsMotor, "-Ax:Y}Mtr")
    z = Cpt(MyBaseMotor, "-Ax:Z}Mtr")


class TXM_SSA(Device):
    v_gap = Cpt(MyBaseMotor, "-Ax:Vgap}Mtr")
    v_ctr = Cpt(MyBaseMotor, "-Ax:Vctr}Mtr")
    h_gap = Cpt(MyBaseMotor, "-Ax:Hgap}Mtr")
    h_ctr = Cpt(MyBaseMotor, "-Ax:Hctr}Mtr")


ssa = TXM_SSA("XF:18IDB-OP{SSA:1", name="ssa")

DetU = DetSupport("XF:18IDB-OP{DetS:U", name="DetU")
DetD = DetSupport("XF:18IDB-OP{DetS:D", name="DetD")

clens = Condenser("XF:18IDB-OP", name="clens")
aper = Aperture("XF:18IDB-OP", name="aper")
zp = Zoneplate("XF:18IDB-OP", name="zp")

zp.wait_for_connection()

phase_ring = PhaseRing("XF:18IDB-OP", name="phase_ring")
betr = BetrandLens("XF:18IDB-OP", name="betr")
zps = TXMSampleStage("XF:18IDB-OP", name="zps")
XEng = MyEpicsMotor("XF:18IDA-OP{Mono:DCM-Ax:En}Mtr", name="XEng")

th2_motor = MyEpicsMotor("XF:18IDA-OP{Mono:DCM-Ax:Th2}Mtr", name="th2_motor")
th2_feedback = EpicsSignal("XF:18IDA-OP{Mono:DCM-Ax:Th2}PID", name="th2_feedback")
th2_feedback_enable = EpicsSignal(
    "XF:18IDA-OP{Mono:DCM-Ax:Th2}PID.FBON", name="th2_feedback_enable"
)

chi2_motor = MyEpicsMotor("XF:18IDA-OP{Mono:DCM-Ax:Chi2}Mtr", name="chi2_motor")
chi2_feedback = EpicsSignal("XF:18IDA-OP{Mono:DCM-Ax:Chi2}PID", name="chi2_feedback")
chi2_feedback_enable = EpicsSignal(
    "XF:18IDA-OP{Mono:DCM-Ax:Chi2}PID.FBON", name="chi2_feedback_enable"
)

shutter_open = EpicsSignal("XF:18IDA-PPS{PSh}Cmd:Opn-Cmd", name="shutter_open")
shutter_close = EpicsSignal("XF:18IDA-PPS{PSh}Cmd:Cls-Cmd", name="shutter_close")
shutter_status = EpicsSignal("XF:18IDA-PPS{PSh}Pos-Sts", name="shutter_status")

shutter = TwoButtonShutter("XF:18IDA-PPS{PSh}", name="shutter")

filter1 = EpicsSignal("XF:18IDB-UT{Fltr:1}Cmd:In-Cmd", name="filter1")
filter2 = EpicsSignal("XF:18IDB-UT{Fltr:2}Cmd:In-Cmd", name="filter2")
filter3 = EpicsSignal("XF:18IDB-UT{Fltr:3}Cmd:In-Cmd", name="filter3")
filter4 = EpicsSignal("XF:18IDB-UT{Fltr:4}Cmd:In-Cmd", name="filter4")

filters = {
    "filter1": filter1,
    "filter2": filter2,
    "filter3": filter3,
    "filter4": filter4,
}

motor_txm = [
    clens.x,
    clens.y1,
    clens.y2,
    clens.z1,
    clens.z2,
    clens.p,
    aper.x,
    aper.y,
    aper.z,
    zp.x,
    zp.y,
    zp.z,
    phase_ring.x,
    phase_ring.y,
    phase_ring.z,
    betr.x,
    betr.y,
    betr.z,
    zps.sx,
    zps.sy,
    zps.sz,
    zps.pi_x,
    zps.pi_r,
    DetU.x,
    DetU.y,
    DetU.z,
    DetD.x,
    DetD.y,
    DetD.z,
    ssa.v_gap,
    ssa.v_ctr,
    ssa.h_gap,
    ssa.h_ctr,
    XEng,
    filter1,
    filter2,
    filter3,
    filter4,
]
