class PZT(Device):
    pos = Cpt(EpicsSignalRO, "GET_POSITION", kind="hinted")
    p_gain = Cpt(EpicsSignal, "GET_SERVO_PGAIN", kind="config")
    i_gain = Cpt(EpicsSignal, "GET_SERVO_IGAIN", kind="config")
    d_gain = Cpt(EpicsSignal, "GET_SERVO_DGAIN", kind="config")
    setpos = Cpt(EpicsSignal, "SET_POSITION.A", kind="normal")
    status = Cpt(EpicsSignal, "GET_SERVO_STATE", kind="normal")
    step_size = Cpt(EpicsSignal, "TWV", kind="normal")

    @property
    def bender(self):
        #        print("stop using PZT.bender")
        return "None"

    @property
    def stat(self):
        return "Enabled" if self.status.get() else "Disabled"


class PZTwForce(PZT):
    loadcell = Cpt(EpicsSignalRO, "W-I", kind="hinted")

    @property
    def bender(self):
        #        print("stop using PZT.bender")
        return self.loadcell.get()


class pzt:
    def __init__(self, pzt_prefix, name, flag=0):
        self.name = name
        self.pos = EpicsSignal(
            str(pzt_prefix) + "GET_POSITION", name=self.name + "_pos"
        )
        self.p_gain = EpicsSignal(
            str(pzt_prefix) + "GET_SERVO_PGAIN", name=self.name + "_p_gain"
        )
        self.i_gain = EpicsSignal(
            str(pzt_prefix) + "GET_SERVO_IGAIN", name=self.name + "_i_gain"
        )
        self.d_gain = EpicsSignal(
            str(pzt_prefix) + "GET_SERVO_DGAIN", name=self.name + "_d_gain"
        )
        self.setting_pv = str(pzt_prefix) + "SET_POSITION.A"
        self.setpos = EpicsSignal(self.setting_pv, name="setpos")

        self.bender = self.pzt_bender(pzt_prefix, flag)
        self.stat = self.stat(pzt_prefix)

    def pzt_bender(self, pzt_prefix, flag):
        return EpicsSignal(str(pzt_prefix) + "W-I").get() if flag else "None"

    def stat(self, pzt_prefix):
        return (
            "Enabled"
            if EpicsSignal(str(pzt_prefix) + "GET_SERVO_STATE").get()
            else "Disabled"
        )


pzt_dcm_chi2 = PZT("XF:18IDA-OP{Mono:DCM-Ax:Chi2Fine}", name="pzt_dcm_chi2")
pzt_dcm_th2 = PZT("XF:18IDA-OP{Mono:DCM-Ax:Th2Fine}", name="pzt_dcm_th2")

# duplicate those signals
pzt_dcm_th2.feedback = dcm_th2.feedback
pzt_dcm_th2.feedback_enable = dcm_th2.feedback_enable

pzt_dcm_chi2.feedback = dcm_chi2.feedback
pzt_dcm_chi2.feedback_enable = dcm_chi2.feedback_enable


pzt_tm = PZTwForce("XF:18IDA-OP{Mir:TM-Ax:Bender}", name="pzt_tm")
pzt_cm = PZTwForce("XF:18IDA-OP{Mir:CM-Ax:Bender}", name="pzt_cm")

pzt_cm_loadcell = pzt_cm.loadcell
pzt_tm_loadcell = pzt_tm.loadcell

# TODO this should be fixed at the IOC level
EpicsSignal(pzt_tm.loadcell.pvname + ".PREC", name="").put(3)
EpicsSignal(pzt_cm.loadcell.pvname + ".PREC", name="").put(3)


motor_pzt = [pzt_dcm_chi2.pos, pzt_dcm_th2.pos, pzt_tm_loadcell, pzt_cm_loadcell]
