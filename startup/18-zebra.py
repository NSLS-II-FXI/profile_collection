print(f"Loading {__file__}...")

import os
import h5py
import datetime
import numpy as np
import time as ttime

from ophyd import Device, EpicsSignal, EpicsSignalRO, Signal
from ophyd import Component as Cpt
from ophyd import FormattedComponent as FC
from ophyd.utils import set_and_wait
from ophyd.status import SubscriptionStatus, WaitTimeoutError
from ophyd.areadetector.filestore_mixins import new_short_uid, resource_factory
from ophyd.sim import NullStatus
from databroker.assets.handlers import HandlerBase
from nslsii.detectors.zebra import Zebra, EpicsSignalWithRBV
from ophyd.device import Staged, RedundantStaging, OrderedDict
from bluesky.plan_stubs import abs_set


# minimum detector acquisition period in second for full frame size
# will move this definition to areaDetector.py
DET_MIN_AP = {"Andor": 0.05, "MaranaU": 0.01, "Oryx": 0.005}

# default velocity for rotating rotary stage back to starting position
# will move this definition to motors.py
ROT_BACK_VEL = 30

# Zebra register overflow constant
ZEBRA_OVERFLOW = 982080.9365713415

BIN_FACS = {
    "Andor": {0: 1, 1: 2, 2: 3, 3: 4, 4: 8},
    "MaranaU": {0: 1, 1: 2, 2: 3, 3: 4, 4: 8},
    "KinetixU": {
        0: 1,
        1: 2,
        2: 4,
    },
    "Oryx": {},
}


class ZebraPositionCaptureData(Device):
    """
    Data arrays for the Zebra position capture function and their metadata.

    ## Not all variables are needed at FXI - CD
    """

    # Data arrays
    div1 = Cpt(EpicsSignal, "PC_DIV1")
    div2 = Cpt(EpicsSignal, "PC_DIV2")
    div3 = Cpt(EpicsSignal, "PC_DIV3")
    div4 = Cpt(EpicsSignal, "PC_DIV4")
    enc1 = Cpt(EpicsSignal, "PC_ENC1")
    enc2 = Cpt(EpicsSignal, "PC_ENC2")
    enc3 = Cpt(EpicsSignal, "PC_ENC3")
    enc4 = Cpt(EpicsSignal, "PC_ENC4")
    filt1 = Cpt(EpicsSignal, "PC_FILT1")
    filt2 = Cpt(EpicsSignal, "PC_FILT2")
    filt3 = Cpt(EpicsSignal, "PC_FILT3")
    filt4 = Cpt(EpicsSignal, "PC_FILT4")
    time = Cpt(EpicsSignal, "PC_TIME")

    # Array sizes
    num_cap = Cpt(EpicsSignal, "PC_NUM_CAP")
    num_down = Cpt(EpicsSignal, "PC_NUM_DOWN")

    # BOOLs to denote arrays with data
    cap_enc1_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B0")
    cap_enc2_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B1")
    cap_enc3_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B2")
    cap_enc4_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B3")
    cap_filt1_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B4")
    cap_filt2_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B5")
    cap_div1_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B6")
    cap_div2_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B7")
    cap_div3_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B8")
    cap_div4_bool = Cpt(EpicsSignal, "PC_BIT_CAP:B9")

    def stage(self):
        super().stage()

    def unstage(self):
        super().unstage()


class ZebraPositionCapture(Device):
    """
    Signals for the position capture function of the Zebra
    """

    # Configuration settings and status PVs
    enc = Cpt(EpicsSignalWithRBV, "PC_ENC")
    egu = Cpt(EpicsSignalRO, "M1:EGU")
    dir = Cpt(EpicsSignalWithRBV, "PC_DIR")
    tspre = Cpt(EpicsSignalWithRBV, "PC_TSPRE")
    trig_source = Cpt(EpicsSignalWithRBV, "PC_ARM_SEL")
    arm = Cpt(EpicsSignal, "PC_ARM")
    disarm = Cpt(EpicsSignal, "PC_DISARM")
    armed = Cpt(EpicsSignalRO, "PC_ARM_OUT")
    gate_source = Cpt(EpicsSignalWithRBV, "PC_GATE_SEL")
    gate_start = Cpt(EpicsSignalWithRBV, "PC_GATE_START")
    gate_width = Cpt(EpicsSignalWithRBV, "PC_GATE_WID")
    gate_step = Cpt(EpicsSignalWithRBV, "PC_GATE_STEP")
    gate_num = Cpt(EpicsSignalWithRBV, "PC_GATE_NGATE")
    gate_ext = Cpt(EpicsSignalWithRBV, "PC_GATE_INP")
    gated = Cpt(EpicsSignalRO, "PC_GATE_OUT")
    pulse_source = Cpt(EpicsSignalWithRBV, "PC_PULSE_SEL")
    pulse_start = Cpt(EpicsSignalWithRBV, "PC_PULSE_START")
    pulse_width = Cpt(EpicsSignalWithRBV, "PC_PULSE_WID")
    pulse_step = Cpt(EpicsSignalWithRBV, "PC_PULSE_STEP")
    pulse_max = Cpt(EpicsSignalWithRBV, "PC_PULSE_MAX")
    pulse = Cpt(EpicsSignalRO, "PC_PULSE_OUT")
    enc_pos1_sync = Cpt(EpicsSignal, "M1:SETPOS.PROC")
    enc_pos2_sync = Cpt(EpicsSignal, "M2:SETPOS.PROC")
    enc_pos3_sync = Cpt(EpicsSignal, "M3:SETPOS.PROC")
    enc_pos4_sync = Cpt(EpicsSignal, "M4:SETPOS.PROC")
    enc_res1 = Cpt(EpicsSignal, "M1:MRES")
    enc_res2 = Cpt(EpicsSignal, "M2:MRES")
    enc_res3 = Cpt(EpicsSignal, "M3:MRES")
    enc_res4 = Cpt(EpicsSignal, "M4:MRES")
    data_in_progress = Cpt(EpicsSignalRO, "ARRAY_ACQ")
    block_state_reset = Cpt(EpicsSignal, "SYS_RESET.PROC")
    data = Cpt(ZebraPositionCaptureData, "")

    def stage(self):
        self.arm.put(1)
        super().stage()

    def unstage(self):
        self.disarm.put(1)
        self.block_state_reset.put(1)
        super().unstage()


class FXIZebraOR(Device):
    # I really appreciate the different indexing for input source
    # Thank you for that

    use1 = Cpt(EpicsSignal, "_ENA:B0")
    use2 = Cpt(EpicsSignal, "_ENA:B1")
    use3 = Cpt(EpicsSignal, "_ENA:B2")
    use4 = Cpt(EpicsSignal, "_ENA:B3")
    input_source1 = Cpt(EpicsSignal, "_INP1")
    input_source2 = Cpt(EpicsSignal, "_INP2")
    input_source3 = Cpt(EpicsSignal, "_INP3")
    input_source4 = Cpt(EpicsSignal, "_INP4")
    invert1 = Cpt(EpicsSignal, "_INV:B0")
    invert2 = Cpt(EpicsSignal, "_INV:B1")
    invert3 = Cpt(EpicsSignal, "_INV:B2")
    invert4 = Cpt(EpicsSignal, "_INV:B3")

    def stage(self):
        super().stage()

    def unstage(self):
        super().unstage()


class ZebraAND(Device):
    # I really appreciate the different indexing for input source
    # Thank you for that
    use1 = Cpt(EpicsSignal, "_ENA:B0")
    use2 = Cpt(EpicsSignal, "_ENA:B1")
    use3 = Cpt(EpicsSignal, "_ENA:B2")
    use4 = Cpt(EpicsSignal, "_ENA:B3")
    input_source1 = Cpt(EpicsSignal, "_INP1")
    input_source2 = Cpt(EpicsSignal, "_INP2")
    input_source3 = Cpt(EpicsSignal, "_INP3")
    input_source4 = Cpt(EpicsSignal, "_INP4")
    invert1 = Cpt(EpicsSignal, "_INV:B0")
    invert2 = Cpt(EpicsSignal, "_INV:B1")
    invert3 = Cpt(EpicsSignal, "_INV:B2")
    invert4 = Cpt(EpicsSignal, "_INV:B3")

    def stage(self):
        super().stage()

    def unstage(self):
        super().unstage()


class ZebraPulse(Device):
    width = Cpt(EpicsSignalWithRBV, "WID")
    input_addr = Cpt(EpicsSignalWithRBV, "INP")
    input_str = Cpt(EpicsSignalRO, "INP:STR", string=True)
    input_status = Cpt(EpicsSignalRO, "INP:STA")
    delay = Cpt(EpicsSignalWithRBV, "DLY")
    delay_sync = Cpt(EpicsSignal, "DLY:SYNC")
    time_units = Cpt(EpicsSignalWithRBV, "PRE", string=True)
    output = Cpt(EpicsSignal, "OUT")

    input_edge = FC(EpicsSignal, "{self._zebra_prefix}POLARITY:{self._edge_addr}")

    _edge_addrs = {
        1: "BC",
        2: "BD",
        3: "BE",
        4: "BF",
    }

    def __init__(
        self,
        prefix,
        *,
        index=None,
        parent=None,
        configuration_attrs=None,
        read_attrs=None,
        **kwargs,
    ):
        if read_attrs is None:
            read_attrs = ["input_addr", "input_edge", "delay", "width", "time_units"]
        if configuration_attrs is None:
            configuration_attrs = []

        zebra = parent
        self.index = index
        self._zebra_prefix = zebra.prefix
        self._edge_addr = self._edge_addrs[index]

        super().__init__(
            prefix,
            configuration_attrs=configuration_attrs,
            read_attrs=read_attrs,
            parent=parent,
            **kwargs,
        )

    def stage(self):
        super().stage()

    def unstage(self):
        super().unstage()


class FXIZebra(Zebra):
    """
    FXI Zebra device.
    """

    pc = Cpt(ZebraPositionCapture, "")
    or1 = Cpt(FXIZebraOR, "OR1")  # XF:18ID-ES:1{Dev:Zebra1}:OR1_INV:B0
    or2 = Cpt(FXIZebraOR, "OR2")
    or3 = Cpt(FXIZebraOR, "OR3")
    or4 = Cpt(FXIZebraOR, "OR4")
    and1 = Cpt(ZebraAND, "AND1")  # XF:18ID-ES:1{Dev:Zebra2}:AND1_ENA:B0
    and2 = Cpt(ZebraAND, "AND2")
    and3 = Cpt(ZebraAND, "AND3")
    and4 = Cpt(ZebraAND, "AND4")
    pulse1 = Cpt(ZebraPulse, "PULSE1_", index=1)  # XF:18ID-ES:1{Dev:Zebra1}:PULSE1_INP
    pulse2 = Cpt(ZebraPulse, "PULSE2_", index=2)
    pulse3 = Cpt(ZebraPulse, "PULSE3_", index=3)
    pulse4 = Cpt(ZebraPulse, "PULSE4_", index=4)

    def stage(self):
        super().stage()

    def unstage(self):
        super().unstage()

    def __init__(self, prefix, *, read_attrs=None, configuration_attrs=None, **kwargs):
        if read_attrs is None:
            read_attrs = []
        if configuration_attrs is None:
            configuration_attrs = []

        super().__init__(
            prefix,
            read_attrs=read_attrs,
            configuration_attrs=configuration_attrs,
            **kwargs,
        )


class ZebraSaver(Device):
    """Device for saving zebra and scaler data via IOC."""
    # Saving business logic
    write_dir = Cpt(EpicsSignal, "write_dir", string=True)
    file_name = Cpt(EpicsSignal, "file_name", string=True)
    full_file_path = Cpt(EpicsSignalRO, "full_file_path")
    acquire = Cpt(EpicsSignal, "acquire", string=True)
    file_stage = Cpt(EpicsSignal, "stage")
    dev_type = Cpt(EpicsSignal, "dev_type")

    # Zebra-related PVs
    enc1 = Cpt(EpicsSignal, "enc1")
    enc2 = Cpt(EpicsSignal, "enc2")
    enc3 = Cpt(EpicsSignal, "enc3")
    zebra_time = Cpt(EpicsSignal, "zebra_time")

    # Scaler-related PVs
    i0 = Cpt(EpicsSignal, "i0")
    im = Cpt(EpicsSignal, "im")
    it = Cpt(EpicsSignal, "it")
    sis_time = Cpt(EpicsSignal, "sis_time")


zs = ZebraSaver("XF:18ID-ES{ZebraSaver:1}:", name="zs")


## Class below requires FXI specific changes
class FXITomoFlyer(Device):
    """
    This is the flyer object for the Zebra.
    This is the position based flyer.
    """

    fast_axis = Cpt(Signal, value="PI_R", kind="config")

    KNOWN_DETS = {"Andor", "MaranaU", "KinetixU", "MaranaD", "KinetixD", "Oryx"}

    CAM_MODES_IN_FLYER = {
        "MARANA-4BV6X": {
            "trigger_mode": ["Internal", "External"],
            "image_mode": ["Continuous", "Fixed"],
            "bin_options": [0, 1, 2, 3, 4],  # => [1x1, 2x2, 3x3, 4x4, 8x8]
            "min_exp": 0.001,
        },
        "SONA-4BV6X": {
            "trigger_mode": ["Internal", "External"],
            "image_mode": ["Continuous", "Fixed"],
            "bin_options": [0, 1, 2, 3, 4],  # => [1x1, 2x2, 3x3, 4x4, 8x8]
            "min_exp": 0.001,
        },
        "KINETIX": {
            "trigger_mode": ["Internal", "Rising Edge"],
            "image_mode": ["Continuous", "Multiple"],
            "bin_options": [0, 1, 2],  # => [1x1, 2x2, 4x4]
            "min_exp": 0.0001,
        },
        "KINETIX22": {
            "trigger_mode": ["Internal", "Rising Edge"],
            "image_mode": ["Continuous", "Multiple"],
            "bin_options": [0, 1, 2],  # => [1x1, 2x2, 4x4]
            "min_exp": 0.0001,
        },
    }
    rot_axis = zps.pi_r

    scn_modes = {
        0: "standard",  # a single scan in a given angle range
        1: "snaked: multiple files",  # back-forth rocking scan with each swing being saved into a file
        2: "snaked: single file",  # back-forth rocking scan being saved into a single file
    }

    dft_pulse_wid = {"ms": 0.002, "s": 0.0005, "10s": 0.003}  # 0: ms  # 1: s  # 2: 10s
    min_exp = {
        "MARANA-4BV6X": 0.001,
        "SONA-4BV6X": 0.001,
        "KINETIX22": 0.0001,
        "Neo": 0.001,
        "Oryx": 0.001,
    }
    pc_trig_dir = {1: 0, -1: 1}  # 1: positive, -1: negative
    rot_var = 0.006
    scan_cfg = {}
    pc_cfg = {}
    _staging_delay = 0.010
    tspre = "s"  ## ['ms', 's', '10s']

    def __init__(self, dets, zebra, *, md={}, scn_mode=0, **kwargs):
        super().__init__("", parent=None, **kwargs)
        self._state = "idle"
        self._dets = dets
        self._filestore_resource = None
        self._encoder = zebra
        self._document_cache = (
            []
        )  # self._document_cache defines resource and datum documents
        self._stage_sigs = {}
        self._last_bulk = None  # self._last_bulk defines event document

        self._md = md

        self.root_path = self.root_path_str()
        self.write_path_template = f"zebra/%Y/%m/%d/"
        self.read_path_template = f"zebra/%Y/%m/%d/"
        self.reg_root = f"zebra/"

        self.scn_mode = self.scn_modes[scn_mode]
        self.extra_stage_sigs = {}
        self.shutter_delay = 0.1  # unit: deg; _shutter_delay/rot_vel > unibliz shutter opening time 1.5ms
        self.use_shutter = True

        _timeout = 10

        self._encoder.pc.block_state_reset.set(1).wait(_timeout)

        ############### Zebra Setup ###############
        ## PC Tab
        self._encoder.pc.data.cap_enc1_bool.set(1).wait(_timeout)
        self._encoder.pc.data.cap_enc2_bool.set(0).wait(_timeout)
        self._encoder.pc.data.cap_enc3_bool.set(0).wait(_timeout)
        self._encoder.pc.data.cap_enc4_bool.set(0).wait(_timeout)
        self._encoder.pc.data.cap_filt1_bool.set(0).wait(_timeout)
        self._encoder.pc.data.cap_filt2_bool.set(0).wait(_timeout)
        self._encoder.pc.data.cap_div1_bool.set(0).wait(_timeout)
        self._encoder.pc.data.cap_div2_bool.set(0).wait(_timeout)
        self._encoder.pc.data.cap_div3_bool.set(0).wait(_timeout)
        self._encoder.pc.data.cap_div4_bool.set(0).wait(_timeout)

        self._encoder.pc.enc.set(0).wait(
            _timeout
        )  # 0: Enc1, 1: Enc2, 2: Enc3, 3: Enc4,
        self._encoder.pc.dir.set(0).wait(_timeout)  # 0: Positive, 1: Negative
        self._encoder.pc.tspre.set(1).wait(_timeout)  # 0: ms, 1: s, 2: 10s

        ## AND tab -- can be used for external triggering
        self._encoder.and1.use1.set(0).wait(_timeout)  # 0: No, 1: Yes
        self._encoder.and1.use2.set(0).wait(_timeout)
        self._encoder.and1.use3.set(0).wait(_timeout)
        self._encoder.and1.use4.set(0).wait(_timeout)
        self._encoder.and1.input_source1.set(0).wait(_timeout)
        self._encoder.and1.input_source2.set(0).wait(_timeout)
        self._encoder.and1.input_source3.set(0).wait(_timeout)
        self._encoder.and1.input_source4.set(0).wait(_timeout)
        self._encoder.and1.invert1.set(0).wait(_timeout)  # 0: No, 1: Yes
        self._encoder.and1.invert2.set(0).wait(_timeout)
        self._encoder.and1.invert3.set(0).wait(_timeout)
        self._encoder.and1.invert4.set(0).wait(_timeout)

        ## OR Tab -- can be used for diagnose
        self._encoder.or1.use1.set(0).wait(_timeout)  # 0: No, 1: Yes
        self._encoder.or1.use2.set(0).wait(_timeout)
        self._encoder.or1.use3.set(0).wait(_timeout)
        self._encoder.or1.use4.set(0).wait(_timeout)
        self._encoder.or1.input_source1.set(0).wait(_timeout)
        self._encoder.or1.input_source2.set(0).wait(_timeout)
        self._encoder.or1.input_source3.set(0).wait(_timeout)
        self._encoder.or1.input_source4.set(0).wait(_timeout)
        self._encoder.or1.invert1.set(0).wait(_timeout)  # 0 = No, 1 = Yes
        self._encoder.or1.invert2.set(0).wait(_timeout)
        self._encoder.or1.invert3.set(0).wait(_timeout)
        self._encoder.or1.invert4.set(0).wait(_timeout)

        ## PULSE tab -- set for fast shutter
        self._encoder.pulse1.input_addr.set(31).wait(_timeout)
        self._encoder.pulse1.input_edge.set(0).wait(_timeout)  # 0 = rising, 1 = falling
        self._encoder.pulse1.delay.set(0).wait(_timeout)
        self._encoder.pulse2.input_addr.set(31).wait(_timeout)
        self._encoder.pulse2.input_edge.set(1).wait(_timeout)  # 0 = rising, 1 = falling
        self._encoder.pulse2.delay.set(0).wait(_timeout)

        ## ENC tab
        self._encoder.pc.enc_pos1_sync.set(1).wait(_timeout)
        self._encoder.pc.enc_pos2_sync.set(0).wait(_timeout)
        self._encoder.pc.enc_pos3_sync.set(0).wait(_timeout)
        self._encoder.pc.enc_pos4_sync.set(0).wait(_timeout)

        ## SYS tab
        self._encoder.output1.ttl.addr.set(53).wait(
            _timeout
        )  # PC_PULSE --> TTL1 --> Camera
        self._encoder.output2.ttl.addr.set(52).wait(
            _timeout
        )  # PC_PULSE --> TTL2 --> fast shutter
        self._encoder.output3.ttl.addr.set(0).wait(_timeout)
        self._encoder.output4.ttl.addr.set(0).wait(_timeout)

    @property
    def encoder(self):
        return self._encoder

    @property
    def detectors(self):
        return tuple(self._dets)

    @detectors.setter
    def detectors(self, value):
        dets = tuple(value)
        if not all(d.name in self.KNOWN_DETS for d in dets):
            raise ValueError(
                f"One or more of {[d.name for d in dets]}"
                f"is not known to the zebra. "
                f"The known detectors are {self.KNOWN_DETS})"
            )
        self._dets = dets

    @classmethod
    def _adjust_zebra_gate_start(cls, zebra_gate_start_val):
        """Workaround solution for setting buggy Zebra "Gate Start"
        Note: the workaround solution below does not work due to the register issue and
              Gate Start RBV issue. it won't be able to take a positive Gate Start value.
              Otherwise, set() would block actions after setting Gate Start.
        Issue 1: "Gate Start" cannot take a positive number; all values are offset
                 by a value ZEBRA_OVERFLOW
        Issue 2: "Gate Start" RBV is reset to another value if the value to be set
                 is smaller than -491040.4681, e.g. -491040.4682 would be reset to
                 1473121.4049, which gives np.log2((491040.4682+1473121.4049)/2.286585E-4)=33
        Workaround: limit ange range to [-491040.4681, 491040.4681].
        """
        if (-491040.4681 > zebra_gate_start_val) or (
            zebra_gate_start_val > 491040.4681
        ):
            raise ValueError(
                "Invalid Zebra Start Gate value. It should be in range [-491040.4681, 491040.4681]!"
            )
        elif -491040.4681 <= zebra_gate_start_val <= 0:
            return zebra_gate_start_val
        else:
            return 982080.9365000001 + zebra_gate_start_val

    def preset_zebra(self, pc_cfg={}):
        print(f"{pc_cfg=}")
        print(f"{self.scn_mode=}")
        ############### PC Arm
        yield from abs_set(
            self._encoder.pc.trig_source, 0, wait=True
        )  # 0 = Soft, 1 = External

        ############### PULSE -- set unibliz trigger to 'external exposure'
        if self.use_shutter:
            yield from abs_set(self._encoder.pulse1.time_units, self.tspre, wait=True)
            yield from abs_set(
                self._encoder.pulse1.width, self.dft_pulse_wid[self.tspre], wait=True
            )
            yield from abs_set(self._encoder.pulse2.time_units, self.tspre, wait=True)
            yield from abs_set(
                self._encoder.pulse2.width, self.dft_pulse_wid[self.tspre], wait=True
            )
            yield from abs_set(self._encoder.output2.ttl.addr, 52, wait=True)
        else:
            yield from abs_set(self._encoder.output2.ttl.addr, 29, wait=True)
        if self.scn_mode == "standard":
            ############### PC Tab ###############
            ## PC Gate
            yield from abs_set(
                self._encoder.pc.gate_source, 0, wait=True
            )  # 0 = Position, 1 = Time, 2 = External
            yield from abs_set(self._encoder.pc.gate_step, 0, wait=True)
            yield from abs_set(self._encoder.pc.gate_num, 1, wait=True)

            ## PC Pulse
            yield from abs_set(
                self._encoder.pc.pulse_source, 0, wait=True
            )  # 0 = Position, 1 = Time, 2 = External
            yield from abs_set(
                self._encoder.pc.pulse_width, self.dft_pulse_wid[self.tspre], wait=True
            )
        elif self.scn_mode == "snaked: multiple files":
            ############### PC Tab ###############
            ## PC Gate
            yield from abs_set(
                self._encoder.pc.gate_source, 0, wait=True
            )  # 0 = Position, 1 = Time, 2 = External
            yield from abs_set(self._encoder.pc.gate_step, 0, wait=True)
            yield from abs_set(self._encoder.pc.gate_num, 1, wait=True)

            ## PC Pulse
            yield from abs_set(
                self._encoder.pc.pulse_source, 0, wait=True
            )  # 0 = Position, 1 = Time, 2 = External
            set_and_wait(
                self._encoder.pc.pulse_width, self.dft_pulse_wid[self.tspre], rtol=0.1
            )
            # set(0).wait(_timeout)
        elif self.scn_mode == "snaked: single file":
            ############### PC Tab ###############
            ## PC Gate
            yield from abs_set(
                self._encoder.pc.gate_source, 2, wait=True
            )  # 0 = Position, 1 = Time, 2 = External
            yield from abs_set(self._encoder.pc.gate_ext, 29, wait=True)
            yield from abs_set(self._encoder.pc.gate_num, 0, wait=True)

            ## PC Pulse
            yield from abs_set(
                self._encoder.pc.pulse_source, 1, wait=True
            )  # 0 = Position, 1 = Time, 2 = External
            set_and_wait(
                self._encoder.pc.pulse_width, self.dft_pulse_wid[self.tspre], rtol=0.1
            )

        for key, val in pc_cfg[self.scn_mode].items():
            set_and_wait(getattr(self._encoder.pc, key), val, rtol=0.1)

    def root_path_str(self):
        data_session = self._md["data_session"]
        cycle = self._md["cycle"]
        if "Commissioning" in get_proposal_type():
            root_path = f"/nsls2/data/fxi-new/proposals/commissioning/{data_session}/assets/"
        else:
            root_path = f"/nsls2/data/fxi-new/proposals/{cycle}/{data_session}/assets/"
        return root_path

    def make_filename(self):
        """Make a filename.
        Taken/Modified from ophyd.areadetector.filestore_mixins
        This is a hook so that the read and write paths can either be modified
        or created on disk prior to configuring the areaDetector plugin.
        Returns
        -------
        filename : str
            The start of the filename
        read_path : str
            Path that ophyd can read from
        write_path : str
            Path that the IOC can write to
        """
        filename = f"{new_short_uid()}.h5"
        formatter = datetime.now().strftime
        write_path = formatter(f"{self.root_path}{self.write_path_template}")
        read_path = formatter(f"{self.root_path}{self.read_path_template}")
        return filename, read_path, write_path

    def stage(self):
        self._stage_with_delay()
        self.root_path = self.root_path_str()
        super.stage()

    def _stage_with_delay(self):
        # Staging taken from https://github.com/bluesky/ophyd/blob/master/ophyd/device.py
        # Device - BlueskyInterface
        """Stage the device for data collection.
        This method is expected to put the device into a state where
        repeated calls to :meth:`~BlueskyInterface.trigger` and
        :meth:`~BlueskyInterface.read` will 'do the right thing'.
        Staging not idempotent and should raise
        :obj:`RedundantStaging` if staged twice without an
        intermediate :meth:`~BlueskyInterface.unstage`.
        This method should be as fast as is feasible as it does not return
        a status object.
        The return value of this is a list of all of the (sub) devices
        stage, including it's self.  This is used to ensure devices
        are not staged twice by the :obj:`~bluesky.run_engine.RunEngine`.
        This is an optional method, if the device does not need
        staging behavior it should not implement `stage` (or
        `unstage`).
        Returns
        -------
        devices : list
            list including self and all child devices staged
        """
        if self._staged == Staged.no:
            pass  # to short-circuit checking individual cases
        elif self._staged == Staged.yes:
            raise RedundantStaging(
                "Device {!r} is already staged. " "Unstage it first.".format(self)
            )
        elif self._staged == Staged.partially:
            raise RedundantStaging(
                "Device {!r} has been partially staged. "
                "Maybe the most recent unstaging "
                "encountered an error before finishing. "
                "Try unstaging again.".format(self)
            )
        self.log.debug("Staging %s", self.name)
        self._staged = Staged.partially

        # Resolve any stage_sigs keys given as strings: 'a.b' -> self.a.b
        stage_sigs = OrderedDict()
        for k, v in self.stage_sigs.items():
            if isinstance(k, str):
                # Device.__getattr__ handles nested attr lookup
                stage_sigs[getattr(self, k)] = v
            else:
                stage_sigs[k] = v

        # Read current values, to be restored by unstage()
        original_vals = {sig: sig.get() for sig in stage_sigs}

        # We will add signals and values from original_vals to
        # self._original_vals one at a time so that
        # we can undo our partial work in the event of an error.

        # Apply settings.
        devices_staged = []
        try:
            for sig, val in stage_sigs.items():
                self.log.debug(
                    "Setting %s to %r (original value: %r)",
                    self.name,
                    val,
                    original_vals[sig],
                )
                sig.set(val, timeout=10).wait()
                ttime.sleep(self._staging_delay)
                # It worked -- now add it to this list of sigs to unstage.
                self._original_vals[sig] = original_vals[sig]
            devices_staged.append(self)

            # Call stage() on child devices.
            for attr in self._sub_devices:
                device = getattr(self, attr)
                if hasattr(device, "stage"):
                    device.stage()
                    devices_staged.append(device)
        except Exception:
            self.log.debug(
                "An exception was raised while staging %s or "
                "one of its children. Attempting to restore "
                "original settings before re-raising the "
                "exception.",
                self.name,
            )
            self.unstage()
            raise
        else:
            self._staged = Staged.yes
        return devices_staged

    def unstage(self):
        self._unstage_with_delay()

    def _unstage_with_delay(self):
        # Staging taken from https://github.com/bluesky/ophyd/blob/master/ophyd/device.py
        # Device - BlueskyInterface
        """Unstage the device.
        This method returns the device to the state it was prior to the
        last `stage` call.
        This method should be as fast as feasible as it does not
        return a status object.
        This method must be idempotent, multiple calls (without a new
        call to 'stage') have no effect.
        Returns
        -------
        devices : list
            list including self and all child devices unstaged
        """
        self.log.debug("Unstaging %s", self.name)
        self._staged = Staged.partially
        devices_unstaged = []

        # Call unstage() on child devices.
        for attr in self._sub_devices[::-1]:
            device = getattr(self, attr)
            if hasattr(device, "unstage"):
                device.unstage()
                devices_unstaged.append(device)

        # Restore original values.
        for sig, val in reversed(list(self._original_vals.items())):
            self.log.debug("Setting %s back to its original value: %r)", self.name, val)
            sig.set(val, timeout=10).wait()
            ttime.sleep(self._staging_delay)
            self._original_vals.pop(sig)
        devices_unstaged.append(self)

        self._staged = Staged.no
        return devices_unstaged

    def kickoff(self, *, scn_cfg={}):
        self._encoder.pc.arm.put(0)
        ttime.sleep(self._staging_delay)
        self._state = "kicked off"

        if scn_cfg["ang_s"] < scn_cfg["ang_e"]:
            self._encoder.pc.dir.put(0)
            try:
                self.rot_axis.user_setpoint.put(scn_cfg["ang_s"] - scn_cfg["taxi_dist"])
            except Exception as e:
                print(e)
                print("Cannot move rotary stage to its taxi position.")
                return
        else:
            self._encoder.pc.dir.put(1)
            try:
                self.rot_axis.user_setpoint.put(scn_cfg["ang_s"] + scn_cfg["taxi_dist"])
            except Exception as e:
                print(e)
                print("Cannot move rotary stage to its taxi position.")
                return

        if scn_cfg["scn_mode"] == "snaked: multiple files":
            self._encoder.pc.gate_start.put(scn_cfg["ang_s"])

        # sync rotary stage encoder
        self._encoder.pc.enc_pos1_sync.put(1)
        ttime.sleep(self._staging_delay)

        # Do a block reset on the zebra
        self._encoder.pc.block_state_reset.put(1)
        ttime.sleep(self._staging_delay)

        return NullStatus()

    def complete(self):
        """
        Call this when all needed data has been collected. This has no idea
        whether that is true, so it will obligingly stop immediately. It is
        up to the caller to ensure that the motion is actually complete.
        """
        # Our acquisition complete PV is: XF:05IDD-ES:1{Dev:Zebra1}:ARRAY_ACQ
        t0 = ttime.monotonic()
        while self._encoder.pc.data_in_progress.get() == 1:
            ttime.sleep(self._staging_delay)
            if (ttime.monotonic() - t0) > 60:
                print(f"{self.name} is behaving badly!")
                self._encoder.pc.disarm.put(1)
                ttime.sleep(0.100)
                if self._encoder.pc.data_in_progress.get() == 1:
                    raise TimeoutError

        self._state = "complete"
        self._encoder.pc.block_state_reset.put(1)

        for d in self._dets:
            d.stop()

        # Set filename/path for zebra data
        f, rp, wp = self.make_filename()
        self.__filename = f
        self.__read_filepath = os.path.join(rp, self.__filename)
        self.__write_filepath = os.path.join(wp, self.__filename)

        self.__filestore_resource, datum_factory_z = resource_factory(
            "ZEBRA_HDF51",
            root="/",
            resource_path=self.__read_filepath,
            resource_kwargs={},
            path_semantics="posix",
        )

        time_datum = datum_factory_z({"column": "zebra_time"})
        enc1_datum = datum_factory_z({"column": "enc1_pi_r"})

        # self._document_cache defines resource and datum documents
        self._document_cache = [("resource", self.__filestore_resource)]
        self._document_cache.extend(
            ("datum", d)
            for d in (
                time_datum,
                enc1_datum,
            )
        )

        # grab the asset documents from all of the child detectors
        for d in self._dets:
            self._document_cache.extend(d.collect_asset_docs())

        # Write the file.
        # @timer_wrapper
        def get_zebra_data():
            try:
                # TODO: Define condition for when to use nano export
                # See SRX's approach: https://github.com/NSLS2/srx-profile-collection/blob/main/startup/32-zebra.py#L953
                # Options: check flyer name, add use_nano_export signal, etc.
                use_nano_export = True  # Placeholder - update trigger logic as needed
                if use_nano_export:
                    # NOTE: this is a new export function which uses the caproto IOC for file saving into the proposal dirs.
                    export_nano_zebra_data(self._encoder, self.__write_filepath, self.fast_axis.get())
                else:
                    # NOTE: this export function is legacy and uses the operator account to write data.
                    export_zebra_data(self._encoder, self.__write_filepath)
            except TimeoutError as ex:
                print(f"Zebra data collection timeout error. {ex}")
                print("Scan continuing...")

        get_zebra_data()

        # Yield a (partial) Event document. The RunEngine will put this
        # into metadatastore, as it does all readings.
        self._last_bulk = {
            "time": ttime.time(),
            "seq_num": 1,
            "data": {
                "zebra_time": time_datum["datum_id"],
                "enc1_pi_r": enc1_datum["datum_id"],
            },
            "timestamps": {
                "zebra_time": time_datum["datum_id"],  # not a typo#
                "enc1_pi_r": time_datum["datum_id"],
            },
        }

        for d in self._dets:
            reading = d.read()
            self._last_bulk["data"].update({k: v["value"] for k, v in reading.items()})
            self._last_bulk["timestamps"].update(
                {k: v["timestamp"] for k, v in reading.items()}
            )

        return NullStatus()

    def describe_collect(self):
        ext_spec = "FileStore:"
        num_zebra_data = self.encoder.pc.data.time.get().shape[0]

        spec = {
            "external": ext_spec,
            "dtype": "array",
            "shape": [num_zebra_data],
            "source": "",  # make this the PV of the array the det is writing
        }

        desc = OrderedDict()  # desc defines data_keys

        desc["zebra_time"] = spec
        desc["zebra_time"]["source"] = getattr(self._encoder.pc.data, "enc1").pvname
        desc["enc1_pi_r"] = spec
        desc["enc1_pi_r"]["source"] = getattr(self._encoder.pc.data, "enc1").pvname

        # Handle the detectors we are going to get
        from pprint import pprint as _pprint
        from pprint import pformat

        for d in self.detectors:
            desc.update(d.describe())

        return {"primary": desc}

    def collect(self):
        # Create records in the FileStore database.
        # move this to stage because I think that describe_collect needs the
        # resource id
        # TODO use ophyd.areadectector.filestoer_mixins.resllource_factory here
        if self._last_bulk is None:
            raise Exception(
                "the order of complete and collect is brittle and out "
                "of sync. This device relies on in-order and 1:1 calls "
                "between complete and collect to correctly create and stash "
                "the asset registry documents"
            )
        # self._last_bulk defines a event document
        yield self._last_bulk
        self._last_bulk = None
        self._state = "idle"

    def collect_asset_docs(self):
        yield from iter(list(self._document_cache))
        self._document_cache.clear()

    def stop(self):
        self._encoder.pc.block_state_reset.put(1)
        pass

    def pause(self):
        "Pausing in the middle of a kickoff nukes the partial dataset."
        self._encoder.pc.block_state_reset.put(1)
        for d in self._dets:
            if hasattr(d, "settings"):
                d.settings.acquire.put(0)
            if hasattr(d, "cam"):
                d.cam.acquire.put(0)
        self._state = "idle"
        self.unstage()

    def resume(self):
        self.unstage()
        self.stage()

    def preset_flyer(self, scn_cfg):
        yield from FXITomoFlyer.bin_det(self.detectors[0], scn_cfg["bin_fac"])
        yield from FXITomoFlyer.prime_det(self.detectors[0])
        yield from FXITomoFlyer.init_mot_r(scn_cfg)
        scn_cfg = yield from FXITomoFlyer.cal_cam_rot_params(self.detectors[0], scn_cfg)
        pc_cfg = FXITomoFlyer.cal_zebra_pc_params(scn_cfg)
        yield from self.preset_zebra(pc_cfg)
        print("preset_flyer is done")
        return scn_cfg, pc_cfg

    def set_pc_step_for_scan(self, scn_cfg, pc_cfg):
        print(f"{pc_cfg=}")
        pc_cfg[self.scn_mode]["dir"] = self.pc_trig_dir[int(scn_cfg["rot_dir"])]
        print(f"{pc_cfg=}")
        yield from abs_set(self.encoder.pc.dir, pc_cfg[self.scn_mode]["dir"], wait=True)
        print(f"{pc_cfg=}")
        pc_cfg[self.scn_mode]["gate_start"] = FXITomoFlyer._adjust_zebra_gate_start(
            scn_cfg["ang_s"]
        )
        print(f"{pc_cfg=}")
        yield from abs_set(
            self.encoder.pc.gate_start, pc_cfg[self.scn_mode]["gate_start"], wait=True
        )

    @classmethod
    def cal_cam_rot_params(cls, det, scn_cfg):
        """_summary_

        Args:
            det (str): choose from the set {"Andor", "MaranaU", "KinetixU", "Oryx"}
            scn_cfg (dict): scan configuration parameters composed of
                'scn_mode': choose between {
                        0: "standard", # a single scan in a given angle range
                        1: "snaked: single file", # back-forth rocking scan being saved into a single file
                        2: "snaked: multiple files" # back-forth rocking scan with each swing being saved into a file
                        }
                'exp_t': detector exposure time in second,
                'acq_p': acquisition period in second,
                'bin_fac': detector binning factor,
                'ang_s': scan starting angle,
                'ang_e': scan end angle,
                'num_swing': number of sub-scans; motion from one side to another side is defined as one swing
                'vel': rotation velocity in deg/sec,
                'tacc': rotation stage acceleration in sec,
                "taxi_dist": taxi distance in unit deg

        Returns:
            dict: scan_cfg
        """
        cam_model = _get_cam_model(det)
        ############### calculate detector parameters ###############
        if scn_cfg["exp_t"] < FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["min_exp"]:
            print(
                "Exposure time is too small for the camera. Reset exposure time to minimum allowed exposure time."
            )
            scn_cfg["exp_t"] = FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["min_exp"]

        acq_p, acq_min = yield from FXITomoFlyer.check_cam_acq_p(
            det, scn_cfg["exp_t"], scn_cfg["acq_p"]
        )

        print(f'{acq_p=} {scn_cfg["acq_p"]=}')
        if acq_p > scn_cfg["acq_p"]:
            print(
                "Acquisition period is too small for the camera. Reset acquisition period to minimum allowed exposure time."
            )
        scn_cfg["acq_p"] = acq_p

        print(
            f"scn_cfg['acq_p']: {scn_cfg['acq_p']}, scn_cfg['exp_t']: {scn_cfg['exp_t']}"
        )

        # if scn_cfg["exp_t"] > (acq_p - acq_min):
        #     scn_cfg["exp_t"] = acq_p - acq_min

        ############### calculate rotary stage parameters ###############
        if scn_cfg["tacc"] <= 0:
            print("Acceleration time cannot be smaller than 0. Reset it to 1 second.")
            scn_cfg["tacc"] = 1

        if scn_cfg["vel"] > cls.rot_axis.max_velo.get():
            print(
                "Designed velocity exceeds the maximum allowed velocity. Reset it to the maximum allowed velocity"
            )
            scn_cfg["vel"] = cls.rot_axis.max_velo.get()
        elif scn_cfg["vel"] < cls.rot_axis.base_velo.get():
            print(
                "Designed velocity is smaller than the minimum allowed velocity. Reset it to the minimum allowed velocity"
            )
            scn_cfg["vel"] = cls.rot_axis.base_velo.get()

        taxi_dist = max(
            np.ceil(
                (scn_cfg["vel"] - cls.rot_axis.base_velo.get()) * scn_cfg["tacc"] / 2
            ),
            1,
        )
        print(f"{taxi_dist=}")
        if not (
            cls.rot_axis.low_limit_switch.get() == 0
            and cls.rot_axis.high_limit_switch.get() == 0
        ):
            if (scn_cfg["ang_s"] - taxi_dist) < cls.rot_axis.low_limit.get():
                print(
                    "Rotation range is beyond the low limit of the rotary stage. Quit!"
                )
                return None
            elif (scn_cfg["ang_s"] - taxi_dist) > cls.rot_axis.high_limit.get():
                print(
                    "Rotation range is beyond the high limit of the rotary stage. Quit!"
                )
                return None

            if (scn_cfg["ang_e"] + taxi_dist) < cls.rot_axis.low_limit.get():
                print(
                    "Rotation range is beyond the low limit of the rotary stage. Quit!"
                )
                return None
            elif (scn_cfg["ang_e"] + taxi_dist) > cls.rot_axis.high_limit.get():
                print(
                    "Rotation range is beyond the high limit of the rotary stage. Quit!"
                )
                return None

        print(f"{taxi_dist=}")
        scn_cfg["taxi_dist"] = taxi_dist
        scn_cfg["ang_step"] = scn_cfg["vel"] * (scn_cfg["acq_p"] + FXITomoFlyer.rot_var)
        if cls.scn_modes[scn_cfg["scn_mode"]] == "snaked: single file":
            scn_cfg["num_images"] = int(
                int(
                    round(
                        abs(scn_cfg["ang_e"] - scn_cfg["ang_s"])
                        / (scn_cfg["acq_p"] * scn_cfg["vel"])
                        + 2 * scn_cfg["tacc"] / scn_cfg["acq_p"]
                    )
                )
                * scn_cfg["num_swing"]
            )
        else:
            scn_cfg["num_images"] = int(
                round(
                    abs(scn_cfg["ang_e"] - scn_cfg["ang_s"])
                    / round(scn_cfg["ang_step"], 3)
                    + 1
                )
            )

        if scn_cfg["ang_s"] < scn_cfg["ang_e"]:
            scn_cfg["rot_dir"] = 1
        else:
            scn_cfg["rot_dir"] = -1
        print(f"scn_cfg: {scn_cfg}")
        return scn_cfg

    @staticmethod
    def check_cam_acq_p(det, exp_t, acq_p):
        yield from abs_set(det.cam.acquire, 0, wait=True)
        cam_model = _get_cam_model(det)
        if cam_model == "MARANA-4BV6X":
            full_acq_min = CAM_RD_CFG[cam_model]["rd_time"][
                det.pre_amp.enum_strs[det.pre_amp.value]
            ]
            acq_min = (
                full_acq_min * (det.cam.size.size_y.value / 2046) + FXITomoFlyer.rot_var
            )  # add vel uncertainty tolerance
        elif cam_model == "SONA-4BV6X":
            full_acq_min = CAM_RD_CFG[cam_model]["rd_time"][
                det.pre_amp.enum_strs[det.pre_amp.value]
            ]
            acq_min = (
                full_acq_min * (det.cam.size.size_y.value / 2048) + FXITomoFlyer.rot_var
            )  # add vel uncertainty tolerance
        elif cam_model == "KINETIX22":
            full_acq_min = CAM_RD_CFG[cam_model]["rd_time"][
                det.cam.readout_port_names[det.cam.readout_port_idx.value]
            ]
            acq_min = (
                full_acq_min * (det.cam.size.size_y.value / 2400) + FXITomoFlyer.rot_var
            )  # add vel uncertainty tolerance
        elif cam_model == "KINETIX":
            full_acq_min = CAM_RD_CFG[cam_model]["rd_time"][
                det.cam.readout_port_names[det.cam.readout_port_idx.value]
            ]
            acq_min = (
                full_acq_min * (det.cam.size.size_y.value / 3200) + FXITomoFlyer.rot_var
            )  # add vel uncertainty tolerance
        acq_p = (
            round(max(acq_min + exp_t, acq_p) * 1000) / 1000.0
        )  # add miminum exposure time
        return acq_p, acq_min

    @classmethod
    def cal_zebra_pc_params(cls, scn_cfg):
        """_summary_

        Args:
            scn_cfg (dict): scan configuration parameters composed of
                'scn_mode': choose between {
                        0: "standard", # a single scan in a given angle range
                        1: "snaked: single file", # back-forth rocking scan being saved into a single file
                        2: "snaked: multiple files" # back-forth rocking scan with each swing being saved into a file
                        }
                'exp_t': detector exposure time in second,
                'acq_p': acquisition period in second,
                'bin_fac': detector binning factor,
                'ang_s': scan starting angle,
                'ang_e': scan end angle,
                'num_swing': number of sub-scans; motion from one side to another side is defined as one swing
                'vel': rotation velocity in deg/sec,
                'tacc': rotation stage acceleration in sec,
                "taxi_dist": taxi distance in unit deg

        Returns:
            dict: pc_cfg
        """
        pc_cfg = {
            "standard": {},
            "snaked: multiple files": {},
            "snaked: single file": {},
        }
        if cls.scn_modes[scn_cfg["scn_mode"]] == "standard":
            pc_cfg["standard"]["pulse_start"] = 0
            pc_cfg["standard"]["pulse_width"] = (
                cls.dft_pulse_wid[cls.tspre] * scn_cfg["vel"]
            )
            pc_cfg["standard"]["pulse_step"] = round(scn_cfg["ang_step"], 3)
            pc_cfg["standard"]["pulse_max"] = int(
                round(
                    abs(scn_cfg["ang_e"] - scn_cfg["ang_s"])
                    / pc_cfg["standard"]["pulse_step"]
                    + 1
                )
            )
            pc_cfg["standard"]["gate_start"] = cls._adjust_zebra_gate_start(
                scn_cfg["ang_s"]
            )
            pc_cfg["standard"]["gate_width"] = (
                abs(scn_cfg["ang_e"] - scn_cfg["ang_s"])
                + pc_cfg["standard"]["pulse_step"]
            )
        elif cls.scn_modes[scn_cfg["scn_mode"]] == "snaked: multiple files":
            pc_cfg["snaked: multiple files"]["pulse_start"] = 0
            pc_cfg["snaked: multiple files"]["pulse_width"] = (
                cls.dft_pulse_wid[cls.tspre] * scn_cfg["vel"]
            )
            pc_cfg["snaked: multiple files"]["pulse_step"] = round(
                scn_cfg["ang_step"], 3
            )
            pc_cfg["snaked: multiple files"]["pulse_max"] = int(
                round(
                    abs(scn_cfg["ang_e"] - scn_cfg["ang_s"])
                    / pc_cfg["snaked: multiple files"]["pulse_step"]
                    + 1
                )
            )
            pc_cfg["snaked: multiple files"]["gate_width"] = (
                abs(scn_cfg["ang_e"] - scn_cfg["ang_s"])
                + pc_cfg["snaked: multiple files"]["pulse_step"]
            )
        elif cls.scn_modes[scn_cfg["scn_mode"]] == "snaked: single file":
            pc_cfg["snaked: single file"]["pulse_start"] = 0
            pc_cfg["snaked: single file"]["pulse_start"] = 0
            pc_cfg["snaked: single file"]["pulse_width"] = cls.dft_pulse_wid[cls.tspre]
            pc_cfg["snaked: single file"]["pulse_step"] = round(scn_cfg["acq_p"], 4)
            pc_cfg["snaked: single file"]["pulse_max"] = int(
                int(
                    round(
                        abs(scn_cfg["ang_e"] - scn_cfg["ang_s"])
                        / (scn_cfg["acq_p"] * scn_cfg["vel"])
                        + 2 * scn_cfg["tacc"] / scn_cfg["acq_p"]
                    )
                )
                * scn_cfg["num_swing"]
            )
        else:
            print("Unrecognized scan mode. Quit")
            return None
        pc_cfg[cls.scn_modes[scn_cfg["scn_mode"]]]["dir"] = cls.pc_trig_dir[
            scn_cfg["rot_dir"]
        ]
        return pc_cfg

    @staticmethod
    def compose_scn_cfg(
        scn_mode, exp_t, acq_p, bin_fac, ang_s, ang_e, vel, tacc, mb_vel, num_swing
    ):
        scn_cfg = {}
        scn_cfg["scn_mode"] = scn_mode
        scn_cfg["exp_t"] = exp_t
        scn_cfg["acq_p"] = acq_p
        scn_cfg["bin_fac"] = 0 if bin_fac is None else bin_fac
        scn_cfg["ang_s"] = ang_s
        scn_cfg["ang_e"] = ang_e
        scn_cfg["vel"] = vel
        scn_cfg["tacc"] = tacc
        scn_cfg["mb_vel"] = mb_vel
        scn_cfg["num_swing"] = num_swing
        return scn_cfg

    @staticmethod
    def _prime_det(det):
        yield from FXITomoFlyer.stop_det(det)
        cam_model = _get_cam_model(det)
        if (
            CAM_RD_CFG[cam_model]["trigger_mode"][det.cam.trigger_mode.value]
            != FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["trigger_mode"][0]
        ):
            yield from abs_set(
                det.cam.trigger_mode,
                FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["trigger_mode"][0],
                wait=True,
            )
        if (
            CAM_RD_CFG[cam_model]["image_mode"][det.cam.image_mode.value]
            != FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["image_mode"][1]
        ):
            yield from abs_set(
                det.cam.image_mode,
                FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["image_mode"][1],
                wait=True,
            )
        yield from abs_set(det.cam.num_images, 5, wait=True)
        yield from abs_set(det.cam.acquire, 1, wait=True)
        print(f"{det.name} is primed!")

    @staticmethod
    def stop_det(det):
        yield from abs_set(det.cam.acquire, 0, wait=True)
        yield from abs_set(det.hdf5.capture, 0, wait=True)

    @staticmethod
    def bin_det(det, bin_fac):
        yield from FXITomoFlyer.stop_det(det)
        cam_model = _get_cam_model(det)
        if cam_model in ["MARANA-4BV6X", "SONA-4BV6X"]:
            if bin_fac is None:
                bin_fac = 0
            if (
                int(bin_fac)
                not in FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["bin_options"]
            ):
                raise ValueError(
                    f"binnng must be in {FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]['bin_options']}"
                )
            try:
                if det.binning.value != bin_fac:
                    yield from abs_set(det.binning, bin_fac, wait=True)
                return bin_fac
            except Exception:
                raise
        elif cam_model in ["KINETIX22", "KINETIX"]:
            if bin_fac is None:
                bin_fac = 0
            if (
                int(bin_fac)
                not in FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["bin_options"]
            ):
                raise ValueError(
                    f"binnng must be in {FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]['bin_options']}"
                )
            if bin_fac == 0:
                binning = 1
            elif bin_fac == 1:
                binning = 2
            elif bin_fac == 2:
                binning = 4
            try:
                if det.cam.bin_x.value != binning:
                    yield from abs_set(det.cam.bin_x, binning, wait=True)
                if det.cam.bin_y.value != binning:
                    yield from abs_set(det.cam.bin_y, binning, wait=True)
                return bin_fac
            except Exception:
                raise

    @staticmethod
    def prime_det(det):
        if (
            det.cam.array_size.array_size_x.value != det.hdf5.array_size.width.value
        ) or (
            det.cam.array_size.array_size_y.value != det.hdf5.array_size.height.value
        ):
            yield from FXITomoFlyer._prime_det(det)

    @staticmethod
    def set_roi_det(det, roi):
        """
        det.cam.min_x.value
        det.cam.min_y.value
        det.cam.size.size_x.value
        det.cam.size.size_y.value
        det.cam.max_size.max_size_x.value
        det.cam.max_size.max_size_y.value
        """
        yield from FXITomoFlyer.stop_det(det)
        if roi is None:
            yield from abs_set(
                det.cam.size.size_x, det.cam.max_size.max_size_x.value, wait=True
            )
            yield from abs_set(det.cam.min_x, 0, wait=True)
            yield from abs_set(
                det.cam.size.size_y, det.cam.max_size.max_size_y.value, wait=True
            )
            yield from abs_set(det.cam.min_y, 0, wait=True)
        elif ((roi["min_x"] + roi["size_x"]) < det.cam.max_size.max_size_x.value) and (
            (roi["min_y"] + roi["size_y"]) < det.cam.max_size.max_size_y.value
        ):
            yield from abs_set(det.cam.size.size_x, roi["size_x"], wait=True)
            yield from abs_set(det.cam.min_x, roi["min_x"], wait=True)
            yield from abs_set(det.cam.size.size_y, roi["size_y"], wait=True)
            yield from abs_set(det.cam.min_y, roi["min_y"], wait=True)
        elif (roi["min_x"] + roi["size_x"]) < det.cam.max_size.max_size_x.value:
            if (roi["min_y"] + roi["size_y"]) < det.cam.max_size.max_size_y.value:
                raise ValueError(
                    "both roi x and y sizes exceed allowed range with given x/y start positions!"
                )
            else:
                raise ValueError(
                    "roi x size exceeds allowed range with given x start position!"
                )
        elif (roi["min_y"] + roi["size_y"]) < det.cam.max_size.max_size_y.value:
            raise ValueError(
                "roi y size exceeds allowed range with given y start position!"
            )

    @staticmethod
    def def_abs_out_pos(
        x_out,
        y_out,
        z_out,
        r_out,
        rel_out_flag,
    ):
        (x_ini, y_ini, z_ini, r_ini) = FXITomoFlyer.get_txm_cur_pos()
        if rel_out_flag:
            mot_x_out = x_ini + x_out if not (x_out is None) else x_ini
            mot_y_out = y_ini + y_out if not (y_out is None) else y_ini
            mot_z_out = z_ini + z_out if not (z_out is None) else z_ini
            mot_r_out = r_ini + r_out if not (r_out is None) else r_ini
        else:
            mot_x_out = x_out if not (x_out is None) else x_ini
            mot_y_out = y_out if not (y_out is None) else y_ini
            mot_z_out = z_out if not (z_out is None) else z_ini
            mot_r_out = r_out if not (r_out is None) else r_ini
        return mot_x_out, mot_y_out, mot_z_out, mot_r_out

    @staticmethod
    def get_txm_cur_pos():
        x_ini = zps.sx.position
        y_ini = zps.sy.position
        z_ini = zps.sz.position
        r_ini = zps.pi_r.position
        return x_ini, y_ini, z_ini, r_ini

    @staticmethod
    def init_mot_r(scn_cfg):
        cur_pos = zps.pi_r.position
        yield from abs_set(zps.pi_r.offset_freeze_switch, 1, wait=True)
        if (-360 > cur_pos) or (cur_pos < 360):
            cur_pos = (
                np.sign((scn_cfg["ang_s"])) * (abs(scn_cfg["ang_s"]) // 360)
            ) * 360 + np.sign(cur_pos) * (abs(cur_pos) % 360)        
            zps.pi_r.set_current_position(cur_pos)
        yield from abs_set(zps.pi_r.acceleration, 1, wait=True)
        yield from abs_set(zps.pi_r.velocity, scn_cfg["mb_vel"], wait=True)
        yield from abs_set(zps.pi_r, scn_cfg["ang_s"], wait=True)

    @staticmethod
    def set_cam_step_for_scan(det, scn_cfg):
        yield from abs_set(det.cam.acquire, 0, wait=True)
        yield from abs_set(det.cam.acquire_time, scn_cfg["exp_t"], wait=True)
        yield from abs_set(det.hdf5.num_capture, scn_cfg["num_images"], wait=True)
        yield from abs_set(det.cam.num_images, scn_cfg["num_images"], wait=True)

    @staticmethod
    def set_mot_r_step_for_scan(scn_cfg):
        yield from abs_set(zps.pi_r.acceleration, scn_cfg["tacc"], wait=True)
        yield from abs_set(zps.pi_r.velocity, scn_cfg["vel"], wait=True)
        yield from abs_set(
            zps.pi_r,
            scn_cfg["ang_s"] - scn_cfg["rot_dir"] * scn_cfg["taxi_dist"],
            wait=True,
        )

    @staticmethod
    def set_cam_mode(det, stage="pre-scan"):
        yield from abs_set(det.cam.acquire, 0, wait=True)
        cam_model = _get_cam_model(det)
        if stage == "pre-scan":
            print(f"{stage=}")
            yield from abs_set(
                det.cam.image_mode,
                CAM_RD_CFG[cam_model]["image_mode"].index(
                    FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["image_mode"][1]
                ),
                wait=True,
            )
            yield from abs_set(
                det.cam.trigger_mode,
                CAM_RD_CFG[cam_model]["trigger_mode"].index(
                    FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["trigger_mode"][1]
                ),
                wait=True,
            )
        elif stage == "ref-scan":
            print(f"{stage=}")
            yield from abs_set(
                det.cam.image_mode,
                CAM_RD_CFG[cam_model]["image_mode"].index(
                    FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["image_mode"][1]
                ),
                wait=True,
            )
            yield from abs_set(
                det.cam.trigger_mode,
                CAM_RD_CFG[cam_model]["trigger_mode"].index(
                    FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["trigger_mode"][0]
                ),
                wait=True,
            )
        elif stage == "post-scan":
            print(f"{stage=}")
            yield from abs_set(
                det.cam.image_mode,
                CAM_RD_CFG[cam_model]["image_mode"].index(
                    FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["image_mode"][0]
                ),
                wait=True,
            )
            yield from abs_set(
                det.cam.trigger_mode,
                CAM_RD_CFG[cam_model]["trigger_mode"].index(
                    FXITomoFlyer.CAM_MODES_IN_FLYER[cam_model]["trigger_mode"][0]
                ),
                wait=True,
            )
        return


Zebra = FXIZebra(
    "XF:18ID-ES:1{Dev:Zebra1}:",
    name="Zebra",
    read_attrs=["pc.data.enc1", "pc.data.time"],
)

try:
    tomo_maranau_flyer = FXITomoFlyer(
        list((MaranaU,)),
        Zebra,
        name="tomo_maranau_flyer",
        md=RE.md,
    )
except:
    print("MaranaU is not online")

try:
    tomo_maranad_flyer = FXITomoFlyer(
        list((MaranaD,)),
        Zebra,
        name="tomo_maranad_flyer",
        md=RE.md,
    )
except:
    print("MaranaD is not online")

try:
    tomo_kinetixu_flyer = FXITomoFlyer(
        list((KinetixU,)),
        Zebra,
        name="tomo_kinetixu_flyer",
        md=RE.md,
    )
except:
    print("KinetixU is not online")

try:
    tomo_kinetixd_flyer = FXITomoFlyer(
        list((KinetixD,)),
        Zebra,
        name="tomo_kinetixd_flyer",
        md=RE.md,
    )
except:
    print("KinetixD is not online")


def _sel_flyer(flyer):
    if flyer is None:
        return tomo_kinetixu_flyer
    elif flyer == "tomo_kinetixu_flyer":
        return tomo_kinetixu_flyer
    elif flyer == "tomo_kinetixd_flyer":
        return tomo_kinetixd_flyer
    elif flyer == "tomo_maranau_flyer":
        return tomo_maranau_flyer
    elif flyer == "tomo_maranad_flyer":
        return tomo_maranad_flyer
    else:
        return flyer


def export_zebra_data(zebra, filepath):
    j = 0
    while zebra.pc.data_in_progress.get() == 1:
        print("Waiting for zebra...")
        ttime.sleep(0.1)
        j += 1
        if j > 10:
            print("THE ZEBRA IS BEHAVING BADLY CARRYING ON")
            break

    time_d = zebra.pc.data.time.get()
    enc1_d = zebra.pc.data.enc1.get()

    size = (len(time_d),)
    with h5py.File(filepath, "w") as f:
        dset0 = f.create_dataset("zebra_time", size, dtype="f")
        dset0[...] = np.array(time_d)
        dset1 = f.create_dataset("enc1_pi_r", size, dtype="f")
        dset1[...] = np.array(enc1_d)


def export_nano_zebra_data(zebra, filepath, fastaxis):
    """
    Export zebra position capture data using the ZebraSaver IOC.

    Parameters
    ----------
    zebra : FXIZebra
        The zebra device
    filepath : str
        Full path for the output HDF5 file
    fastaxis : str
        The fast scanning axis. Options:
        - 'PI_R': rotation axis (adjusts enc1)
        - 'SX': sample X axis (adjusts enc2)
        - 'SY': sample Y axis (adjusts enc3)
    """
    j = 0
    while zebra.pc.data_in_progress.get() == 1:
        print("Waiting for zebra...")
        ttime.sleep(0.1)
        j += 1
        if j > 10:
            print("THE ZEBRA IS BEHAVING BADLY CARRYING ON")
            break

    time_d = zebra.pc.data.time.get()
    enc1_d = zebra.pc.data.enc1.get()
    enc2_d = zebra.pc.data.enc2.get()
    enc3_d = zebra.pc.data.enc3.get()

    px = zebra.pc.pulse_step.get()
    if fastaxis == 'PI_R':
        enc1_d = enc1_d + (px / 2)
    elif fastaxis == 'SX':
        enc2_d = enc2_d + (px / 2)
    elif fastaxis == 'SY':
        enc3_d = enc3_d + (px / 2)

    zs.enc1.put(enc1_d)
    zs.enc2.put(enc2_d)
    zs.enc3.put(enc3_d)
    zs.zebra_time.put(time_d)

    write_dir = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)

    zs.dev_type.put("zebra")
    zs.write_dir.put(write_dir)
    zs.file_name.put(file_name)
    zs.file_stage.put("staged")

    def cb(value, old_value, **kwargs):
        if old_value in ["acquiring", 1] and value in ["idle", 0]:
            return True
        return False

    st = SubscriptionStatus(zs.acquire, callback=cb, run=False)
    zs.acquire.put(1)
    try:
        st.wait(timeout=60)
    except WaitTimeoutError:
        print("Zebra-save timed out! Continuing...")

    zs.file_stage.put("unstaged")


def export_sis_data(ion, filepath, zebra):
    """
    Export scaler/ion chamber data using the ZebraSaver IOC.

    Parameters
    ----------
    ion : FXIScaler
        The scaler device (sclr1)
    filepath : str
        Full path for the output HDF5 file
    zebra : FXIZebra
        The zebra device (for getting expected point count)
    """
    N = ion.nuse_all.get()
    t = ion.mca1.get(timeout=5.0)
    i = ion.mca2.get(timeout=5.0)
    im = ion.mca3.get(timeout=5.0)
    it = ion.mca4.get(timeout=5.0)

    while len(t) == 0 and len(t) != len(i):
        t = ion.mca1.get(timeout=5.0)
        i = ion.mca2.get(timeout=5.0)
        im = ion.mca3.get(timeout=5.0)
        it = ion.mca4.get(timeout=5.0)

    if len(i) != N:
        print(f'Scaler did not collect enough points.')
        t = ion.mca1.get(timeout=5.0)
        i = ion.mca2.get(timeout=5.0)
        im = ion.mca3.get(timeout=5.0)
        it = ion.mca4.get(timeout=5.0)
        if len(i) != N:
            print(f'Nope. Only received {len(i)}/{N} points.')

    correct_length = N // 2
    # Only consider even points
    t = t[1::2]
    i = i[1::2]
    im = im[1::2]
    it = it[1::2]

    if len(t) != correct_length:
        correction_factor = correct_length - len(t)
        print(f"Adding {correction_factor} points to scaler!")
        correction_list = [1e10 for _ in range(0, int(correction_factor))]
        new_t = list(t) + correction_list
        new_i = list(i) + correction_list
        new_im = list(im) + correction_list
        new_it = list(it) + correction_list
    else:
        new_t = t
        new_i = i
        new_im = im
        new_it = it

    zs.i0.put(new_i)
    zs.im.put(new_im)
    zs.it.put(new_it)
    zs.sis_time.put(new_t)

    write_dir = os.path.dirname(filepath)
    file_name = os.path.basename(filepath)

    zs.dev_type.put("scaler")
    zs.write_dir.put(write_dir)
    zs.file_name.put(file_name)
    zs.file_stage.put("staged")

    def cb(value, old_value, **kwargs):
        if old_value in ["acquiring", 1] and value in ["idle", 0]:
            return True
        return False

    st = SubscriptionStatus(zs.acquire, callback=cb, run=False)
    zs.acquire.put(1)
    try:
        st.wait(timeout=60)
    except WaitTimeoutError:
        print("Scaler-save timed out! Continuing...")

    zs.file_stage.put("unstaged")
