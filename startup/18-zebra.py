import os
import h5py
import datetime
import numpy as np
import time as ttime

from ophyd import Device, EpicsSignal, EpicsSignalRO
from ophyd import Component as Cpt
from ophyd import FormattedComponent as FC
from ophyd.utils import set_and_wait
from ophyd.areadetector.filestore_mixins import new_short_uid, resource_factory
from ophyd.sim import NullStatus
from databroker.assets.handlers import HandlerBase
from nslsii.detectors.zebra import Zebra, EpicsSignalWithRBV
from ophyd.device import Staged, RedundantStaging, OrderedDict
from bluesky.plan_stubs import abs_set


# minimum detector acquisition period in second for full frame size
# will move this definition to areaDetector.py
DET_MIN_AP = {"Andor": 0.05, "Marana": 0.01, "Oryx": 0.005}

# default velocity for rotating rotary stage back to starting position
# will move this definition to motors.py
ROT_BACK_VEL = 30

BIN_FACS = {"Andor": {0: 1, 1: 2, 2: 3, 3: 4, 4: 8}, "Marana": {}, "Oryx": {}}


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


## Class below requires FXI specific changes
class FXITomoFlyer(Device):
    """
    This is the flyer object for the Zebra.
    This is the position based flyer.
    """

    root_path = "/nsls2/data/fxi-new/legacy/"
    write_path_template = f"zebra/%Y/%m/%d/"
    read_path_template = f"zebra/%Y/%m/%d/"
    reg_root = f"zebra/"

    KNOWN_DETS = {"Andor", "Marana", "Oryx"}
    rot_axis = zps.pi_r
    # dummy_axis = ophyd.sim.SynAxis(name="TOMO_DUMMY")

    scn_modes = {
        0: "standard",  # a single scan in a given angle range
        1: "snaked: multiple files",  # back-forth rocking scan with each swing being saved into a file
        2: "snaked: single file",  # back-forth rocking scan being saved into a single file
    }

    dft_pulse_wid = {"ms": 0.002, "s": 0.0005, "10s": 0.003}  # 0: ms  # 1: s  # 2: 10s
    pc_trig_dir = {1: 0, -1: 1}  # 1: positive, -1: negative
    scan_cfg = {}
    pc_cfg = {}
    _staging_delay = 0.010
    tspre = "s"  ## ['ms', 's', '10s']

    def __init__(self, dets, zebra, *, reg=db.reg, scn_mode=0, **kwargs):
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

        self.reg = reg
        self.scn_mode = self.scn_modes[scn_mode]
        self.extra_stage_sigs = {}
        self.shutter_delay = 0.1  # unit: deg; _shutter_delay/rot_vel > unibliz shutter opening time 1.5ms
        self.use_shutter = True

        set_and_wait(self._encoder.pc.block_state_reset, 1)

        ############### Zebra Setup ###############
        ## PC Tab
        set_and_wait(self._encoder.pc.data.cap_enc1_bool, 1)
        set_and_wait(self._encoder.pc.data.cap_enc2_bool, 0)
        set_and_wait(self._encoder.pc.data.cap_enc3_bool, 0)
        set_and_wait(self._encoder.pc.data.cap_enc4_bool, 0)
        set_and_wait(self._encoder.pc.data.cap_filt1_bool, 0)
        set_and_wait(self._encoder.pc.data.cap_filt2_bool, 0)
        set_and_wait(self._encoder.pc.data.cap_div1_bool, 0)
        set_and_wait(self._encoder.pc.data.cap_div2_bool, 0)
        set_and_wait(self._encoder.pc.data.cap_div3_bool, 0)
        set_and_wait(self._encoder.pc.data.cap_div4_bool, 0)

        set_and_wait(self._encoder.pc.enc, 0)  # 0: Enc1, 1: Enc2, 2: Enc3, 3: Enc4,
        set_and_wait(self._encoder.pc.dir, 0)  # 0: Positive, 1: Negative
        set_and_wait(self._encoder.pc.tspre, 1)  # 0: ms, 1: s, 2: 10s

        ## AND tab -- can be used for external triggering
        set_and_wait(self._encoder.and1.use1, 0)  # 0: No, 1: Yes
        set_and_wait(self._encoder.and1.use2, 0)
        set_and_wait(self._encoder.and1.use3, 0)
        set_and_wait(self._encoder.and1.use4, 0)
        set_and_wait(self._encoder.and1.input_source1, 0)
        set_and_wait(self._encoder.and1.input_source2, 0)
        set_and_wait(self._encoder.and1.input_source3, 0)
        set_and_wait(self._encoder.and1.input_source4, 0)
        set_and_wait(self._encoder.and1.invert1, 0)  # 0: No, 1: Yes
        set_and_wait(self._encoder.and1.invert2, 0)
        set_and_wait(self._encoder.and1.invert3, 0)
        set_and_wait(self._encoder.and1.invert4, 0)

        ## OR Tab -- can be used for diagnose
        set_and_wait(self._encoder.or1.use1, 0)  # 0: No, 1: Yes
        set_and_wait(self._encoder.or1.use2, 0)
        set_and_wait(self._encoder.or1.use3, 0)
        set_and_wait(self._encoder.or1.use4, 0)
        set_and_wait(self._encoder.or1.input_source1, 0)
        set_and_wait(self._encoder.or1.input_source2, 0)
        set_and_wait(self._encoder.or1.input_source3, 0)
        set_and_wait(self._encoder.or1.input_source4, 0)
        set_and_wait(self._encoder.or1.invert1, 0)  # 0 = No, 1 = Yes
        set_and_wait(self._encoder.or1.invert2, 0)
        set_and_wait(self._encoder.or1.invert3, 0)
        set_and_wait(self._encoder.or1.invert4, 0)

        ## PULSE tab -- set for fast shutter
        set_and_wait(self._encoder.pulse1.input_addr, 31)
        set_and_wait(self._encoder.pulse1.input_edge, 0)  # 0 = rising, 1 = falling
        set_and_wait(self._encoder.pulse1.delay, 0)
        set_and_wait(self._encoder.pulse2.input_addr, 31)
        set_and_wait(self._encoder.pulse2.input_edge, 1)  # 0 = rising, 1 = falling
        set_and_wait(self._encoder.pulse2.delay, 0)

        ## ENC tab
        set_and_wait(self._encoder.pc.enc_pos1_sync, 1)
        set_and_wait(self._encoder.pc.enc_pos2_sync, 0)
        set_and_wait(self._encoder.pc.enc_pos3_sync, 0)
        set_and_wait(self._encoder.pc.enc_pos4_sync, 0)

        ## SYS tab
        set_and_wait(self._encoder.output1.ttl.addr, 53)  # PC_PULSE --> TTL1 --> Camera
        set_and_wait(
            self._encoder.output2.ttl.addr, 52
        )  # PC_PULSE --> TTL2 --> fast shutter
        set_and_wait(self._encoder.output3.ttl.addr, 0)
        set_and_wait(self._encoder.output4.ttl.addr, 0)

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

    def preset_zebra(self, pc_cfg={}):
        ############### PC Arm
        yield from abs_set(
            self._encoder.pc.trig_source, 0, wait=True
        )  # 0 = Soft, 1 = External
        ############### PC Pulse
        # yield from abs_set(self._encoder.pc.pulse_width, self.dft_pulse_wid[self.tspre], wait=True)

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
        # self.set_stage_sigs()
        self._stage_with_delay()
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

        print("\nin complet: complete starts")

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
            export_zebra_data(self._encoder, self.__write_filepath)

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
        print(
            f"\nin complete: {type(self._last_bulk)=}\n{type(self._document_cache)=}\n"
        )
        print(f"\nin complete: {(self._last_bulk)=}\n{(self._document_cache)=}\n")

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
        for d in self.detectors:
            desc.update(d.describe())
        # print(f"\nin describe_collect {desc=}\n")

        # # Handle the ion chamber that the zebra is collecting
        # desc["i0"] = spec
        # desc["i0"]["source"] = self._sis.mca2.pvname
        # desc["i0_time"] = spec
        # desc["i0_time"]["source"] = self._sis.mca1.pvname
        # desc["im"] = spec
        # desc["im"]["source"] = self._sis.mca3.pvname
        # desc["it"] = spec
        # desc["it"]["source"] = self._sis.mca4.pvname

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
        yield from FXITomoFlyer.init_mot_r(scn_cfg)
        yield from FXITomoFlyer.set_cam_mode(self.detectors[0], stage="pre-scan")
        scn_cfg = FXITomoFlyer.cal_cam_rot_params(self.detectors[0], scn_cfg)
        pc_cfg = FXITomoFlyer.cal_zebra_pc_params(scn_cfg)
        yield from self.preset_zebra(pc_cfg)
        print("preset_flyer is done")
        return scn_cfg, pc_cfg

    def set_pc_step_for_scan(self, scn_mode, pc_cfg):
        yield from abs_set(self.encoder.pc.dir, pc_cfg[scn_mode]["dir"], wait=True)
        yield from abs_set(
            self.encoder.pc.gate_start, pc_cfg[scn_mode]["gate_start"], wait=True
        )

    # @staticmethod
    # def set_wait(field, val, wait=0.01):
    #     set_and_wait(field, val, poll_time=wait)
    #     # while field.get() != val:
    #     #     field.put(val, timeout=10)
    #     #     ttime.sleep(0.01)
    #     # ttime.sleep(wait)

    @classmethod
    def cal_cam_rot_params(cls, det, scn_cfg):
        """_summary_

        Args:
            det (str): choose from the set {"Andor", "Marana", "Oryx"}
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
        ############### calculate detector parameters ###############
        acq_p = FXITomoFlyer.check_cam_exp(det, scn_cfg["acq_p"], scn_cfg["bin_fac"])

        if acq_p > scn_cfg["acq_p"]:
            print(
                "Acquisition period is too small for the camera. Reset acquisition period to minimum allowed exposure time."
            )
            scn_cfg["acq_p"] = acq_p

        if scn_cfg["exp_t"] > acq_p - 0.002:
            scn_cfg["exp_t"] = acq_p - 0.002

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

        taxi_dist = np.ceil(
            (scn_cfg["vel"] - cls.rot_axis.base_velo.get()) * scn_cfg["tacc"] / 2
        )
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

        scn_cfg["taxi_dist"] = taxi_dist
        scn_cfg["ang_step"] = scn_cfg["vel"] * scn_cfg["acq_p"]
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

        return scn_cfg

    @staticmethod
    def check_cam_exp(det, acq_p, bin_fac):
        acq_min = DET_MIN_AP[det.name] / BIN_FACS[det.name][bin_fac]
        acq_p = max(acq_min, acq_p)
        return acq_p

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
            pc_cfg["standard"]["gate_start"] = scn_cfg["ang_s"]
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
            # pc_cfg["snaked: multiple files"]["gate_start"] = scn_cfg["ang_s"]
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
            # pc_cfg["snaked: single file"]["gate_start"] = scn_cfg["ang_s"]
            # pc_cfg["snaked: single file"]["gate_width"] =
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
    def prime_det(det):
        if not list(det.hdf5.array_size.get()):
            if det.cam.trigger_mode.get() != 0:
                yield from abs_set(det.cam.trigger_mode, 0, wait=True)
            if det.cam.image_mode.get() != 0:
                yield from abs_set(det.cam.image_mode, 0, wait=True)
            yield from abs_set(det.cam.num_images, 5, wait=True)
            yield from abs_set(det.cam.acquire, 1, wait=False)

    @staticmethod
    def bin_det(det, bin_fac):
        yield from abs_set(det.cam.acquire, 0, wait=False)
        if bin_fac is None:
            bin_fac = 0
        if int(bin_fac) not in [0, 1, 2, 3, 4]:
            raise ValueError("binnng must be in [0, 1, 2, 3, 4]")
        yield from abs_set(det.binning, bin_fac, wait=True)
        FXITomoFlyer.prime_det(det)

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
        yield from abs_set(zps.pi_r.offset_freeze_switch, 1)
        ang_max = max(scn_cfg["ang_s"], scn_cfg["ang_e"])

        if ang_max > 0:
            yield from abs_set(zps.pi_r.user_offset, np.ceil(ang_max / 360) * 360)

        if abs(scn_cfg["ang_s"] - cur_pos) > 360:
            cur_pos = scn_cfg["ang_s"] // 360 * 360 + cur_pos % 360
            zps.pi_r.set_current_position(cur_pos)
        yield from abs_set(zps.pi_r.acceleration, 1, wait=True)
        yield from abs_set(zps.pi_r.velocity, scn_cfg["mb_vel"], wait=True)
        yield from abs_set(zps.pi_r, scn_cfg["ang_s"], wait=True)

    @staticmethod
    def set_cam_step_for_scan(det, scn_cfg):
        yield from abs_set(det.cam.acquire_time, scn_cfg["exp_t"], wait=True)
        yield from abs_set(det.hdf5.num_capture, scn_cfg["num_images"], wait=True)
        yield from abs_set(det.cam.num_images, scn_cfg["num_images"], wait=True)

    @staticmethod
    def set_mot_r_step_for_scan(scn_cfg):
        yield from abs_set(zps.pi_r.acceleration, scn_cfg["tacc"], wait=True)
        yield from abs_set(zps.pi_r.velocity, scn_cfg["vel"], wait=True)
        yield from abs_set(
            zps.pi_r.user_setpoint,
            scn_cfg["ang_s"] - scn_cfg["rot_dir"] * scn_cfg["taxi_dist"],
            wait=True,
        )

    @staticmethod
    def set_cam_mode(cam, stage="pre-scan"):
        if stage == "pre-scan":
            yield from abs_set(cam.cam.image_mode, 0, wait=True)
            yield from abs_set(cam.cam.trigger_mode, 4, wait=True)
        elif stage == "ref-scan":
            yield from abs_set(cam.cam.image_mode, 0, wait=True)
            yield from abs_set(cam.cam.trigger_mode, 0, wait=True)
        elif stage == "post-scan":
            yield from abs_set(cam.cam.image_mode, 1, wait=True)
            yield from abs_set(cam.cam.trigger_mode, 0, wait=True)


Zebra = FXIZebra(
    "XF:18ID-ES:1{Dev:Zebra1}:",
    name="Zebra",
    read_attrs=["pc.data.enc1", "pc.data.time"],
)

tomo_flyer = FXITomoFlyer(
    list((Andor,)),
    Zebra,
    name="tomo_flyer",
)


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


class ZebraHDF5Handler(HandlerBase):
    HANDLER_NAME = "ZEBRA_HDF51"

    def __init__(self, resource_fn):
        self._handle = h5py.File(resource_fn, "r")

    def __call__(self, *, column):
        return self._handle[column][:]

    def close(self):
        self._handle.close()
        self._handle = None
        super().close()


db.reg.register_handler("ZEBRA_HDF51", ZebraHDF5Handler, overwrite=True)
