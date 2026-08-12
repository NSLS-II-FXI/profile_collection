print(f"Loading {__file__}...")

from ophyd import Device, Component as Cpt, EpicsScaler, EpicsSignal, EpicsSignalRO
from ophyd.device import DynamicDeviceComponent as DDC
from collections import OrderedDict


# The source for the scaler code is github.com/NSLS-II-SRX/profile_collection/startup/30-scaler.py


class EpicsSignalROLazyier(EpicsSignalRO):
    def get(self, *args, timeout=5, **kwargs):
        return super().get(*args, timeout=timeout, **kwargs)


def _scaler_fields(attr_base, field_base, range_, **kwargs):
    defn = OrderedDict()
    for i in range_:
        attr = "{attr}{i}".format(attr=attr_base, i=i)
        suffix = "{field}{i}".format(field=field_base, i=i)
        defn[attr] = (EpicsSignalROLazyier, suffix, kwargs)

    return defn


class FXIScaler(EpicsScaler):
    acquire_mode = Cpt(EpicsSignal, "AcquireMode")
    acquiring = Cpt(EpicsSignal, "Acquiring")
    asyn = Cpt(EpicsSignal, "Asyn")
    channel1_source = Cpt(EpicsSignal, "Channel1Source")
    channel_advance = Cpt(EpicsSignal, "ChannelAdvance", string=True)
    channels = DDC(_scaler_fields("chan", ".S", range(1, 33)))
    client_wait = Cpt(EpicsSignal, "ClientWait")
    count_on_start = Cpt(EpicsSignal, "CountOnStart")
    current_channel = Cpt(EpicsSignal, "CurrentChannel")
    disable_auto_count = Cpt(EpicsSignal, "DisableAutoCount")
    do_read_all = Cpt(EpicsSignal, "DoReadAll")
    dwell = Cpt(EpicsSignal, "Dwell")
    elapsed_real = Cpt(EpicsSignal, "ElapsedReal")
    enable_client_wait = Cpt(EpicsSignal, "EnableClientWait")
    erase_all = Cpt(EpicsSignal, "EraseAll")
    erase_start = Cpt(EpicsSignal, "EraseStart")
    firmware = Cpt(EpicsSignal, "Firmware")
    hardware_acquiring = Cpt(EpicsSignal, "HardwareAcquiring")
    input_mode = Cpt(EpicsSignal, "InputMode")
    max_channels = Cpt(EpicsSignal, "MaxChannels")
    model = Cpt(EpicsSignal, "Model")
    mux_output = Cpt(EpicsSignal, "MUXOutput")
    nuse_all = Cpt(EpicsSignal, "NuseAll")
    output_mode = Cpt(EpicsSignal, "OutputMode")
    output_polarity = Cpt(EpicsSignal, "OutputPolarity")
    prescale = Cpt(EpicsSignal, "Prescale")
    preset_real = Cpt(EpicsSignal, "PresetReal")
    read_all = Cpt(EpicsSignal, "ReadAll")
    read_all_once = Cpt(EpicsSignal, "ReadAllOnce")
    set_acquiring = Cpt(EpicsSignal, "SetAcquiring")
    set_client_wait = Cpt(EpicsSignal, "SetClientWait")
    snl_connected = Cpt(EpicsSignal, "SNL_Connected")
    software_channel_advance = Cpt(EpicsSignal, "SoftwareChannelAdvance")
    count_mode = Cpt(EpicsSignal, ".CONT")
    start_all = Cpt(EpicsSignal, "StartAll")
    stop_all = Cpt(EpicsSignal, "StopAll")
    user_led = Cpt(EpicsSignal, "UserLED")
    wfrm = Cpt(EpicsSignal, "Wfrm")
    mca1 = Cpt(EpicsSignalRO, "mca1")
    mca2 = Cpt(EpicsSignalRO, "mca2")
    mca3 = Cpt(EpicsSignalRO, "mca3")
    mca4 = Cpt(EpicsSignalRO, "mca4")

    def __init__(self, prefix, **kwargs):
        super().__init__(prefix, **kwargs)
        self.stage_sigs[self.count_mode] = "OneShot"


sclr1 = FXIScaler("XF:18IDB-ES{Sclr:1}", name="sclr1")


class SR570(Device):
    # SR570 preamps are controlled via one-way RS232 connection. The IOC can keep track only of the
    # settings change via EPICS. It does not know the actual settings if the changes are made
    # manually using buttons on the hardware unit.

    init = Cpt(EpicsSignal, "init.PROC")
    reset = Cpt(EpicsSignal, "reset.PROC")

    sensitivity_num = Cpt(EpicsSignal, "sens_num", string=True)
    sensitivity_unit = Cpt(EpicsSignal, "sens_unit", string=True)

    offset_on = Cpt(EpicsSignal, "offset_on", string=True)
    offset_sign = Cpt(EpicsSignal, "offset_sign", string=True)
    offset_num = Cpt(EpicsSignal, "offset_num", string=True)
    offset_unit = Cpt(EpicsSignal, "offset_unit", string=True)
    offset_u_put = Cpt(
        EpicsSignal,
        "off_u_put",
    )
    offset_u_tweak = Cpt(EpicsSignal, "offset_u_tweak")
    offset_cal = Cpt(EpicsSignal, "offset_cal", string=True)

    bias_put = Cpt(EpicsSignal, "bias_put")
    bias_tweak = Cpt(EpicsSignal, "bias_tweak")
    bias_on = Cpt(EpicsSignal, "bias_on", string=True)

    filter_type = Cpt(EpicsSignal, "filter_type", string=True)
    filter_reset = Cpt(EpicsSignal, "filter_reset.PROC")
    filter_low_freq = Cpt(EpicsSignal, "low_freq", string=True)
    filter_high_freq = Cpt(EpicsSignal, "high_freq", string=True)

    gain_mode = Cpt(EpicsSignal, "gain_mode", string=True)
    invert_on = Cpt(EpicsSignal, "invert_on", string=True)
    blank_on = Cpt(EpicsSignal, "blank_on", string=True)


class SR570_PREAMPS(Device):
    unit1 = Cpt(SR570, "{SR570:1}")
    unit2 = Cpt(SR570, "{SR570:2}")
    unit3 = Cpt(SR570, "{SR570:3}")
    unit4 = Cpt(SR570, "{SR570:4}")


sr570_preamps = SR570_PREAMPS("XF:18IDB-CT", name="sr570_preamps")


class WienerHVCrateChannel(Device):
    status_dec = Cpt(EpicsSignalRO, "StatusDec")  # Bits 0-7

    # Meanings of the status bits:
    # outputOn (0)                              output channel is on
    # outputInhibit(1)                          external (hardware-)inhibit of the output channel
    # outputFailureMinSenseVoltage (2)          Sense voltage is too low
    # outputFailureMaxSenseVoltage (3)          Sense voltage is too high
    # outputFailureMaxTerminalVoltage (4)       Terminal voltage is too high
    # outputFailureMaxCurrent (5)               Current is too high
    # outputFailureMaxTemperature (6)           Heat sink temperature is too high
    # outputFailureMaxPower (7)                 Output power is too high

    switch_on_off = Cpt(EpicsSignal, "Switch")
    V_set = Cpt(EpicsSignal, "V-Set")
    V_sense = Cpt(EpicsSignalRO, "V-Sense")
    I_set_limit = Cpt(EpicsSignal, "I-SetLimit")
    I_sense = Cpt(EpicsSignalRO, "I-Sense")
    temperature = Cpt(EpicsSignalRO, "Temperature")
    V_fall_rate = Cpt(EpicsSignal, "V-FallRate")
    V_rise_rate = Cpt(EpicsSignal, "V-RiseRate")


class WienerHVCrate(Device):
    u0 = Cpt(WienerHVCrateChannel, "HV:u0}")
    u1 = Cpt(WienerHVCrateChannel, "HV:u1}")
    u2 = Cpt(WienerHVCrateChannel, "HV:u2}")
    u3 = Cpt(WienerHVCrateChannel, "HV:u3}")
    u4 = Cpt(WienerHVCrateChannel, "HV:u4}")
    u5 = Cpt(WienerHVCrateChannel, "HV:u5}")
    u6 = Cpt(WienerHVCrateChannel, "HV:u6}")
    u7 = Cpt(WienerHVCrateChannel, "HV:u7}")


hv_crate = WienerHVCrate("XF:18IDB-OP{WPS:01-", name="hv_crate")
