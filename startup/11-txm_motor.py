from ophyd import (EpicsMotor, EpicsSignalRO, Device, Component as Cpt)
from ophyd import EpicsSignal

from nslsii.devices import TwoButtonShutter

class MyBaseMotor(EpicsMotor):
    dial_readback = Cpt(EpicsSignalRO, '.DRBV')
    dial_counts = Cpt(EpicsSignalRO, '.RRBV')
    motor_res = Cpt(EpicsSignalRO, '.MRES')
    encoder_res = Cpt(EpicsSignalRO, '.ERES')
    motor_stat = Cpt(EpicsSignalRO, '.STAT')

class MyEpicsMotor(MyBaseMotor):
    
    def stop(self, success=False):
        self.user_setpoint.set(self.user_readback.get())
        Device.stop(self, success=success)   
	

class Condenser(Device):
    x = Cpt(MyEpicsMotor, '{CLens:1-Ax:X}Mtr')
    y1 = Cpt(MyEpicsMotor, '{CLens:1-Ax:Y1}Mtr')
    y2 = Cpt(MyEpicsMotor, '{CLens:1-Ax:Y2}Mtr')
    z1 = Cpt(MyEpicsMotor, '{CLens:1-Ax:Z1}Mtr')
    z2 = Cpt(MyEpicsMotor, '{CLens:1-Ax:Z2}Mtr')
    p = Cpt(MyEpicsMotor, '{CLens:1-Ax:P}Mtr')  

class Zoneplate(Device):
    x = Cpt(MyEpicsMotor, '{ZP:1-Ax:X}Mtr')
    y = Cpt(MyEpicsMotor, '{ZP:1-Ax:Y}Mtr')
    z = Cpt(MyBaseMotor, '{TXM-ZP:1-Ax:Z}Mtr')

class Aperture(Device):
    x = Cpt(MyEpicsMotor, '{Aper:1-Ax:X}Mtr')
    y = Cpt(MyEpicsMotor, '{Aper:1-Ax:Y}Mtr')
    z = Cpt(MyBaseMotor, '{TXM-Aper:1-Ax:Z}Mtr')

class PhaseRing(Device):
    x = Cpt(MyEpicsMotor, '{PR:1-Ax:X}Mtr')
    y = Cpt(MyEpicsMotor, '{PR:1-Ax:Y}Mtr')
    z = Cpt(MyBaseMotor, '{TXM-PH:1-Ax:Z}Mtr')

class BetrandLens(Device):
    x = Cpt(MyEpicsMotor, '{BLens:1-Ax:X}Mtr')
    y = Cpt(MyEpicsMotor, '{BLens:1-Ax:Y}Mtr')
    z = Cpt(MyBaseMotor, '{BLens:1-Ax:Z}Mtr')

class TXMSampleStage(Device):
    sx = Cpt(MyEpicsMotor, '{Env:1-Ax:Xl}Mtr')
    sy = Cpt(MyEpicsMotor, '{Env:1-Ax:Yl}Mtr')
    sz = Cpt(MyEpicsMotor, '{Env:1-Ax:Zl}Mtr')
    pi_x = Cpt(MyBaseMotor, '{TXM:1-Ax:X}Mtr')
    pi_r = Cpt(MyEpicsMotor, '{TXM:1-Ax:R}Mtr')


class DetSupport(Device):
    x = Cpt(MyEpicsMotor, '-Ax:X}Mtr')
    y = Cpt(MyEpicsMotor, '-Ax:Y}Mtr')
    z = Cpt(MyBaseMotor, '-Ax:Z}Mtr')

class TXM_SSA(Device):
    v_gap = Cpt(MyBaseMotor, '-Ax:Vgap}Mtr')
    v_ctr = Cpt(MyBaseMotor, '-Ax:Vctr}Mtr')
    h_gap = Cpt(MyBaseMotor, '-Ax:Hgap}Mtr')
    h_ctr = Cpt(MyBaseMotor, '-Ax:Hctr}Mtr')




ssa = TXM_SSA('XF:18IDB-OP{SSA:1', name='ssa')

DetU = DetSupport('XF:18IDB-OP{DetS:U', name='DetU')
DetD = DetSupport('XF:18IDB-OP{DetS:D', name='DetD')

clens = Condenser('XF:18IDB-OP', name='clens')
aper = Aperture('XF:18IDB-OP', name='aper')
zp = Zoneplate('XF:18IDB-OP', name='zp')
phase_ring = PhaseRing('XF:18IDB-OP', name='phase_ring')
betr = BetrandLens('XF:18IDB-OP', name='betr')
zps = TXMSampleStage('XF:18IDB-OP', name='zps')
XEng = MyEpicsMotor('XF:18IDA-OP{Mono:DCM-Ax:En}Mtr', name='XEng')

th2_motor = MyEpicsMotor('XF:18IDA-OP{Mono:DCM-Ax:Th2}Mtr', name='th2_motor')
th2_feedback = EpicsSignal('XF:18IDA-OP{Mono:DCM-Ax:Th2}PID', name='th2_feedback')
th2_feedback_enable = EpicsSignal('XF:18IDA-OP{Mono:DCM-Ax:Th2}PID.FBON', name='th2_feedback_enable')

shutter_open = EpicsSignal('XF:18IDA-PPS{PSh}Cmd:Opn-Cmd', name='shutter_open')
shutter_close = EpicsSignal('XF:18IDA-PPS{PSh}Cmd:Cls-Cmd', name='shutter_close')
shutter_status = EpicsSignal('XF:18IDA-PPS{PSh}Pos-Sts', name='shutter_status')

shutter = TwoButtonShutter('XF:18IDA-PPS{PSh}', name='shutter')

filter1 = EpicsSignal('XF:18IDB-UT{Fltr:1}Cmd:In-Cmd', name='filter1')
filter2 = EpicsSignal('XF:18IDB-UT{Fltr:2}Cmd:In-Cmd', name='filter2')
filter3 = EpicsSignal('XF:18IDB-UT{Fltr:3}Cmd:In-Cmd', name='filter3')
filter4 = EpicsSignal('XF:18IDB-UT{Fltr:4}Cmd:In-Cmd', name='filter4')

motor_txm = [clens.x, clens.y1, clens.y2, clens.z1, clens.z2, clens.p,
             aper.x, aper.y, aper.z,
             zp.x, zp.y, zp.z,
             phase_ring.x, phase_ring.y, phase_ring.z,
             betr.x, betr.y, betr.z,
             zps.sx, zps.sy, zps.sz, zps.pi_x, zps.pi_r,
             DetU.x, DetU.y, DetU.z,
             DetD.x, DetD.y, DetD.z,
             ssa.v_gap, ssa.v_ctr, ssa.h_gap, ssa.h_ctr,
             XEng]

