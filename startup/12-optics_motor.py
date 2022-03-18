# from ophyd import (EpicsMotor, Device, Component as Cpt)


class mirror(Device):
    x = Cpt(MyEpicsMotor, "-Ax:X}Mtr")
    yaw = Cpt(MyEpicsMotor, "-Ax:Yaw}Mtr")
    y = Cpt(MyEpicsMotor, "-Ax:Y}Mtr")
    p = Cpt(MyEpicsMotor, "-Ax:P}Mtr")
    r = Cpt(MyEpicsMotor, "-Ax:R}Mtr")
    xu = Cpt(MyEpicsMotor, "-Ax:XU}Mtr")
    xd = Cpt(MyEpicsMotor, "-Ax:XD}Mtr")
    yu = Cpt(MyEpicsMotor, "-Ax:YU}Mtr")
    ydi = Cpt(MyEpicsMotor, "-Ax:YDI}Mtr")
    ydo = Cpt(MyEpicsMotor, "-Ax:YDO}Mtr")


class DCM(Device):
    th1 = Cpt(MyEpicsMotor, "-Ax:Th1}Mtr")
    dy2 = Cpt(MyEpicsMotor, "-Ax:dY2}Mtr")
    th2 = Cpt(MyEpicsMotor, "-Ax:Th2}Mtr")
    chi2 = Cpt(MyEpicsMotor, "-Ax:Chi2}Mtr")
    eng = Cpt(MyEpicsMotor, "-Ax:En}Mtr")


class PBSL(Device):
    x_gap = Cpt(MyEpicsMotor, "-Ax:XGap}Mtr")
    y_gap = Cpt(MyEpicsMotor, "-Ax:YGap}Mtr")
    x_ctr = Cpt(MyEpicsMotor, "-Ax:XCtr}Mtr")
    y_ctr = Cpt(MyEpicsMotor, "-Ax:YCtr}Mtr")
    top = Cpt(MyEpicsMotor, "-Ax:T}Mtr")
    bot = Cpt(MyEpicsMotor, "-Ax:B}Mtr")
    ob = Cpt(MyEpicsMotor, "-Ax:O}Mtr")
    ib = Cpt(MyEpicsMotor, "-Ax:I}Mtr")


cm = mirror("XF:18IDA-OP{Mir:CM", name="cm")
tm = mirror("XF:18IDA-OP{Mir:TM", name="tm")
dcm = DCM("XF:18IDA-OP{Mono:DCM", name="dcm")
pbsl = PBSL("XF:18IDA-OP{PBSL:1", name="pbsl")

motor_optics = [
    cm.x,
    cm.yaw,
    cm.y,
    cm.p,
    cm.r,
    cm.xu,
    cm.xd,
    cm.yu,
    cm.ydi,
    cm.ydo,
    tm.x,
    tm.yaw,
    tm.y,
    tm.p,
    tm.r,
    tm.xu,
    tm.xd,
    tm.yu,
    tm.ydi,
    tm.ydo,
    dcm.th1,
    dcm.dy2,
    dcm.th2,
    dcm.chi2,
    dcm.eng,
    pbsl.x_gap,
    pbsl.y_gap,
    pbsl.x_ctr,
    pbsl.y_ctr,
    pbsl.top,
    pbsl.bot,
    pbsl.ob,
    pbsl.ib,
]

get_ipython().register_magics(BlueskyMagics)
# BlueskyMagics.positioners = motor_txm + motor_optics
