import uuid
import sys
import time
from warnings import warn
from bluesky.plan_stubs import mv, mvr, abs_set
from ophyd.sim import motor1 as fake_motor
from ophyd.sim import motor2 as fake_x_motor
import subprocess
import pandas as pd
import numpy as np
import warnings


try:
    from cytools import partition
except ImportError:
    from toolz import partition
from bluesky import plan_patterns
from bluesky.utils import Msg, short_uid as _short_uid, make_decorator

warnings.filterwarnings("ignore")


#########################
def _move_sample_out(out_x, out_y, out_z, out_r, repeat=1, rot_first_flag=1):
    """
    move out by relative distance
    """
    """
    if relative_move_flag:
        x_out = zps.sx.position + out_x
        y_out = zps.sy.position + out_y
        z_out = zps.sz.position + out_z
        r_out = zps.pi_r.position + out_r
    else:
    """
    x_out = out_x
    y_out = out_y
    z_out = out_z
    r_out = out_r

    for i in range(repeat):
        if rot_first_flag:
            yield from mv(zps.pi_r, r_out)
            yield from mv(zps.sx, x_out, zps.sy, y_out, zps.sz, z_out)
        else:
            yield from mv(zps.sx, x_out, zps.sy, y_out, zps.sz, z_out)
            yield from mv(zps.pi_r, r_out)


def _move_sample_in(in_x, in_y, in_z, in_r, repeat=1, trans_first_flag=1):
    """
    move in at absolute position
    """
    for i in range(repeat):
        if trans_first_flag:
            yield from mv(zps.sx, in_x, zps.sy, in_y, zps.sz, in_z)
            yield from mv(zps.pi_r, in_r)
        else:
            yield from mv(zps.pi_r, in_r)
            yield from mv(zps.sx, in_x, zps.sy, in_y, zps.sz, in_z)


# def _close_shutter(simu=False):
#    if simu:
#        print('testing: close shutter')
#    else:
#        print('closing shutter ... ')
#        # yield from mv(shutter, 'Close')
#        i = 0
#        reading = (yield from bps.rd(shutter_status))
#        while not reading: # if 1:  closed; if 0: open
#            yield from abs_set(shutter_close, 1, wait=True)
#            yield from bps.sleep(5)
#            i += 1
#            print(f'try closing again ...')
#            if i > 5:
#                print('fails to close shutter')
#                raise Exception('fails to close shutter')
#                break
#        #yield from abs_set(shutter_close, 1)
#        #yield from bps.sleep(1)


def _close_shutter(simu=False):
    if simu:
        print("testing: close shutter")
    else:
        yield from mv(shutter, 'Close')

'''
def _close_shutter(simu=False):
    if simu:
        print("testing: close shutter")
    else:
        print("closing shutter ... ")
        # yield from mv(shutter, 'Close')
        i = 0
        reading = (yield from bps.rd(shutter_status))
        while not reading:  # if 1:  closed; if 0: open
            #yield from abs_set(shutter_open, 1, wait=True)
            #yield from bps.sleep(2)
            yield from abs_set(shutter_close, 1, wait=True)
            yield from bps.sleep(3)
            i += 1
            print(f"try closing again ...")
            if i > 20:
                print("fails to close shutter")
                raise Exception("fails to close shutter")
                break
        # yield from abs_set(shutter_close, 1)
        # yield from bps.sleep(1)
'''

def _open_shutter(simu=False):
    if simu:
        print("testing: open shutter")
    else:
        yield from mv(shutter, 'Open')


'''
def _open_shutter(simu=False):
    if simu:
        print("testing: open shutter")
    else:
        print("opening shutter ... ")
        # yield from mv(shutter, 'Open')
        i = 0
        reading = (yield from bps.rd(shutter_status))
        while reading:  # if 1:  closed; if 0: open
            yield from abs_set(shutter_open, 1, wait=True)
            yield from bps.sleep(1)
            i += 1
            if i > 5:
                print("fails to open shutter")
                raise Exception("fails to open shutter")
                break
        # yield from abs_set(shutter_open, 1)
        # yield from bps.sleep(1)
'''

def _set_andor_param(exposure_time=0.1, period=0.1, chunk_size=1):
    #    yield from mv(Andor.cam.acquire, 0)
    #    yield from mv(Andor.cam.image_mode, 0)
    #    yield from mv(Andor.cam.num_images, chunk_size)
    #    yield from mv(Andor.cam.acquire_time, exposure_time)

    yield from abs_set(Andor.cam.acquire, 0, wait=True)
    yield from abs_set(Andor.cam.image_mode, 0, wait=True)
    yield from abs_set(Andor.cam.num_images, chunk_size, wait=True)
    yield from abs_set(Andor.cam.acquire_time, exposure_time, wait=True)
    yield from abs_set(Andor.cam.acquire_period, period)
#    Andor.cam.acquire_period.put(period)


def _set_rotation_speed(rs=1):
    yield from abs_set(zps.pi_r.velocity, rs)


def _take_image(detectors, motor, num, stream_name='primary'):
    if not (type(detectors) == list):
        detectors = list(detectors)
    if not (type(motor) == list):
        motor = list(motor)
    for i in range(num):
        yield from trigger_and_read(detectors + motor, name=stream_name)


def _take_dark_image(detectors, motor, num_dark=1, simu=False):
    yield from _close_shutter(simu)
    yield from _take_image(detectors, motor, num_dark)#, stream_name='dark')


def _take_bkg_image(
    out_x,
    out_y,
    out_z,
    out_r,
    detectors,
    motor,
    num_bkg=1,
    simu=False,
    traditional_sequence_flag=1,
):
    yield from _move_sample_out(
        out_x, out_y, out_z, out_r, repeat=2, rot_first_flag=traditional_sequence_flag
    )
    yield from _take_image(detectors, motor, num_bkg)#, stream_name='flat')


def _xanes_per_step(eng, detectors, motor, move_flag=1, move_clens_flag=1, info_flag=0):
    yield from move_zp_ccd(
        eng, move_flag=move_flag, move_clens_flag=move_clens_flag, info_flag=info_flag
    )
    yield from bps.sleep(0.1)
    if not (type(detectors) == list):
        detectors = list(detectors)
    if not (type(motor) == list):
        motor = list(motor)
    yield from trigger_and_read(detectors + motor)


#################################


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        results = method(*args, **kw)
        te = time.time()
        print("Time for {0:s}: {1:2.2f}".format(method.__name__, te - ts))
        return results

    return timed


def tomo_scan(
    start,
    stop,
    num,
    exposure_time=1,
    bkg_num=10,
    dark_num=10,
    chunk_size=1,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    relative_move_flag=1,
    traditional_sequence_flag=1,
    note="",
    simu=False,
    md=None,
):
    """
    Script for running Tomography scan
    Use as: RE(tomo_scan(start, stop, num, exposure_time=1, bkg_num=10, dark_num=10, out_x=0, out_y=0, note='', md=None))

    Input:
    ------
    start: start angle
    stop: stop angle
    num: number of scan angles
    exposure time: second (default: 0.1)
    bkg_num: number of background image to be taken
    dark_num: number of dark image to be taken
    out_x: relative movement of zps.sx stage where sample is out (um)
    out_y: relative movement of zps.sy stage where sample is out (um)
    note: string
    md: metadate (default: None)
    """
    global ZONE_PLATE

    detectors = [Andor, ic3]
    yield from _set_andor_param(
        exposure_time=exposure_time, period=exposure_time, chunk_size=chunk_size
    )

    motor_eng = XEng
    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if out_x else motor_x_ini
        motor_y_out = motor_y_ini + out_y if out_y else motor_y_ini
        motor_z_out = motor_z_ini + out_z if out_z else motor_z_ini
        motor_r_out = motor_r_ini + out_r if out_r else motor_r_ini
    else:
        motor_x_out = out_x if out_x else motor_x_ini
        motor_y_out = out_y if out_y else motor_y_ini
        motor_z_out = out_z if out_z else motor_z_ini
        motor_r_out = out_r if out_r else motor_r_ini

    motor = [motor_eng, zps.sx, zps.sy, zps.sz, zps.pi_r]
    _md = {
        "detectors": [det.name for det in detectors],
        "x_ray_energy": XEng.position,
        "num_angles": num,
        "num_bkg_images": bkg_num,
        "num_dark_images": dark_num,
        "plan_args": {
            "start": start,
            "stop": stop,
            "num": num,
            "exposure_time": exposure_time,
            "bkg_num": bkg_num,
            "dark_num": dark_num,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "relative_move_flag": relative_move_flag,
            "note": note if note else "None",
        },
        "zone_plate": ZONE_PLATE,
        "plan_name": "tomo_scan",
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "zone_plate": ZONE_PLATE,
        "note": note if note else "None",
        # 'motor_pos':  wh_pos(print_on_screen=0),
    }

    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)
    steps = np.linspace(start, stop, num)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def tomo_inner_scan():
        # close shutter, dark images
        print("\nshutter closed, taking dark images...")
        yield from _close_shutter(simu)
        yield from _take_dark_image(detectors, motor, num_dark=dark_num, simu=simu)
        # Open shutter, tomo images
        yield from _open_shutter(simu)
        print("shutter opened, starting tomo_scan...")
        for step in steps:  # take tomography images
            # yield from one_1d_step(detectors, zps.pi_r, step)
            yield from mv(zps.pi_r, step)
            yield from _take_image(detectors, motor, 1)

        print("\n\nTaking background images...")
        yield from _take_bkg_image(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            detectors,
            motor,
            num_bkg=1,
            simu=False,
            traditional_sequence_flag=traditional_sequence_flag,
        )
        # close shutter, move sample back
        yield from _close_shutter(simu)
        yield from _move_sample_in(
            motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini, repeat=2
        )

    yield from tomo_inner_scan()
    print("tomo-scan is disabled, try to use fly_scan")


def xanes_scan(
    eng_list,
    exposure_time=0.1,
    chunk_size=5,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    simu=False,
    relative_move_flag=1,
    note="",
    rot_first_flag=1,
    md=None,
):
    """
    Scan the energy and take 2D image, will take background after take all images for all energy points
    Example: RE(xanes_scan([8.9, 9.0, 9.1], exposure_time=0.1, bkg_num=10, dark_num=10, out_x=1, out_y=0, note='xanes scan test'))

    Inputs:
    -------
    eng_list: list or numpy array,
        energy in unit of keV

    exposure_time: float
        in unit of seconds

    chunk_size: int
        number of background images == num of dark images

    out_x: float, default is 0
        relative movement of sample in "x" direction using zps.sx to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_y: float, default is 0
        relative movement of sample in "y" direction using zps.sy to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_z: float, default is 0
        relative movement of sample in "z" direction using zps.sz to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_r: float, default is 0
        relative movement of sample by rotating "out_r" degrees, using zps.pi_r to move out sample
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    note: string
        adding note to the scan

    simu: Bool, default is False
        True: will simulate closing/open shutter without really closing/opening
        False: will really close/open shutter


    """
    global ZONE_PLATE
    detectors = [Andor, ic3]
    period = exposure_time if exposure_time >= 0.05 else 0.05
    yield from _set_andor_param(exposure_time, period, chunk_size)
    motor_eng = XEng
    eng_ini = XEng.position

    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if not (out_x is None) else motor_x_ini
        motor_y_out = motor_y_ini + out_y if not (out_y is None) else motor_y_ini
        motor_z_out = motor_z_ini + out_z if not (out_z is None) else motor_z_ini
        motor_r_out = motor_r_ini + out_r if not (out_r is None) else motor_r_ini
    else:
        motor_x_out = out_x if not (out_x is None) else motor_x_ini
        motor_y_out = out_y if not (out_y is None) else motor_y_ini
        motor_z_out = out_z if not (out_z is None) else motor_z_ini
        motor_r_out = out_r if not (out_r is None) else motor_r_ini

    rs_ini = (yield from bps.rd(zps.pi_r.velocity))
    motor = [motor_eng, zps.sx, zps.sy, zps.sz, zps.pi_r]

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motor],
        "num_eng": len(eng_list),
        "num_bkg_images": chunk_size,
        "num_dark_images": chunk_size,
        "chunk_size": chunk_size,
        "out_x": out_x,
        "out_y": out_y,
        "out_r": out_z,
        "out_z": out_r,
        "relative_move_flag": relative_move_flag,
        "exposure_time": exposure_time,
        "eng_list": eng_list,
        "XEng": XEng.position,
        "plan_args": {
            "eng_list": "eng_list",
            "exposure_time": exposure_time,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "relative_move_flag": relative_move_flag,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "xanes_scan",
        "hints": {},
        "operator": "FXI",
        "zone_plate": ZONE_PLATE,
        "note": note if note else "None",
        #'motor_pos':  wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def xanes_inner_scan():
        print("\ntake {} dark images...".format(chunk_size))
        yield from _set_rotation_speed(rs=30)
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

        print(
            "\nopening shutter, and start xanes scan: {} images per each energy... ".format(
                chunk_size
            )
        )
        yield from _open_shutter(simu)
        for eng in eng_list:
            yield from _xanes_per_step(eng, detectors, motor, move_flag=1, info_flag=0)
        yield from _move_sample_out(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            repeat=2,
            rot_first_flag=rot_first_flag,
        )
        print(
            "\ntake bkg image after xanes scan, {} per each energy...".format(
                chunk_size
            )
        )
        for eng in eng_list:
            yield from _xanes_per_step(eng, detectors, motor, move_flag=1, info_flag=0)
        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            repeat=2,
            trans_first_flag=rot_first_flag,
        )
        yield from move_zp_ccd(eng_ini, info_flag=0)

        print("closing shutter")
        yield from _close_shutter(simu)

    yield from xanes_inner_scan()
    txt1 = get_scan_parameter()
    eng_list = np.round(eng_list, 5)
    if len(eng_list) > 10:
        txt2 = f"eng_list: {eng_list[0:10]}, ... {eng_list[-5:]}\n"
    else:
        txt2 = f"eng_list: {eng_list}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)
    print(txt)


def xanes_scan2(
    eng_list,
    exposure_time=0.1,
    chunk_size=5,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    simu=False,
    relative_move_flag=1,
    note="",
    flt=[],
    rot_first_flag=1,
    md=None,
):
    """
    Different from xanes_scan:  In xanes_scan2, it moves out sample and take background image at each energy

    Scan the energy and take 2D image
    Example: RE(xanes_scan([8.9, 9.0, 9.1], exposure_time=0.1, bkg_num=10, dark_num=10, out_x=1, out_y=0, note='xanes scan test'))

    Inputs:
    -------
    eng_list: list or numpy array,
        energy in unit of keV

    exposure_time: float
        in unit of seconds

    chunk_size: int
        number of background images == num of dark images

    out_x: float, default is 0
        relative movement of sample in "x" direction using zps.sx to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_y: float, default is 0
        relative movement of sample in "y" direction using zps.sy to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_z: float, default is 0
        relative movement of sample in "z" direction using zps.sz to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_r: float, default is 0
        relative movement of sample by rotating "out_r" degrees, using zps.pi_r to move out sample
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    note: string
        adding note to the scan

    simu: Bool, default is False
        True: will simulate closing/open shutter without really closing/opening
        False: will really close/open shutter

    """
    global ZONE_PLATE
    detectors = [Andor, ic3, ic4]
    period = exposure_time if exposure_time >= 0.05 else 0.05
    yield from _set_andor_param(exposure_time, period, chunk_size)
    motor_eng = XEng
    eng_ini = XEng.position

    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if not (out_x is None) else motor_x_ini
        motor_y_out = motor_y_ini + out_y if not (out_y is None) else motor_y_ini
        motor_z_out = motor_z_ini + out_z if not (out_z is None) else motor_z_ini
        motor_r_out = motor_r_ini + out_r if not (out_r is None) else motor_r_ini
    else:
        motor_x_out = out_x if not (out_x is None) else motor_x_ini
        motor_y_out = out_y if not (out_y is None) else motor_y_ini
        motor_z_out = out_z if not (out_z is None) else motor_z_ini
        motor_r_out = out_r if not (out_r is None) else motor_r_ini

    rs_ini = (yield from bps.rd(zps.pi_r.velocity))

    motor = [motor_eng, zps.sx, zps.sy, zps.sz, zps.pi_r]

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motor],
        "num_eng": len(eng_list),
        "num_bkg_images": chunk_size,
        "num_dark_images": chunk_size,
        "chunk_size": chunk_size,
        "out_x": out_x,
        "out_y": out_y,
        "exposure_time": exposure_time,
        "eng_list": eng_list,
        "XEng": XEng.position,
        "plan_args": {
            "eng_list": "eng_list",
            "exposure_time": exposure_time,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "our_r": out_r,
            "relative_move_flag": relative_move_flag,
            "note": note if note else "None",
            "filters": [t.name for t in flt if flt],
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "xanes_scan2",
        "hints": {},
        "operator": "FXI",
        "zone_plate": ZONE_PLATE,
        "note": note if note else "None",
        #'motor_pos':  wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def xanes_inner_scan():
        yield from _set_rotation_speed(rs=30)
        # yield from abs_set(motor_r.velocity, 30)
        # take dark image
        print("\ntake {} dark images...".format(chunk_size))
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

        print(
            "\nopening shutter, and start xanes scan: {} images per each energy... ".format(
                chunk_size
            )
        )
        yield from _open_shutter(simu)

        for eng in eng_list:
            yield from _xanes_per_step(
                eng, detectors, motor, move_flag=1, move_clens_flag=0, info_flag=0
            )
            if len(flt):
                for filt in flt:
                    yield from mv(filt, 1)
                    yield from bps.sleep(0.5)
            yield from _take_bkg_image(
                motor_x_out,
                motor_y_out,
                motor_z_out,
                motor_r_out,
                detectors,
                motor,
                num_bkg=1,
                simu=simu,
                traditional_sequence_flag=rot_first_flag,
            )
            yield from _move_sample_in(
                motor_x_ini,
                motor_y_ini,
                motor_z_ini,
                motor_r_ini,
                repeat=2,
                trans_first_flag=rot_first_flag,
            )
            if len(flt):
                for filt in flt:
                    yield from mv(filt, 0)
                    yield from bps.sleep(0.5)
        yield from move_zp_ccd(eng_ini, move_flag=1, info_flag=0)
        print("closing shutter")
        yield from _close_shutter(simu=simu)

    yield from xanes_inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    txt1 = get_scan_parameter()
    eng_list = np.round(eng_list, 5)
    if len(eng_list) > 10:
        txt2 = f"eng_list: {eng_list[0:10]}, ... {eng_list[-5:]}\n"
    else:
        txt2 = f"eng_list: {eng_list}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)
    print(txt)


def xanes_scan3(
    eng_list,
    exposure_time=0.1,
    chunk_size=5,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    simu=False,
    relative_move_flag=1,
    note="",
    filters=[],
    rot_first_flag=1,
    md=None,
):
    """
    Different from xanes_scan:  In xanes_scan2, it moves out sample and take background image at each energy
    will move filter in when take bkg image

    Scan the energy and take 2D image
    Example: RE(xanes_scan([8.9, 9.0, 9.1], exposure_time=0.1, bkg_num=10, dark_num=10, out_x=1, out_y=0, note='xanes scan test'))

    Inputs:
    -------
    eng_list: list or numpy array,
        energy in unit of keV

    exposure_time: float
        in unit of seconds

    chunk_size: int
        number of background images == num of dark images

    out_x: float, default is 0
        relative movement of sample in "x" direction using zps.sx to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_y: float, default is 0
        relative movement of sample in "y" direction using zps.sy to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_z: float, default is 0
        relative movement of sample in "z" direction using zps.sz to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_r: float, default is 0
        relative movement of sample by rotating "out_r" degrees, using zps.pi_r to move out sample
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    note: string
        adding note to the scan

    simu: Bool, default is False
        True: will simulate closing/open shutter without really closing/opening
        False: will really close/open shutter

    """
    global ZONE_PLATE
    detectors = [Andor, ic3]
    period = exposure_time if exposure_time >= 0.05 else 0.05
    yield from _set_andor_param(exposure_time, period, chunk_size)
    motor_eng = XEng
    eng_ini = XEng.position

    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if not (out_x is None) else motor_x_ini
        motor_y_out = motor_y_ini + out_y if not (out_y is None) else motor_y_ini
        motor_z_out = motor_z_ini + out_z if not (out_z is None) else motor_z_ini
        motor_r_out = motor_r_ini + out_r if not (out_r is None) else motor_r_ini
    else:
        motor_x_out = out_x if not (out_x is None) else motor_x_ini
        motor_y_out = out_y if not (out_y is None) else motor_y_ini
        motor_z_out = out_z if not (out_z is None) else motor_z_ini
        motor_r_out = out_r if not (out_r is None) else motor_r_ini

    rs_ini = (yield from bps.rd(zps.pi_r.velocity))

    motor = [motor_eng, zps.sx, zps.sy, zps.sz, zps.pi_r]

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motor],
        "num_eng": len(eng_list),
        "num_bkg_images": chunk_size,
        "num_dark_images": chunk_size,
        "chunk_size": chunk_size,
        "out_x": out_x,
        "out_y": out_y,
        "exposure_time": exposure_time,
        "eng_list": eng_list,
        "XEng": XEng.position,
        "plan_args": {
            "eng_list": "eng_list",
            "exposure_time": exposure_time,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "our_r": out_r,
            "relative_move_flag": relative_move_flag,
            "filters": [filt.name for filt in filters],
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "xanes_scan2",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        "zone_plate": ZONE_PLATE,
        #'motor_pos':  wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def xanes_inner_scan():
        yield from _set_rotation_speed(rs=30)
        # yield from abs_set(motor_r.velocity, 30)
        # take dark image
        print("\ntake {} dark images...".format(chunk_size))
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

        print(
            "\nopening shutter, and start xanes scan: {} images per each energy... ".format(
                chunk_size
            )
        )
        yield from _open_shutter(simu)

        for eng in eng_list:
            yield from _xanes_per_step(eng, detectors, motor, move_flag=1, info_flag=0)
            if len(filters):
                for filt in filters:
                    yield from mv(filt, 1)
                    yield from bps.sleep(0.5)
            yield from _take_bkg_image(
                motor_x_out,
                motor_y_out,
                motor_z_out,
                motor_r_out,
                detectors,
                motor,
                num_bkg=1,
                simu=simu,
                traditional_sequence_flag=rot_first_flag,
            )
            if len(filters):
                for filt in filters:
                    yield from mv(filt, 0)
                    yield from bps.sleep(0.5)
            yield from _move_sample_in(
                motor_x_ini,
                motor_y_ini,
                motor_z_ini,
                motor_r_ini,
                repeat=2,
                trans_first_flag=rot_first_flag,
            )
        yield from move_zp_ccd(eng_ini, move_flag=1, info_flag=0)
        print("closing shutter")
        yield from _close_shutter(simu=simu)

    yield from xanes_inner_scan()
    txt1 = get_scan_parameter()
    eng_list = np.round(eng_list, 5)
    if len(eng_list) > 10:
        txt2 = f"eng_list: {eng_list[0:10]}, ... {eng_list[-5:]}\n"
    else:
        txt2 = f"eng_list: {eng_list}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)
    print(txt)


def mv_stage(motor, pos):
    grp = _short_uid("set")
    yield Msg("checkpoint")
    yield Msg("set", motor, pos, group=grp)
    yield Msg("wait", None, group=grp)

'''
def eng_scan(
    eng_list,
    eng_start=8.3,
    eng_end=8.8,
    steps=100,
    num=1,
    detectors=[ic3, ic4],
    delay_time=1,
    note="",
    elem="Ni",
    md=None,
):
    """
    Input:
    ----------
        eng_start: float, energy start in keV

        eng_end: float, energy stop in keV

        steps: int, number of energies

        num: int, number of repeating scans

        detectors: list, detector list, e.g.[ic3, ic4, Andor]

        delay_time: float, delay time after moving motors, in sec
    
    """

    det = [det.name for det in detectors]
    det_name = ""
    for i in range(len(det)):
        det_name += det[i]
        det_name += ", "
    det_name = "[" + det_name[:-2] + "]"
    txt = f"eng_scan(eng_start={eng_start}, eng_end={eng_end}, steps={steps}, num={num}, detectors={det_name}, delay_time={delay_time})\n  Consisting of:\n"
    insert_text(txt)
    print(txt)

    if len(eng_list):
        steps = len(eng_list)
        eng_start, eng_end = eng_list[0], eng_list[-1]

    check_eng_range([eng_start, eng_end])
    #    set_ic_dwell_time(dwell_time=dwell_time)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for i in range(num):
        #      yield from scan([ic3, ic4], XEng, eng_start/1000, eng_end/1000, steps)
        yield from eng_scan_delay(
            eng_start,
            eng_end,
            steps,
            detectors,
            delay_time=delay_time,
            note="",
        )
        h = db[-1]
        y0 = np.array(list(h.data(detectors[0].name)))
        y1 = np.array(list(h.data(detectors[1].name)))

        r = np.log(y0 / y1)
        if not len(eng_list):
            x = np.linspace(eng_start, eng_end, steps)
        else:
            x = eng_list

        ax1.plot(x, r, ".-")
        r_dif = np.array([0] + list(np.diff(r)))
        ax2.plot(x, r_dif, ".-")

    ax1.title.set_text("ratio of: {0}/{1}".format(detectors[0].name, detectors[1].name))
    ax2.title.set_text("differential of: {0}/{1}".format(detectors[0].name, detectors[1].name))
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    txt_finish = '## "eng_scan()" finished'
    insert_text(txt_finish)
    spec = np.zeros([steps, 2])
    spec[:, 0] = x
    spec[:, 1] = r
    np.savetxt(f"{elem}_spec.txt", spec)
'''

def eng_scan(
    start, stop=None, num=1, detectors=[ic3, ic4], delay_time=1, note="", md=None
):
    """

    Input:
    ----------
    start: float, energy start in keV

    end: float, energy stop in keV

    num: int, number of energies

    detectors: list, detector list, e.g.[ic3, ic4, Andor]

    delay_time: float, delay time after moving motors, in sec

    note: string    

    """
    global ZONE_PLATE
    # detectors=[ic3, ic4]
    motor_x = XEng
    motor_x_ini = motor_x.position  # initial position of motor_x
    if isinstance(start, (list, np.ndarray)):
        steps = start
        num = len(start)
        stop = start[-1]
        start = start[0]
        print("1:", steps)
    else:
        if stop is None:
            stop = start
        steps = np.linspace(start, stop, num)
        print("2:", steps)
    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [motor_x.name],
        "XEng": XEng.position,
        "plan_name": "eng_scan_delay",
        "plan_args": {
            "start": start,
            "stop": stop,
            "num": num,
            "detectors": "detectors",
            "delay_time": delay_time,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        "zone_plate": ZONE_PLATE,
        #'motor_pos':  wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(motor_x.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + [motor_x])
    @run_decorator(md=_md)
    def eng_inner_scan():
        for step in steps:
            yield from mv(motor_x, step)
            yield from bps.sleep(delay_time)
            yield from trigger_and_read(list(detectors) + [motor_x])
        yield from mv(motor_x, motor_x_ini)

    yield from eng_inner_scan()
    h = db[-1]
    scan_id = h.start["scan_id"]
    det = [det.name for det in detectors]
    det_name = ""
    for i in range(len(det)):
        det_name += det[i]
        det_name += ", "
    det_name = "[" + det_name[:-2] + "]"
    txt1 = get_scan_parameter()
    txt2 = f"detectors = {det_name}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)
    print(txt)


# ### Backup before modification 09/23/2019
#
# def fly_scan(exposure_time=0.1, relative_rot_angle = 180, period=0.15, chunk_size=20, out_x=None, out_y=2000, out_z=None,  out_r=None, rs=1, note='', simu=False, relative_move_flag=1, traditional_sequence_flag=1, filters=[], md=None):
#    '''
#    Inputs:
#    -------
#    exposure_time: float, in unit of sec
#
#    relative_rot_angle: float,
#        total rotation angles start from current rotary stage (zps.pi_r) position
#
#    period: float, in unit of sec
#        period of taking images, "period" should >= "exposure_time"
#
#    chunk_size: int, default setting is 20
#        number of images taken for each trigger of Andor camera
#
#    out_x: float, default is 0
#        relative movement of sample in "x" direction using zps.sx to move out sample (in unit of um)
#        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
#
#    out_y: float, default is 0
#        relative movement of sample in "y" direction using zps.sy to move out sample (in unit of um)
#        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
#
#    out_z: float, default is 0
#        relative movement of sample in "z" direction using zps.sz to move out sample (in unit of um)
#        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
#
#    out_r: float, default is 0
#        relative movement of sample by rotating "out_r" degrees, using zps.pi_r to move out sample
#        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
#
#    rs: float, default is 1
#        rotation speed in unit of deg/sec
#
#    note: string
#        adding note to the scan
#
#    simu: Bool, default is False
#        True: will simulate closing/open shutter without really closing/opening
#        False: will really close/open shutter
#
#    '''
#    global ZONE_PLATE
#    motor_x_ini = zps.sx.position
#    motor_y_ini = zps.sy.position
#    motor_z_ini = zps.sz.position
#    motor_r_ini = zps.pi_r.position
#
#    if relative_move_flag:
#        motor_x_out = motor_x_ini + out_x if out_x else motor_x_ini
#        motor_y_out = motor_y_ini + out_y if out_y else motor_y_ini
#        motor_z_out = motor_z_ini + out_z if out_z else motor_z_ini
#        motor_r_out = motor_r_ini + out_r if out_r else motor_r_ini
#    else:
#        motor_x_out = out_x if out_x else motor_x_ini
#        motor_y_out = out_y if out_y else motor_y_ini
#        motor_z_out = out_z if out_z else motor_z_ini
#        motor_r_out = out_r if out_r else motor_r_ini
#
#    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]
#
#    detectors = [Andor, ic3]
#    offset_angle = -0.5 * rs
#    current_rot_angle = zps.pi_r.position
#
#    target_rot_angle = current_rot_angle + relative_rot_angle
#    _md = {'detectors': ['Andor'],
#           'motors': [mot.name for mot in motor],
#           'XEng': XEng.position,
#           'ion_chamber': ic3.name,
#           'plan_args': {'exposure_time': exposure_time,
#                         'relative_rot_angle': relative_rot_angle,
#                         'period': period,
#                         'chunk_size': chunk_size,
#                         'out_x': out_x,
#                         'out_y': out_y,
#                         'out_z': out_z,
#                         'out_r': out_r,
#                         'rs': rs,
#                         'relative_move_flag': relative_move_flag,
#                         'traditional_sequence_flag': traditional_sequence_flag,
#                         'filters': [filt.name for filt in filters] if filters else 'None',
#                         'note': note if note else 'None',
#                         'zone_plate': ZONE_PLATE,
#                        },
#           'plan_name': 'fly_scan',
#           'num_bkg_images': chunk_size,
#           'num_dark_images': chunk_size,
#           'chunk_size': chunk_size,
#           'plan_pattern': 'linspace',
#           'plan_pattern_module': 'numpy',
#           'hints': {},
#           'operator': 'FXI',
#           'note': note if note else 'None',
#           'zone_plate': ZONE_PLATE,
#           #'motor_pos': wh_pos(print_on_screen=0),
#            }
#    _md.update(md or {})
#    try:  dimensions = [(zps.pi_r.hints['fields'], 'primary')]
#    except (AttributeError, KeyError):    pass
#    else: _md['hints'].setdefault('dimensions', dimensions)
#
#    yield from _set_andor_param(exposure_time=exposure_time, period=period, chunk_size=chunk_size)
#    yield from _set_rotation_speed(rs=rs)
#    print('set rotation speed: {} deg/sec'.format(rs))
#
#
#    @stage_decorator(list(detectors) + motor)
#    @bpp.monitor_during_decorator([zps.pi_r])
#    @run_decorator(md=_md)
#    def fly_inner_scan():
#        #close shutter, dark images: numer=chunk_size (e.g.20)
#        print('\nshutter closed, taking dark images...')
#        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)
#
#        #open shutter, tomo_images
#        yield from _open_shutter(simu=simu)
#        print ('\nshutter opened, taking tomo images...')
#        yield from mv(zps.pi_r, current_rot_angle + offset_angle)
#        status = yield from abs_set(zps.pi_r, target_rot_angle, wait=False)
#        yield from bps.sleep(1)
#        while not status.done:
#            yield from trigger_and_read(list(detectors) + motor)
#        # bkg images
#        print ('\nTaking background images...')
#        yield from _set_rotation_speed(rs=30)
#        for flt in filters:
#            yield from mv(flt, 1)
#            yield from mv(flt, 1)
#        yield from bps.sleep(1)
#        yield from _take_bkg_image(motor_x_out, motor_y_out, motor_z_out, motor_r_out, detectors, motor, num_bkg=1, simu=False, traditional_sequence_flag=traditional_sequence_flag)
#        yield from _close_shutter(simu=simu)
#        yield from _move_sample_in(motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini, trans_first_flag=traditional_sequence_flag)
#        for flt in filters:
#            yield from mv(flt, 0)
#    uid = yield from fly_inner_scan()
#    yield from mv(Andor.cam.image_mode, 1)
#    print('scan finished')
#    txt = get_scan_parameter(print_flag=0)
#    insert_text(txt)
#    print(txt)
#    return uid


def fly_scan(
    exposure_time=0.1,
    start_angle = None,
    relative_rot_angle=180,
    period=0.15,
    chunk_size=20,
    out_x=None,
    out_y=2000,
    out_z=None,
    out_r=None,
    rs=1,
    note="",
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    filters=[],
    rot_back_velo=30,
    md=None,
    binning=[1, 1]
):
    """
    Inputs:
    -------
    exposure_time: float, in unit of sec

    start_angle: float
        starting angle 

    relative_rot_angle: float, 
        total rotation angles start from current rotary stage (zps.pi_r) position

    period: float, in unit of sec
        period of taking images, "period" should >= "exposure_time"

    chunk_size: int, default setting is 20
        number of images taken for each trigger of Andor camera

    out_x: float, default is 0
        relative movement of sample in "x" direction using zps.sx to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_y: float, default is 0
        relative movement of sample in "y" direction using zps.sy to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_z: float, default is 0
        relative movement of sample in "z" direction using zps.sz to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_r: float, default is 0
        relative movement of sample by rotating "out_r" degrees, using zps.pi_r to move out sample
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    rs: float, default is 1
        rotation speed in unit of deg/sec

    note: string
        adding note to the scan

    simu: Bool, default is False
        True: will simulate closing/open shutter without really closing/opening
        False: will really close/open shutter
    
    """
    global ZONE_PLATE

    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if not (start_angle is None):
        yield from mv(zps.pi_r, start_angle)

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if not (out_x is None) else motor_x_ini
        motor_y_out = motor_y_ini + out_y if not (out_y is None) else motor_y_ini
        motor_z_out = motor_z_ini + out_z if not (out_z is None) else motor_z_ini
        motor_r_out = motor_r_ini + out_r if not (out_r is None) else motor_r_ini
    else:
        motor_x_out = out_x if not (out_x is None) else motor_x_ini
        motor_y_out = out_y if not (out_y is None) else motor_y_ini
        motor_z_out = out_z if not (out_z is None) else motor_z_ini
        motor_r_out = out_r if not (out_r is None) else motor_r_ini

    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    detectors = [Andor, ic3]
    offset_angle = -2 * rs
    current_rot_angle = zps.pi_r.position

    target_rot_angle = current_rot_angle + relative_rot_angle
    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motor],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exposure_time,
            "start_angle": start_angle,
            "relative_rot_angle": relative_rot_angle,
            "period": period,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "rs": rs,
            "relative_move_flag": relative_move_flag,
            "rot_first_flag": rot_first_flag,
            "filters": [filt.name for filt in filters] if filters else "None",
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "fly_scan",
        "num_bkg_images": 20,
        "num_dark_images": 20,
        "chunk_size": chunk_size,
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        "zone_plate": ZONE_PLATE,
        #'motor_pos': wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(zps.pi_r.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.bin_y, binning[0],
                  Andor.cam.bin_x, binning[1])
    yield from _set_andor_param(
        exposure_time=exposure_time, period=period, chunk_size=chunk_size
    )
    yield from _set_rotation_speed(rs=rs)
    print("set rotation speed: {} deg/sec".format(rs))

    @stage_decorator(list(detectors) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        # close shutter, dark images: numer=chunk_size (e.g.20)
        print("\nshutter closed, taking dark images...")
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

        # open shutter, tomo_images
        yield from _open_shutter(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from mv(zps.pi_r, current_rot_angle + offset_angle)
        status = yield from abs_set(zps.pi_r, target_rot_angle, wait=False)
        yield from bps.sleep(2)
        while not status.done:
            yield from trigger_and_read(list(detectors) + motor)
        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=rot_back_velo)
        #        yield from abs_set(zps.pi_r.velocity, rs)
        for flt in filters:
            yield from mv(flt, 1)
            yield from mv(flt, 1)
        yield from bps.sleep(1)
        yield from _take_bkg_image(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            detectors,
            motor,
            num_bkg=1,
            simu=False,
            traditional_sequence_flag=rot_first_flag,
        )
        yield from _close_shutter(simu=simu)
        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            trans_first_flag=rot_first_flag,
        )
        for flt in filters:
            yield from mv(flt, 0)

    uid = yield from fly_inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    print("scan finished")
    txt = get_scan_parameter(print_flag=0)
    insert_text(txt)
    print(txt)
    return uid


def grid2D_rel(
    motor1,
    start1,
    stop1,
    num1,
    motor2,
    start2,
    stop2,
    num2,
    exposure_time=0.05,
    delay_time=0,
    note="",
    md=None,
):
    # detectors=[ic3, ic4]
    global ZONE_PLATE
    detectors = [Andor, ic3]
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.image_mode, 0)
    yield from mv(Andor.cam.num_images, 1)
    yield from mv(detectors[0].cam.acquire_time, exposure_time)
    yield from mv(Andor.cam.acquire_period, exposure_time)

    motor1_ini = motor1.position
    motor2_ini = motor2.position

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [motor1.name, motor2.name],
        "XEng": XEng.position,
        "plan_name": "grid2D_rel",
        "plan_args": {
            "motor1": motor1.name,
            "start1": start1,
            "stop1": stop1,
            "num1": num1,
            "motor2": motor2.name,
            "start2": start2,
            "stop2": stop2,
            "num2": num2,
            "exposure_time": exposure_time,
            "delay_time": delay_time,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        "zone_plate": ZONE_PLATE,
        #'motor_pos':  wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(motor1.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    motor1_s = motor1_ini + start1
    motor1_e = motor1_ini + stop1
    steps1 = np.linspace(motor1_s, motor1_e, num1)

    motor2_s = motor2_ini + start2
    motor2_e = motor2_ini + stop2
    steps2 = np.linspace(motor2_s, motor2_e, num2)
    print(steps1)
    print(steps2)

    @stage_decorator(list(detectors) + [motor1, motor2])
    @run_decorator(md=_md)
    def grid2D_rel_inner():
        for i in range(num1):
            yield from mv(motor1, steps1[i])
            for j in range(num2):
                yield from mv(motor2, steps2[j])
                yield from bps.sleep(delay_time)
                yield from trigger_and_read(list(detectors) + [motor1, motor2])
        yield from mv(motor1, motor1_ini, motor2, motor2_ini)

    yield from grid2D_rel_inner()

    h = db[-1]
    scan_id = h.start["scan_id"]
    det = [det.name for det in detectors]
    det_name = ""
    for i in range(len(det)):
        det_name += det[i]
        det_name += ", "
    det_name = "[" + det_name[:-2] + "]"
    txt1 = get_scan_parameter()
    txt2 = f"detectors = {det_name}"
    txt = txt1 + "\n" + txt2 + "\n"
    insert_text(txt)
    print(txt)


def delay_count(detectors, num=1, delay=None, *, note="", plot_flag=0, md=None):
    """
    same function as the default "count", 
    re_write it in order to add auto-logging
    """
    global ZONE_PLATE
    if num is None:
        num_intervals = None
    else:
        num_intervals = num - 1
    _md = {
        "detectors": [det.name for det in detectors],
        "num_points": num,
        "XEng": XEng.position,
        "num_intervals": num_intervals,
        "plan_args": {
            "detectors": "detectors",
            "num": num,
            "delay": delay,
            "note": note,
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "delay_count",
        "hints": {},
        "note": note if note else "None",
        "zone_plate": ZONE_PLATE,
    }
    _md.update(md or {})
    _md["hints"].setdefault("dimensions", [(("time",), "primary")])

    @bpp.stage_decorator(detectors)
    @bpp.run_decorator(md=_md)
    def inner_count():
        return (
            yield from bps.repeat(
                partial(bps.trigger_and_read, detectors), num=num, delay=delay
            )
        )

    uid = yield from inner_count()
    h = db[-1]
    scan_id = h.start["scan_id"]
    if plot_flag:
        plot1d(scan_id)
    det = [det.name for det in detectors]
    det_name = ""
    for i in range(len(det)):
        det_name += det[i]
        det_name += ", "
    det_name = "[" + det_name[:-2] + "]"

    txt1 = get_scan_parameter()
    txt2 = f"detectors = {det_name}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)
    print(txt)
    return uid


def delay_scan(
    detectors,
    motor,
    start,
    stop,
    steps,
    exposure_time=0.1,
    sleep_time=1.0,
    plot_flag=0,
    note="",
    md=None,
    simu=False,
):
    """
    add sleep_time to regular 'scan' for each scan_step

    Inputs:
    ---------
    detectors: list of dectectors, e.g., [Andor, ic3]

    motor: list of motors, e.g., zps.sx

    start: float, motor start position

    stop: float, motor stop position

    steps: int, number of steps for motor motion

    exposure_time: float, in unit of sec

    sleep time: float, in unit of sec

    plot_flag: 0 or 1
        if 1: will plot detector signal vs. motor
        if 0: not plot

    note: string
    
    """
    global ZONE_PLATE
    if Andor in detectors:
        yield from _set_andor_param(exposure_time, period=exposure_time, chunk_size=1)

    # motor = dcm.th2
    # motor = pzt_dcm_th2.setpos
    motor_ini = motor.position
    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [motor.name],
        "XEng": XEng.position,
        "plan_args": {
            "detectors": "detectors",
            "motor": motor.name,
            "start": start,
            "stop": stop,
            "steps": steps,
            "exposure_time": exposure_time,
            "sleep_time": sleep_time,
            "plot_flag": plot_flag,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "delay_scan",
        "zone_plate": ZONE_PLATE,
        "hints": {},
        #'motor_pos':  wh_pos(print_on_screen=0),
        "operator": "FXI",
    }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    my_var = np.linspace(start, stop, steps)

    @stage_decorator(list(detectors) + [motor])
    @run_decorator(md=_md)
    def delay_inner_scan():
        for x in my_var:
            yield from mv_stage(motor, x)
            yield from bps.sleep(sleep_time)
            yield from trigger_and_read(list(detectors + [motor]))
        yield from mv(motor, motor_ini)

    if simu:
        uid = yield from delay_inner_scan()
    else:
        yield from abs_set(shutter_open, 1, wait=True)
        yield from bps.sleep(1)
        yield from abs_set(shutter_open, 1)
        yield from bps.sleep(1)
        uid = yield from delay_inner_scan()
        yield from abs_set(shutter_close, 1, wait=True)
        yield from bps.sleep(1)
        yield from abs_set(shutter_close, 1)
        yield from bps.sleep(1)
    h = db[-1]
    scan_id = h.start["scan_id"]
    if plot_flag:
        plot1d(scan_id)

    det = [det.name for det in detectors]
    det_name = ""
    for i in range(len(det)):
        det_name += det[i]
        det_name += ", "
    det_name = "[" + det_name[:-2] + "]"

    txt1 = get_scan_parameter()
    txt2 = f"detectors = {det_name}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)
    print(txt)
    return uid


"""
def xanes_3d_scan(eng_list, exposure_time, relative_rot_angle, period, chunk_size=20, out_x=0, out_y=0, rs=3, parkpos=None, note=''):

    id_list=[]
    txt1 = f'xanes_3d_scan(eng_list=eng_list, exposure_time={exposure_time}, relative_rot_angle={relative_rot_angle}, period={period}, chunk_size={chunk_size}, out_x={out_x}, out_y={out_y}, rs={rs}, parkpos={park_pos}, note={note if note else "None"})'
    txt2 = f'eng_list = {eng_list}'
    txt = tx11 + txt2
    insert_text(txt)
    print(txt)

   
    for eng in eng_list:
        RE(move_zp_ccd(eng))
        RE(fly_scan(exposure_time, relative_rot_angle, period, chunk_size, out_x, out_y, rs, parkpos, note))
        scan_id=db[-1].start['scan_id']
        id_list.append(int(scan_id))
        print('current energy: {} --> scan_id: {}\n'.format(eng, scan_id))
    return my_eng_list, id_list

"""


def raster_2D_scan(
    x_range=[-1, 1],
    y_range=[-1, 1],
    exposure_time=0.1,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    img_sizeX=2560,
    img_sizeY=2160,
    pxl=17.2,
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    note="",
    scan_x_flag=1,
    filters=[],
    md=None,
):
    """
    scanning large area by moving samples at different 2D block position, defined by x_range and y_range, only work for Andor camera at full resolution (2160 x 2560)
    for example, set x_range=[-1,1] and y_range=[-2, 2] will totally take 3 x 5 = 15 images and stitch them together

    Inputs:
    -------

    x_range: two-elements list, e.g., [-1, 1], in unit of horizontal screen size 

    y_range: two-elements list, e.g., [-1, 1], in unit of horizontal screen size

    exposure_time: float

    out_x: float, default is 0
        relative movement of sample in "x" direction using zps.sx to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_y: float, default is 0
        relative movement of sample in "y" direction using zps.sy to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_z: float, default is 0
        relative movement of sample in "z" direction using zps.sz to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_r: float, default is 0
        relative movement of sample by rotating "out_r" degrees, using zps.pi_r to move out sample
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
       
    img_sizeX: int, default is 2560, it is the pixel number for Andor camera horizontal

    img_sizeY: int, default is 2160, it is the pixel number for Andor camera vertical

    pxl: float, pixel size, default is 17.2, in unit of nm/pix

    note: string

    scan_x_flag: 1 or 0
        if 1: scan x and y
        if 0: scan z and y

    simu: Bool, default is False
        True: will simulate closing/open shutter without really closing/opening
        False: will really close/open shutter
              
    """
    global ZONE_PLATE
    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]
    detectors = [Andor, ic3]
    yield from _set_andor_param(
        exposure_time=exposure_time, period=exposure_time, chunk_size=1
    )

    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if not (out_x is None) else motor_x_ini
        motor_y_out = motor_y_ini + out_y if not (out_y is None) else motor_y_ini
        motor_z_out = motor_z_ini + out_z if not (out_z is None) else motor_z_ini
        motor_r_out = motor_r_ini + out_r if not (out_r is None) else motor_r_ini
    else:
        motor_x_out = out_x if not (out_x is None) else motor_x_ini
        motor_y_out = out_y if not (out_y is None) else motor_y_ini
        motor_z_out = out_z if not (out_z is None) else motor_z_ini
        motor_r_out = out_r if not (out_r is None) else motor_r_ini

    img_sizeX = np.int(img_sizeX)
    img_sizeY = np.int(img_sizeY)
    x_range = np.int_(x_range)
    y_range = np.int_(y_range)

    print("hello1")
    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motor],
        "num_bkg_images": 5,
        "num_dark_images": 5,
        "x_range": x_range,
        "y_range": y_range,
        "out_x": out_x,
        "out_y": out_y,
        "out_z": out_z,
        "exposure_time": exposure_time,
        "XEng": XEng.position,
        "plan_args": {
            "x_range": x_range,
            "y_range": y_range,
            "exposure_time": exposure_time,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "img_sizeX": img_sizeX,
            "img_sizeY": img_sizeY,
            "pxl": pxl,
            "note": note if note else "None",
            "relative_move_flag": relative_move_flag,
            "rot_first_flag": rot_first_flag,
            "note": note if note else "None",
            "scan_x_flag": scan_x_flag,
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "raster_2D",
        "hints": {},
        "operator": "FXI",
        "zone_plate": ZONE_PLATE,
        "note": note if note else "None",
        #'motor_pos':  wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def raster_2D_inner():
        # take dark image
        print("take 5 dark image")
        yield from _take_dark_image(detectors, motor, num_dark=5, simu=simu)

        print("open shutter ...")
        yield from _open_shutter(simu)

        print("taking mosaic image ...")
        for ii in np.arange(x_range[0], x_range[1] + 1):
            if scan_x_flag == 1:
                yield from mv(zps.sx, motor_x_ini + ii * img_sizeX * pxl * 1.0 / 1000)
                yield from mv(zps.sx, motor_x_ini + ii * img_sizeX * pxl * 1.0 / 1000)
            else:
                yield from mv(zps.sz, motor_z_ini + ii * img_sizeX * pxl * 1.0 / 1000)
                yield from mv(zps.sz, motor_z_ini + ii * img_sizeX * pxl * 1.0 / 1000)
            sleep_time = (x_range[-1] - x_range[0]) * img_sizeX * pxl * 1.0 / 1000 / 600
            yield from bps.sleep(sleep_time)
            for jj in np.arange(y_range[0], y_range[1] + 1):
                yield from mv(zps.sy, motor_y_ini + jj * img_sizeY * pxl * 1.0 / 1000)
                yield from _take_image(detectors, motor, 1)
        #                yield from trigger_and_read(list(detectors) + motor)

        print("moving sample out to take 5 background image")
        if len(filters):
            for filt in filters:
                yield from mv(filt, 1)
                yield from bps.sleep(0.5)
        yield from _take_bkg_image(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            detectors,
            motor,
            num_bkg=5,
            simu=simu,
            traditional_sequence_flag=rot_first_flag,
        )

        # move sample in
        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            repeat=1,
            trans_first_flag=1 - rot_first_flag,
        )
        if len(filters):
            for filt in filters:
                yield from mv(filt, 0)
                yield from bps.sleep(0.5)
        print("closing shutter")
        yield from _close_shutter(simu)

    yield from raster_2D_inner()
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)


def raster_2D_scan2(
    x_range=[-1, 1],
    y_range=[-1, 1],
    exposure_time=0.1,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    img_sizeX=2560,
    img_sizeY=2160,
    pxl=17.2,
    num_bkg=1,
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    note="",
    scan_x_flag=1,
    md=None,
):
    """
    scanning large area by moving samples at different 2D block position, defined by x_range and y_range, only work for Andor camera at full resolution (2160 x 2560)
    for example, set x_range=[-1,1] and y_range=[-2, 2] will totally take 3 x 5 = 15 images and stitch them together

    Different from raster_2D_scan that this scan will take backgound image for every movement

    Inputs:
    -------

    x_range: two-elements list, e.g., [-1, 1], in unit of horizontal screen size 

    y_range: two-elements list, e.g., [-1, 1], in unit of horizontal screen size

    exposure_time: float

    out_x: float, default is 0
        relative movement of sample in "x" direction using zps.sx to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_y: float, default is 0
        relative movement of sample in "y" direction using zps.sy to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_z: float, default is 0
        relative movement of sample in "z" direction using zps.sz to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_r: float, default is 0
        relative movement of sample by rotating "out_r" degrees, using zps.pi_r to move out sample
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
       
    img_sizeX: int, default is 2560, it is the pixel number for Andor camera horizontal

    img_sizeY: int, default is 2160, it is the pixel number for Andor camera vertical

    pxl: float, pixel size, default is 17.2, in unit of nm/pix

    note: string

    scan_x_flag: 1 or 0
        if 1: scan x and y
        if 0: scan z and y

    simu: Bool, default is False
        True: will simulate closing/open shutter without really closing/opening
        False: will really close/open shutter
              
    """
    global ZONE_PLATE
    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]
    detectors = [Andor, ic3]
    yield from _set_andor_param(
        exposure_time=exposure_time, period=exposure_time, chunk_size=1
    )

    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if not (out_x is None) else motor_x_ini
        motor_y_out = motor_y_ini + out_y if not (out_y is None) else motor_y_ini
        motor_z_out = motor_z_ini + out_z if not (out_z is None) else motor_z_ini
        motor_r_out = motor_r_ini + out_r if not (out_r is None) else motor_r_ini
    else:
        motor_x_out = out_x if not (out_x is None) else motor_x_ini
        motor_y_out = out_y if not (out_y is None) else motor_y_ini
        motor_z_out = out_z if not (out_z is None) else motor_z_ini
        motor_r_out = out_r if not (out_r is None) else motor_r_ini

    img_sizeX = np.int(img_sizeX)
    img_sizeY = np.int(img_sizeY)
    x_range = np.int(x_range)
    y_range = np.int(y_range)

    print("hello1")
    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motor],
        "num_bkg_images": 5,
        "num_dark_images": 5,
        "x_range": x_range,
        "y_range": y_range,
        "out_x": out_x,
        "out_y": out_y,
        "out_z": out_z,
        "exposure_time": exposure_time,
        "XEng": XEng.position,
        "plan_args": {
            "x_range": x_range,
            "y_range": y_range,
            "exposure_time": exposure_time,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "img_sizeX": img_sizeX,
            "img_sizeY": img_sizeY,
            "pxl": pxl,
            "num_bkg": num_bkg,
            "note": note if note else "None",
            "relative_move_flag": relative_move_flag,
            "rot_first_flag": rot_first_flag,
            "note": note if note else "None",
            "scan_x_flag": scan_x_flag,
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "raster_2D_2",
        "hints": {},
        "operator": "FXI",
        "zone_plate": ZONE_PLATE,
        "note": note if note else "None",
        #'motor_pos':  wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def raster_2D_inner():
        # take dark image
        print("take 5 dark image")
        yield from _take_dark_image(detectors, motor, num_dark=5, simu=simu)

        print("open shutter ...")
        yield from _open_shutter(simu)

        print("taking mosaic image ...")
        for ii in np.arange(x_range[0], x_range[1] + 1):

            for jj in np.arange(y_range[0], y_range[1] + 1):
                if scan_x_flag == 1:
                    yield from mv(
                        zps.sx, motor_x_ini + ii * img_sizeX * pxl * 1.0 / 1000
                    )
                    yield from mv(
                        zps.sx, motor_x_ini + ii * img_sizeX * pxl * 1.0 / 1000
                    )
                else:
                    yield from mv(
                        zps.sz, motor_z_ini + ii * img_sizeX * pxl * 1.0 / 1000
                    )
                    yield from mv(
                        zps.sz, motor_z_ini + ii * img_sizeX * pxl * 1.0 / 1000
                    )
                yield from mv(zps.sy, motor_y_ini + jj * img_sizeY * pxl * 1.0 / 1000)
                yield from _take_image(detectors, motor, 1)
                #                yield from trigger_and_read(list(detectors) + motor)
                yield from _take_bkg_image(
                    motor_x_out,
                    motor_y_out,
                    motor_z_out,
                    motor_r_out,
                    detectors,
                    motor,
                    num_bkg=num_bkg,
                    simu=simu,
                    traditional_sequence_flag=rot_first_flag,
                )

        # print('moving sample out to take 5 background image')
        # yield from _take_bkg_image(motor_x_out, motor_y_out, motor_z_out, motor_r_out, detectors, motor, num_bkg=5, simu=simu,traditional_sequence_flag=rot_first_flag)

        # move sample in
        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            repeat=1,
            trans_first_flag=1 - rot_first_flag,
        )

        print("closing shutter")
        yield from _close_shutter(simu)

    yield from raster_2D_inner()
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)


def multipos_2D_xanes_scan(
    eng_list,
    x_list,
    y_list,
    z_list,
    r_list,
    out_x,
    out_y,
    out_z,
    out_r,
    chunk_size=5,
    exposure_time=0.1,
    repeat_num=1,
    sleep_time=0,
    relative_move_flag=1,
    note="",
):
    num = len(x_list)
    txt = f"Multipos_2D_xanes_scan(eng_list, x_list, y_list, z_list, out_x={out_x}, out_y={out_y}, out_z={out_z}, out_r={out_r}, exposure_time={exposure_time}, note={note})"
    insert_text(txt)
    txt = "Take 2D_xanes at multiple position, containing following scans:"
    insert_text(txt)
    for rep in range(repeat_num):
        print(f"round: {rep}")
        for i in range(num):
            print(
                f"current position[{i}]: x={x_list[i]}, y={y_list[i]}, z={z_list[i]}\n"
            )
            my_note = (
                note + f"_position_{i}: x={x_list[i]}, y={y_list[i]}, z={z_list[i]}"
            )
            yield from mv(
                zps.sx,
                x_list[i],
                zps.sy,
                y_list[i],
                zps.sz,
                z_list[i],
                zps.pi_r,
                r_list[i],
            )
            yield from xanes_scan2(
                eng_list,
                exposure_time=exposure_time,
                chunk_size=chunk_size,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                out_r=out_r,
                relative_move_flag=relative_move_flag,
                note=my_note,
            )
        yield from bps.sleep(sleep_time)
    yield from mv(
        zps.sx, x_list[0], zps.sy, y_list[0], zps.sz, z_list[0], zps.pi_r, r_list[0]
    )
    insert_text("Finished the multipos_2D_xanes_scan")


def multipos_2D_xanes_scan2(
    eng_list,
    x_list,
    y_list,
    z_list,
    r_list,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    repeat_num=1,
    exposure_time=0.2,
    sleep_time=1,
    chunk_size=5,
    simu=False,
    relative_move_flag=True,
    note="",
    md=None,
    binning=[1, 1]
):
    """
    Different from multipos_2D_xanes_scan. In the current scan, it take image at all locations and then move out sample to take background image.

    For example:
    RE(multipos_2D_xanes_scan2(Ni_eng_list, x_list=[0,1,2], y_list=[2,3,4], z_list=[0,0,0], r_list=[0,0,0], out_x=1000, out_y=0, out_z=0, out_r=90, repeat_num=2, exposure_time=0.1, sleep_time=60, chunk_size=5, relative_move_flag=True, note='sample')
    
    Inputs:
    --------
    eng_list: list or numpy array,
           energy in unit of keV

    x_list: list or numpy array,
            x_position, in unit of um

    y_list: list or numpy array,
            y_position, in unit of um

    z_list: list or numpy array,
            z_position, in unit of um

    r_list: list or numpy array,
            rotation_angle, in unit of degree

    out_x: float, default is 0
        relative movement of sample in "x" direction using zps.sx to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_y: float, default is 0
        relative movement of sample in "y" direction using zps.sy to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_z: float, default is 0
        relative movement of sample in "z" direction using zps.sz to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_r: float, default is 0
        relative movement of sample by rotating "out_r" degrees, using zps.pi_r to move out sample
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
    
    repeat_num: integer, default is 1
        repeating multiposition xanes scans

    exposure_time: float
           in unit of seconds

    sleep_time: float(int)
           in unit of seconds

    chunk_size: int
           number of background images == num of dark images ==  num of image for each energy
   
    relative_move_flag:
          if 1: relative movement of out_x, out_y, out_z, and out_r
          if 0: set absolute position of x, y, z, r to move out sample

    note: string
    
    """
    print(eng_list)
    print(x_list)
    print(y_list)
    print(z_list)
    print(r_list)
    print(out_x)
    print(out_y)
    print(out_z)
    print(out_r)
    global ZONE_PLATE
    txt = "starting multipos_2D_xanes_scan2:"
    insert_text(txt)
    detectors = [Andor, ic3, ic4]
    period = max(0.05, exposure_time)
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.bin_y, binning[0],
                  Andor.cam.bin_x, binning[1])
    yield from _set_andor_param(exposure_time, period=period, chunk_size=chunk_size)

    eng_ini = XEng.position

    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if not (out_x is None) else motor_x_ini
        motor_y_out = motor_y_ini + out_y if not (out_y is None) else motor_y_ini
        motor_z_out = motor_z_ini + out_z if not (out_z is None) else motor_z_ini
        motor_r_out = motor_r_ini + out_r if not (out_r is None) else motor_r_ini
    else:
        motor_x_out = out_x if not (out_x is None) else motor_x_ini
        motor_y_out = out_y if not (out_y is None) else motor_y_ini
        motor_z_out = out_z if not (out_z is None) else motor_z_ini
        motor_r_out = out_r if not (out_r is None) else motor_r_ini

    motor = [XEng, zps.sx, zps.sy, zps.sz, zps.pi_r]

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motor],
        "num_eng": len(eng_list),
        "num_bkg_images": chunk_size,
        "num_dark_images": chunk_size,
        "chunk_size": chunk_size,
        "out_x": out_x,
        "out_y": out_y,
        "exposure_time": exposure_time,
        "eng_list": eng_list,
        "num_pos": len(x_list),
        "XEng": XEng.position,
        "plan_args": {
            "eng_list": "eng_list",
            "x_list": x_list,
            "y_list": y_list,
            "z_list": z_list,
            "r_list": r_list,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "repeat_num": repeat_num,
            "exposure_time": exposure_time,
            "sleep_time": sleep_time,
            "chunk_size": chunk_size,
            "relative_move_flag": relative_move_flag,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "multipos_2D_xanes_scan2",
        "hints": {},
        "operator": "FXI",
        "zone_plate": ZONE_PLATE,
        "note": note if note else "None",
        #'motor_pos':  wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def inner_scan():
        # close shutter and take dark image
        num = len(x_list)  # num of position points
        print(f"\ntake {chunk_size} dark images...")
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=False)
        yield from bps.sleep(1)

        # start repeating xanes scan
        print(
            f"\nopening shutter, and start xanes scan: {chunk_size} images per each energy... "
        )

        yield from _open_shutter(simu)
        for rep in range(repeat_num):
            print(f"repeat multi-pos xanes scan #{rep}")
            for eng in eng_list:
                yield from move_zp_ccd(eng, move_flag=1, info_flag=0)
                yield from _open_shutter(simu)
                for i in range(num):
                    # take image at multiple positions
                    yield from mv(
                        zps.sx,
                        x_list[i],
                        zps.sy,
                        y_list[i],
                        zps.sz,
                        z_list[i],
                        zps.pi_r,
                        r_list[i],
                    )
                    yield from mv(
                        zps.sx,
                        x_list[i],
                        zps.sy,
                        y_list[i],
                        zps.sz,
                        z_list[i],
                        zps.pi_r,
                        r_list[i],
                    )
                    yield from trigger_and_read(list(detectors) + motor)
                # move sample out to take background
                yield from _take_bkg_image(
                    motor_x_out,
                    motor_y_out,
                    motor_z_out,
                    motor_r_out,
                    detectors,
                    motor,
                    num_bkg=1,
                    simu=simu,
                )
                # move sample in to the first position
                yield from _move_sample_in(
                    motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini
                )
            # end of eng_list
            # close shutter and sleep
            yield from _close_shutter(simu)
            # sleep
            if rep < repeat_num: 
                print(f"\nsleep for {sleep_time} seconds ...")
                yield from bps.sleep(sleep_time)

        yield from mv(
            zps.sx, x_list[0], zps.sy, y_list[0], zps.sz, z_list[0], zps.pi_r, r_list[0]
        )

    yield from inner_scan()
    txt1 = get_scan_parameter()
    eng_list = np.round(eng_list, 5)
    if len(eng_list) > 10:
        txt2 = f"eng_list: {eng_list[0:10]}, ... {eng_list[-5:]}\n"
    else:
        txt2 = f"eng_list: {eng_list}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)


def multipos_2D_xanes_scan3(
    eng_list,
    x_list,
    y_list,
    z_list,
    r_list,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    repeat_num=1,
    exposure_time=0.2,
    sleep_time=1,
    chunk_size=5,
    simu=False,
    relative_move_flag=1,
    note="",
    md=None,
):
    """
    Different from multipos_2D_xanes_scan2. In the current scan, it take image at all locations at all energies and then move out sample to take background image at all energies again.

    For example:
    RE(multipos_2D_xanes_scan3(Ni_eng_list, x_list=[0,1,2], y_list=[2,3,4], z_list=[0,0,0], r_list=[0,0,0], out_x=1000, out_y=0, out_z=0, out_r=90, repeat_num=2, sleep_time=60, note='sample')
    
    Inputs:
    --------
    eng_list: list or numpy array,
           energy in unit of keV

    x_list: list or numpy array,
            x_position, in unit of um

    y_list: list or numpy array,
            y_position, in unit of um

    z_list: list or numpy array,
            z_position, in unit of um

    r_list: list or numpy array,
            rotation_angle, in unit of degree

    out_x: float, default is 0
        relative movement of sample in "x" direction using zps.sx to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_y: float, default is 0
        relative movement of sample in "y" direction using zps.sy to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_z: float, default is 0
        relative movement of sample in "z" direction using zps.sz to move out sample (in unit of um)
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    out_r: float, default is 0
        relative movement of sample by rotating "out_r" degrees, using zps.pi_r to move out sample
        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z

    exposure_time: float
           in unit of seconds

    sleep_time: float(int)
           in unit of seconds

    chunk_size: int
           number of background images == num of dark images ==  num of image for each energy

    note: string
    
    """
    global ZONE_PLATE
    txt = "starting multipos_2D_xanes_scan3"
    insert_text(txt)
    detectors = [Andor, ic3]
    period = max(0.05, exposure_time)
    yield from _set_andor_param(exposure_time, period=period, chunk_size=chunk_size)
    eng_ini = XEng.position

    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if not (out_x is None) else motor_x_ini
        motor_y_out = motor_y_ini + out_y if not (out_y is None) else motor_y_ini
        motor_z_out = motor_z_ini + out_z if not (out_z is None) else motor_z_ini
        motor_r_out = motor_r_ini + out_r if not (out_r is None) else motor_r_ini
    else:
        motor_x_out = out_x if not (out_x is None) else motor_x_ini
        motor_y_out = out_y if not (out_y is None) else motor_y_ini
        motor_z_out = out_z if not (out_z is None) else motor_z_ini
        motor_r_out = out_r if not (out_r is None) else motor_r_ini

    motor = [XEng, zps.sx, zps.sy, zps.sz, zps.pi_r]

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motor],
        "num_eng": len(eng_list),
        "num_bkg_images": chunk_size,
        "num_dark_images": chunk_size,
        "chunk_size": chunk_size,
        "out_x": out_x,
        "out_y": out_y,
        "exposure_time": exposure_time,
        "eng_list": eng_list,
        "num_pos": len(x_list),
        "XEng": XEng.position,
        "plan_args": {
            "eng_list": "eng_list",
            "x_list": x_list,
            "y_list": y_list,
            "z_list": z_list,
            "r_list": r_list,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "repeat_num": repeat_num,
            "exposure_time": exposure_time,
            "sleep_time": sleep_time,
            "chunk_size": chunk_size,
            "relative_move_flag": relative_move_flag,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "multipos_2D_xanes_scan3",
        "hints": {},
        "operator": "FXI",
        "zone_plate": ZONE_PLATE,
        "note": note if note else "None",
        #'motor_pos':  wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def inner_scan():
        # close shutter and take dark image
        num = len(x_list)  # num of position points
        print(f"\ntake {chunk_size} dark images...")
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=False)

        # start repeating xanes scan
        print(
            f"\nopening shutter, and start xanes scan: {chunk_size} images per each energy... "
        )

        yield from _open_shutter(simu)
        for rep in range(repeat_num):
            print(f"repeat multi-pos xanes scan #{rep}")
            for eng in eng_list:
                yield from move_zp_ccd(eng, move_flag=1, info_flag=0)
                yield from _open_shutter(simu)
                for i in range(num):
                    # take image at multiple positions
                    yield from mv(
                        zps.sx,
                        x_list[i],
                        zps.sy,
                        y_list[i],
                        zps.sz,
                        z_list[i],
                        zps.pi_r,
                        r_list[i],
                    )
                    yield from mv(
                        zps.sx,
                        x_list[i],
                        zps.sy,
                        y_list[i],
                        zps.sz,
                        z_list[i],
                        zps.pi_r,
                        r_list[i],
                    )
                    yield from trigger_and_read(list(detectors) + motor)
            # move sample out to take background
            for eng in eng_list:
                yield from move_zp_ccd(eng, move_flag=1, info_flag=0)
                yield from _move_sample_out(
                    motor_x_out, motor_y_out, motor_z_out, motor_r_out
                )
                yield from trigger_and_read(list(detectors) + motor)

            yield from _move_sample_in(
                motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini, 2
            )
            yield from _close_shutter(simu)
            # sleep
            print(f"\nsleep for {sleep_time} seconds ...")
            yield from bps.sleep(sleep_time)
        yield from mv(
            zps.sx, x_list[0], zps.sy, y_list[0], zps.sz, z_list[0], zps.pi_r, r_list[0]
        )

    yield from inner_scan()
    txt1 = get_scan_parameter()
    eng_list = np.round(eng_list, 5)
    if len(eng_list) > 10:
        txt2 = f"eng_list: {eng_list[0:10]}, ... {eng_list[-5:]}\n"
    else:
        txt2 = f"eng_list: {eng_list}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)


def raster_2D_xanes2(
    eng_list,
    x_range=[-1, 1],
    y_range=[-1, 1],
    exposure_time=0.1,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    img_sizeX=2560,
    img_sizeY=2160,
    pxl=17.2,
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    note="",
    md=None,
):

    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if not (out_x is None) else motor_x_ini
        motor_y_out = motor_y_ini + out_y if not (out_y is None) else motor_y_ini
        motor_z_out = motor_z_ini + out_z if not (out_z is None) else motor_z_ini
        motor_r_out = motor_r_ini + out_r if not (out_r is None) else motor_r_ini
    else:
        motor_x_out = out_x if not (out_x is None) else motor_x_ini
        motor_y_out = out_y if not (out_y is None) else motor_y_ini
        motor_z_out = out_z if not (out_z is None) else motor_z_ini
        motor_r_out = out_r if not (out_r is None) else motor_r_ini

    x_list, y_list, z_list, r_list = [], [], [], []
    for ii in np.arange(x_range[0], x_range[1] + 1):
        for jj in np.arange(y_range[0], y_range[1] + 1):
            x = motor_x_ini + ii * img_sizeX * pxl * 1.0 / 1000
            y = motor_y_ini + jj * img_sizeY * pxl * 1.0 / 1000
            x_list.append(x)
            y_list.append(y)
            z_list.append(motor_z_ini)
            r_list.append(motor_r_ini)
    print(
        f"x_list = {x_list}\ny_list = {y_list}\nz_list = {z_list}\nr_list = {r_list}\n"
    )
    insert_text("raster_2D_xanes2 contains following scans:")
    yield from multipos_2D_xanes_scan2(
        eng_list,
        x_list,
        y_list,
        z_list,
        r_list,
        motor_x_out,
        motor_y_out,
        motor_z_out,
        motor_r_out,
        chunk_size=4,
        exposure_time=exposure_time,
        repeat_num=1,
        sleep_time=0,
        relative_move_flag=0,
        note=note,
    )
    insert_text("finished raster_2D_xanes2")


def raster_2D_xanes3(
    eng_list,
    x_range=[-1, 1],
    y_range=[-1, 1],
    exposure_time=0.1,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    img_sizeX=2560,
    img_sizeY=2160,
    pxl=20,
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    note="",
    md=None,
):

    insert_text("raster_2D_xanes3 contains following raster_2D_scans:")
    for eng in eng_list:
        yield from move_zp_ccd(
            eng, move_flag=1, info_flag=1, move_clens_flag=1, move_det_flag=0
        )
        yield from raster_2D_scan(
            x_range,
            y_range,
            exposure_time,
            out_x,
            out_y,
            out_z,
            out_r,
            img_sizeX,
            img_sizeY,
            pxl,
            simu,
            relative_move_flag,
            rot_first_flag,
            note,
            scan_x_flag=1,
            md=md,
        )

    insert_text("finished raster_2D_xanes3")


"""
def repeat_multipos_2D_xanes_scan2(eng_list, x_list, y_list, z_list, r_list, out_x=0, out_y=0, out_z=0, out_r=0, exposure_time=0.2,  chunk_size=5, repeat_num=1, sleep_time=60, simu=False, relative_move_flag=1, note='', md=None):

    txt = f'starting "repeat_multipos_2D_xanes_scan2", consists of following scans:'
    print(txt)
    insert_text(txt)    
    for i in range(repeat_num):
        print(f'repeat #{i}:\n ')
        yield from multipos_2D_xanes_scan2(eng_list, x_list, y_list, z_list, r_list, out_x, out_y, out_z, out_r, exposure_time,  chunk_size, simu, relative_move_flag, note, md)
        print(f'sleeping for {sleep_time} sec ......')
        yield from bps.sleep(sleep_time)
    insert_text('" repeat_multipos_2D_xanes_scan2" finished !')

"""


def multipos_count(
    x_list,
    y_list,
    z_list,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    exposure_time=0.1,
    repeat_num=1,
    sleep_time=0,
    note="",
    simu=False,
    relative_move_flag=1,
    md=None,
):
    global ZONE_PLATE
    detectors = [Andor, ic3]
    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    _md = {
        "detectors": ["Andor"],
        "motors": "zps_sx, zps_sy, zps_sz",
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "x_list": f"{x_list}",
            "y_list": f"{y_list}",
            "z_list": f"{z_list}",
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "exposure_time": exposure_time,
            "repeat_num": repeat_num,
            "sleep_time": sleep_time,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "multipos_count",
        "num_dark_images": 10,
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        "zone_plate": ZONE_PLATE,
        "motor_pos": wh_pos(print_on_screen=0),
    }

    period = max(0.05, exposure_time)
    yield from _set_andor_param(exposure_time, period=period, chunk_size=1)

    txt = f"multipos_count(x_list, y_list, z_list,  out_x={out_x}, out_y={out_y}, out_z={out_z}, out_r={out_r}, exposure_time={exposure_time}, repeat_num={repeat_num}, sleep_time={sleep_time}, note={note})"
    insert_text(txt)
    insert_text(f"x_list={x_list}")
    insert_text(f"y_list={y_list}")
    insert_text(f"z_list={z_list}")
    num = len(x_list)

    _md.update(md or {})
    try:
        dimensions = [(zps.sx.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list([Andor, ic3]) + [zps.sx, zps.sy, zps.sz, zps.pi_r])
    @run_decorator(md=_md)
    def inner_scan():
        print("\nshutter closed, taking 10 dark images...")
        yield from _close_shutter(simu=simu)

        yield from _take_image(detectors, motor, num=10)

        for repeat in range(repeat_num):
            print("\nshutter open ...")
            for i in range(num):
                yield from _open_shutter(simu=simu)
                print(
                    f"\n\nmove to position[{i+1}]: x={x_list[i]}, y={y_list[i]}, z={z_list[i]}\n\n"
                )
                yield from mv(zps.sx, x_list[i], zps.sy, y_list[i], zps.sz, z_list[i])
                x_ini = zps.sx.position
                y_ini = zps.sy.position
                z_ini = zps.sz.position
                r_ini = zps.pi_r.position

                if relative_move_flag:
                    x_target = x_ini + out_x if not (out_x is None) else x_ini
                    y_target = y_ini + out_y if not (out_y is None) else y_ini
                    z_target = z_ini + out_z if not (out_z is None) else z_ini
                    r_target = r_ini + out_r if not (out_r is None) else r_ini
                else:
                    x_target = out_x if not (out_x is None) else x_ini
                    y_target = out_y if not (out_y is None) else y_ini
                    z_target = out_z if not (out_z is None) else z_ini
                    r_target = out_r if not (out_r is None) else r_ini
                yield from trigger_and_read(
                    list([Andor, ic3]) + [zps.sx, zps.sy, zps.sz, zps.pi_r]
                )
                yield from mv(zps.pi_r, r_target)
                yield from mv(zps.sx, x_target, zps.sy, y_target, zps.sz, z_target)
                yield from trigger_and_read(
                    list([Andor, ic3]) + [zps.sx, zps.sy, zps.sz, zps.pi_r]
                )
                yield from mv(zps.sx, x_ini, zps.sy, y_ini, zps.sz, z_ini)
                yield from mv(zps.pi_r, r_ini)
            yield from _close_shutter(simu=simu)
            print(f"sleep for {sleep_time} sec ...")

    yield from inner_scan()
    print("scan finished")
    txt = get_scan_parameter()
    insert_text(txt)


def xanes_3D(
    eng_list,
    exposure_time=0.05,
    start_angle = None,
    relative_rot_angle=185,
    period=0.05,
    chunk_size=20,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    rs=2,
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    note="",
    binning = [2, 2]
):
    txt = "start 3D xanes scan, containing following fly_scan:\n"
    insert_text(txt)
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.bin_y, binning[0],
                  Andor.cam.bin_x, binning[1])
    for eng in eng_list:
        yield from move_zp_ccd(eng, move_flag=1)
        my_note = note + f"_energy={eng}"
        yield from bps.sleep(1)
        print(f"current energy: {eng}")
        # yield from fly_scan(exposure_time, relative_rot_angle=relative_rot_angle, period=period, out_x=out_x, out_y=out_y, out_z=out_z, out_r= out_r, rs=rs, note=my_note, simu=simu, relative_move_flag=relative_move_flag, traditional_sequence_flag=traditional_sequence_flag)
        yield from fly_scan(
            exposure_time,
            start_angle=start_angle,
            relative_rot_angle=relative_rot_angle,
            period=period,
            chunk_size = chunk_size,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            rs=rs,
            relative_move_flag=relative_move_flag,
            note=my_note,
            simu=simu,
            rot_first_flag=rot_first_flag,
        )
        yield from bps.sleep(1)
    yield from mv(Andor.cam.image_mode, 1)
    export_pdf(1)


# def fly_scan_repeat(exposure_time=0.03, relative_rot_angle = 185, period=0.05, chunk_size=20, out_x=0, out_y=-100, out_z=0,  out_r=0, rs=6, note='', repeat=1, sleep_time=0, simu=False, relative_move_flag=1, traditional_sequence_flag=1, md=None):
#    for i in range(repeat):
#        yield from fly_scan(exposure_time=exposure_time, relative_rot_angle = relative_rot_angle, period=period, chunk_size=chunk_size,
#                 out_x=out_x, out_y=out_y, out_z=out_z,  out_r=out_r, rs=rs, note=note, simu=simu,
#                 relative_move_flag=relative_move_flag, traditional_sequence_flag=traditional_sequence_flag, md=md)
#        if i != repeat-1:
#            yield from bps.sleep(sleep_time)
#    export_pdf(1)


def fly_scan_repeat(
    exposure_time=0.03,
    start_angle = None,
    relative_rot_angle=185,
    period=0.05,
    chunk_size=20,
    x_list=[],
    y_list=[],
    z_list=[],
    out_x=0,
    out_y=-100,
    out_z=0,
    out_r=0,
    rs=6,
    note="",
    repeat=1,
    sleep_time=0,
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    rot_back_velo=30,
    md=None,
):
    nx = len(x_list)
    ny = len(y_list)
    nz = len(z_list)
    if nx == 0 & ny == 0 & nz == 0:
        for i in range(repeat):
            yield from fly_scan(
                exposure_time=exposure_time,
                start_angle = start_angle,
                relative_rot_angle=relative_rot_angle,
                period=period,
                chunk_size=chunk_size,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                out_r=out_r,
                rs=rs,
                note=note,
                simu=simu,
                relative_move_flag=relative_move_flag,
                rot_first_flag=rot_first_flag,
                rot_back_velo=rot_back_velo,
                md=md,
            )
            print(
                f"Scan at time point {i:3d} is finished; sleep for {sleep_time:3.1f} seconds now."
            )
            insert_text(
                f"Scan at time point {i:3d} is finished; sleep for {sleep_time:3.1f} seconds now."
            )
            if i != repeat - 1:
                yield from bps.sleep(sleep_time)
        export_pdf(1)
    else:
        if nx != ny or nx != nz or ny != nz:
            print(
                "!!!!! Position lists are not equal in length. Please check your position list definition !!!!!"
            )
        else:
            for i in range(repeat):
                for j in range(nx):
                    yield from mv(
                        zps.sx, x_list[j], zps.sy, y_list[j], zps.sz, z_list[j]
                    )
                    yield from fly_scan(
                        exposure_time=exposure_time,
                        start_angle=start_angle,
                        relative_rot_angle=relative_rot_angle,
                        period=period,
                        chunk_size=chunk_size,
                        out_x=out_x,
                        out_y=out_y,
                        out_z=out_z,
                        out_r=out_r,
                        rs=rs,
                        note=note,
                        simu=simu,
                        relative_move_flag=relative_move_flag,
                        rot_first_flag=rot_first_flag,
                        rot_back_velo=rot_back_velo,
                        md=md,
                    )
                insert_text(
                    f"Scan at time point {i:3d} is finished; sleep for {sleep_time:3.1f} seconds now."
                )
                print(
                    f"Scan at time point {i:3d} is finished; sleep for {sleep_time:3.1f} seconds now."
                )
                if i != repeat - 1:
                    yield from bps.sleep(sleep_time)
            export_pdf(1)


def multi_pos_xanes_3D(
    eng_list,
    x_list,
    y_list,
    z_list,
    r_list,
    start_angle = None,
    exposure_time=0.05,
    relative_rot_angle=185,
    period=0.05,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    rs=2,
    simu=False,
    relative_move_flag=1,
    traditional_sequence_flag=1,
    note="",
    sleep_time=0,
    binning = [2, 2],
    repeat=1,
):
    n = len(x_list)
    for rep in range(repeat):
        for i in range(n):
            yield from mv(
                zps.sx,
                x_list[i],
                zps.sy,
                y_list[i],
                zps.sz,
                z_list[i],
                zps.pi_r,
                r_list[i],
            )
            txt = f"start xanes_3D at pos1: x={x_list[i]}, y={y_list[i]}, z={z_list[i]}\nrepeat:{rep}"
            insert_text(txt)
            print(f"{txt}\n##########################\n\n\n\n")
            yield from xanes_3D(
                eng_list,
                exposure_time=exposure_time,
                start_angle =start_angle,
                relative_rot_angle=relative_rot_angle,
                period=period,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                out_r=out_r,
                rs=rs,
                simu=simu,
                relative_move_flag=relative_move_flag,
                rot_first_flag=traditional_sequence_flag,
                note=note,
                binning = [2, 2],
            )
        print(f"sleep for {sleep_time} sec\n\n\n\n")
        yield from bps.sleep(sleep_time)

