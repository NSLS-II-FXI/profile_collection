import uuid
import sys
import time
from warnings import warn
from bluesky.plan_stubs import mv, mvr
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
from bluesky.utils import (Msg, short_uid as _short_uid, make_decorator)

warnings.filterwarnings('ignore')






######################### 
def _move_sample_out(out_x, out_y, out_z, out_r, repeat=1):
    '''
    move out by relative distance
    '''
    x_out = zps.sx.position + out_x
    y_out = zps.sy.position + out_y
    z_out = zps.sz.position + out_z
    r_out = zps.pi_r.position + out_r
    for i in range(repeat):
        yield from mv(zps.pi_r, r_out)
        yield from mv(zps.sx, x_out, zps.sy, y_out, zps.sz, z_out)


def _move_sample_in(in_x, in_y, in_z, in_r, repeat=1):
    '''
    move in at absolute position
    '''
    for i in range(repeat):
        yield from mv(zps.sx, in_x, zps.sy, in_y, zps.sz, in_z)
        yield from mv(zps.pi_r, in_r)


def _close_shutter(simu=False):
    if simu:
        print('testing: close shutter')
    else:
        print('closing shutter ... ')
        i = 0
        while not shutter_status.value: # if 1:  closed; if 0: open
            yield from abs_set(shutter_close, 1)
            yield from bps.sleep(1)
            i += 1
            if i > 10:
                print('fails to close shutter')
                break
        #yield from abs_set(shutter_close, 1)
        #yield from bps.sleep(1)

def _open_shutter(simu=False):
    if simu:
        print('testing: open shutter')
    else:
        print('opening shutter ... ')
        i = 0
        while shutter_status.value: # if 1:  closed; if 0: open
            yield from abs_set(shutter_open, 1)
            yield from bps.sleep(1)
            i += 1
            if i >10:
                print('fails to open shutter')
                break
        #yield from abs_set(shutter_open, 1)
        #yield from bps.sleep(1)


def _set_andor_param(exposure_time=0.1, period=0.1, chunk_size=1):
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.image_mode, 0)
    yield from mv(Andor.cam.num_images, chunk_size)
    yield from mv(Andor.cam.acquire_time, exposure_time)
    Andor.cam.acquire_period.put(period)


def _set_rotation_speed(rs=1):
    yield from abs_set(zps.pi_r.velocity, rs)


def _take_image(detectors, motor, num):
    if not (type(detectors) == list):
        detectors = list(detectors)
    if not (type(motor) == list):
        motor = list(motor)
    for i in range(num):
        yield from trigger_and_read(detectors + motor)


def _take_dark_image(detectors, motor, num_dark=1, simu=False):
    yield from _close_shutter(simu)
    yield from _take_image(detectors, motor, num_dark)


def _take_bkg_image(out_x, out_y, out_z, out_r, detectors, motor, num_bkg=1, simu=False):
    yield from _move_sample_out(out_x, out_y, out_z, out_r, repeat=2)
    yield from _take_image(detectors, motor, num_bkg)

    

def _xanes_per_step(eng, detectors, motor, move_flag=1, info_flag=0):
    yield from move_zp_ccd(eng, move_flag=move_flag, info_flag=info_flag)
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
        print ('Time for {0:s}: {1:2.2f}'.format(method.__name__, te-ts))
        return results
    return timed






def tomo_scan(start, stop, num, exposure_time=1, bkg_num=10, dark_num=10, out_x=0, out_y=0, note='', md=None):
    '''
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
    '''
    detectors=[Andor]
    yield from abs_set(detectors[0].cam.acquire_time, exposure_time)
    yield from mv(Andor.cam.num_images, 1)

    #motor_x = phase_ring.x
    motor_x = zps.sx # move sample y
    motor_x_ini = motor_x.position # initial position of motor_x
    motor_x_out = motor_x_ini + out_x  # 'out position' of motor_x
    motor_y = zps.sy # move sample y
    motor_y_ini = motor_y.position # initial position of motor_x
    motor_y_out = motor_y_ini + out_y  # 'out position' of motor_x


    motor_rot = zps.pi_r
    #motor_rot = zps.sx
    motor_rot_ini = motor_rot.position  # initial position of motor_x

    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor_rot.name],
           'x_ray_energy': XEng.position,
           'num_angles': num,
           'num_bkg_images': bkg_num,
           'num_dark_images': dark_num,
           'plan_args': {'start': start, 'stop': stop, 'num': num,
                         'exposure_time': exposure_time,
                         'bkg_num': bkg_num, 'dark_num': dark_num,
                         'out_x': out_x, 'out_y': out_y, 
                         'note': note if note else 'None'},
           'plan_name': 'tomo_scan',
           'plan_pattern': 'linspace',
           'plan_pattern_module': 'numpy',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos':  wh_pos(print_on_screen=0),
            }
    _md.update(md or {})
    try:  dimensions = [(motor_rot.hints['fields'], 'primary')]
    except (AttributeError, KeyError):    pass
    else: _md['hints'].setdefault('dimensions', dimensions)
    steps = np.linspace(start, stop, num)
    @stage_decorator(list(detectors) + [motor_rot, motor_x])
    @run_decorator(md=_md)
    def tomo_inner_scan():
        #close shutter, dark images
        print('\nshutter closed, taking dark images...')
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(2)
        yield from abs_set(shutter_close, 1, wait=True)
        for num in range(dark_num):   # close the shutter, and take 10(default) dark image when stage is at out position
            yield from trigger_and_read(list(detectors) + [motor_rot])
        # Open shutter, tomo images
        yield from abs_set(shutter_open, 1, wait=True)
        time.sleep(2)
        yield from abs_set(shutter_open, 1, wait=True)
        print ('shutter opened, pi_x position: {0}\n\nstarting tomo_scan...'.format(motor_x.position))
        for step in steps:  # take tomography images
            yield from one_1d_step(detectors, motor_rot, step)
#        yield from mv_stage(motor_rot, motor_rot_ini)

        print ('\n\nTaking background images...\npi_x position: {0}'.format(motor_x.position))
        yield from mv_stage(motor_x, motor_x_out)
        yield from mv_stage(motor_y, motor_y_out)
        for num in range(bkg_num):    # take 10 background image when stage is at out position
            yield from trigger_and_read(list(detectors) + [motor_rot])
        # close shutter, move sample back
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(2)
        yield from abs_set(shutter_close, 1, wait=True)
        yield from mv_stage(motor_x, motor_x_ini)
        yield from mv_stage(motor_y, motor_y_ini)
        yield from mv_stage(motor_rot, motor_rot_ini)
 #   return (yield from tomo_inner_scan())
    print('tomo-scan is disabled, try to use fly_scan')





def xanes_scan(eng_list, exposure_time=0.1, chunk_size=5, out_x=0, out_y=0, out_z=0, out_r=0, simu=False, note='', md=None):
    '''
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

    out_x: float
           relative move amount of zps.sx motor
           (in unit of um, to move out sample)

    out_y: float
           relative move amount of zps.sy motor
           (in unit of um, to move out sample)

    note: string

    '''
    detectors=[Andor, ic3]
    period = exposure_time if exposure_time >= 0.05 else 0.05
    yield from _set_andor_param(exposure_time, period, chunk_size)

    motor_eng = XEng
    eng_ini = XEng.position
    motor_x = zps.sx # move sample y
    motor_x_ini = motor_x.position # initial position of motor_x
    motor_x_out = motor_x_ini + out_x  # 'out position' of motor_x
    motor_y = zps.sy # move sample y
    motor_y_ini = motor_y.position # initial position of motor_y
    motor_y_out = motor_y_ini + out_y  # 'out position' of motor_y
    motor_z = zps.sz
    motor_z_ini = motor_z.position # initial position of motor_y
    motor_z_out = motor_z_ini + out_z  # 'out position' of motor_y
    motor_r = zps.pi_r
    motor_r_ini = motor_r.position # initial position of motor_y
    motor_r_out = motor_r_ini + out_r  # 'out position' of motor_y

    rs_ini = motor_r.velocity.value
    motor = [motor_eng, motor_x, motor_y, motor_z, motor_r]

    _md = {'detectors': [det.name for det in detectors],
           'motors': [mot.name for mot in motor],
           'num_eng': len(eng_list),
           'num_bkg_images': chunk_size,
           'num_dark_images': chunk_size,
           'chunk_size': chunk_size,
           'out_x': out_x,
           'out_y': out_y,
           'exposure_time': exposure_time,
           'eng_list': eng_list,
           'XEng': XEng.position,
           'plan_args': {'eng_list': 'eng_list',
                         'exposure_time': exposure_time,                         
                         'chunk_size': chunk_size,
                         'out_x': out_x,
                         'out_y': out_y,
                         'out_z': out_z,
                         'out_r': out_r,
                         'note': note if note else 'None'
                        },     
           'plan_name': 'xanes_scan',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos':  wh_pos(print_on_screen=0),
            }
    _md.update(md or {})
    try:   dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):  pass
    else:   _md['hints'].setdefault('dimensions', dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def xanes_inner_scan():
        print('\ntake {} dark images...'.format(chunk_size))
        yield from _set_rotation_speed(rs=30)
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

        print('\nopening shutter, and start xanes scan: {} images per each energy... '.format(chunk_size))
        yield from _open_shutter(simu)
        for eng in eng_list:
            yield from _xanes_per_step(eng, detectors, motor, move_flag=1, info_flag=0)  
        yield from _move_sample_out(out_x, out_y, out_z, out_r, repeat=2)
        print('\ntake bkg image after xanes scan, {} per each energy...'.format(chunk_size))
        for eng in eng_list:
            yield from _xanes_per_step(eng, detectors, motor, move_flag=1, info_flag=0)
        yield from _move_sample_in(motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini, repeat=2)
        yield from move_zp_ccd(eng_ini, info_flag=0)

        print('closing shutter')
        yield from _close_shutter(simu)
        
    yield from xanes_inner_scan()
    txt1 = get_scan_parameter()    
    eng_list = np.round(eng_list, 5)
    if len(eng_list) > 10:
        txt2 = f'eng_list: {eng_list[0:10]}, ... {eng_list[-5:]}\n'
    else:
        txt2 = f'eng_list: {eng_list}'
    txt = txt1 + '\n' + txt2
    insert_text(txt)
    print(txt)
    



def xanes_scan2(eng_list, exposure_time=0.1, chunk_size=5, out_x=0, out_y=0, out_z=0, out_r=0, simu=False, note='', md=None):
    '''
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

    out_x: float
           relative move amount of zps.sx motor
           (in unit of um, to move out sample)

    out_y: float
           relative move amount of zps.sy motor
           (in unit of um, to move out sample)

    note: string

    '''
    detectors=[Andor, ic3]
    period = exposure_time if exposure_time >= 0.05 else 0.05
    yield from _set_andor_param(exposure_time,period, chunk_size)

    motor_eng = XEng
    eng_ini = XEng.position
    motor_x = zps.sx # move sample y
    motor_x_ini = motor_x.position # initial position of motor_x
    motor_x_out = motor_x_ini + out_x  # 'out position' of motor_x
    motor_y = zps.sy # move sample y
    motor_y_ini = motor_y.position # initial position of motor_y
    motor_y_out = motor_y_ini + out_y  # 'out position' of motor_y
    motor_z = zps.sz
    motor_z_ini = motor_z.position # initial position of motor_y
    motor_z_out = motor_z_ini + out_z  # 'out position' of motor_y
    motor_r = zps.pi_r
    motor_r_ini = motor_r.position # initial position of motor_y
    motor_r_out = motor_r_ini + out_r  # 'out position' of motor_y

    rs_ini = motor_r.velocity.value

    motor = [motor_eng, motor_x, motor_y, motor_z, motor_r]

    _md = {'detectors': [det.name for det in detectors],
           'motors': [mot.name for mot in motor],
           'num_eng': len(eng_list),
           'num_bkg_images': chunk_size,
           'num_dark_images': chunk_size,
           'chunk_size': chunk_size,
           'out_x': out_x,
           'out_y': out_y,
           'exposure_time': exposure_time,
           'eng_list': eng_list,
           'XEng': XEng.position,
           'plan_args': {'eng_list': 'eng_list',
                         'exposure_time': exposure_time,
                         'chunk_size': chunk_size,
                         'out_x': out_x,
                         'out_y': out_y,
                         'out_z': out_z,
                         'our_r': out_r,
                         'note': note if note else 'None'
                         },              
           'plan_name': 'xanes_scan2',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos':  wh_pos(print_on_screen=0),
            }
    _md.update(md or {})
    try:   dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):  pass
    else:   _md['hints'].setdefault('dimensions', dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def xanes_inner_scan():
        yield from _set_rotation_speed(rs=30)
        #yield from abs_set(motor_r.velocity, 30)
        # take dark image
        print('\ntake {} dark images...'.format(chunk_size))
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)


        print('\nopening shutter, and start xanes scan: {} images per each energy... '.format(chunk_size))
        yield from _open_shutter(simu)

        for eng in eng_list:
            yield from _xanes_per_step(eng, detectors, motor, move_flag=1, info_flag=0)
            yield from _take_bkg_image(out_x, out_y, out_z, out_r, detectors, motor, num_bkg=1, simu=simu)
            yield from _move_sample_in(motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini, repeat=2)

        yield from move_zp_ccd(eng_ini, move_flag=1, info_flag=0)
        print('closing shutter')
        yield from _close_shutter(simu=simu)

    yield from xanes_inner_scan()
    txt1 = get_scan_parameter()
    eng_list = np.round(eng_list, 5)
    if len(eng_list) > 10:
        txt2 = f'eng_list: {eng_list[0:10]}, ... {eng_list[-5:]}\n'
    else:
        txt2 = f'eng_list: {eng_list}'
    txt = txt1 + '\n' + txt2
    insert_text(txt)
    print(txt)


def mv_stage(motor, pos):
        grp = _short_uid('set')
        yield Msg('checkpoint')
        yield Msg('set', motor, pos, group=grp)
        yield Msg('wait', None, group=grp)


def eng_scan(eng_start, eng_end, steps, num=10, detectors=[ic3, ic4], delay_time=1, note='', md=None):
    '''
    Input:
    ----------
        eng_start: float, energy start in keV

        eng_end: float, energy stop in keV

        steps: int, number of energies

        num: int, number of repeating scans

        detectors: list, detector list, e.g.[ic3, ic4, Andor]

        delay_time: float, delay time after moving motors, in sec
    
    '''
    det = [det.name for det in detectors]
    det_name = ''
    for i in range(len(det)):
        det_name += det[i]
        det_name += ', '
    det_name = '[' + det_name[:-2] + ']'
    txt = f'eng_scan(eng_start={eng_start}, eng_end={eng_end}, steps={steps}, num={num}, detectors={det_name}, delay_time={delay_time})\n  Consisting of:\n'
    insert_text(txt)
    print(txt)
    check_eng_range([eng_start, eng_end])
#    set_ic_dwell_time(dwell_time=dwell_time)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for i in range(num):
  #      yield from scan([ic3, ic4], XEng, eng_start/1000, eng_end/1000, steps)
        yield from eng_scan_delay(eng_start, eng_end, steps, detectors, delay_time=delay_time, note='')
        h = db[-1]
        y0 = np.array(list(h.data(ic3.name)))
        y1 = np.array(list(h.data(ic4.name)))

        r = np.log(y0/y1)
        x = np.linspace(eng_start, eng_end, steps)

        ax1.plot(x, r, '.-')
        r_dif = np.array([0] + list(np.diff(r)))
        ax2.plot(x, r_dif, '.-')

    ax1.title.set_text('ratio of: {0}/{1}'.format(ic3.name, ic4.name))
    ax2.title.set_text('differential of: {0}/{1}'.format(ic3.name, ic4.name))
    fig.subplots_adjust(hspace=.5)
    plt.show()
    txt_finish='## "eng_scan()" finished'
    insert_text(txt_finish)

    


def eng_scan_delay(start, stop, num, detectors=[ic3, ic4], delay_time=1, note='', md=None):
    # detectors=[ic3, ic4]
    motor_x = XEng
    motor_x_ini = motor_x.position # initial position of motor_x
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor_x.name],
           'XEng': XEng.position,
           'plan_name': 'eng_scan_delay',
           'plan_args': {'start': start,
                         'stop': stop,
                         'num': num,
                         'detectors': 'detectors',
                         'delay_time': delay_time,
                         'note': note if note else 'None'
                         },   
           'plan_pattern': 'linspace',
           'plan_pattern_module': 'numpy',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos':  wh_pos(print_on_screen=0),
            }
    _md.update(md or {})
    try:  dimensions = [(motor_x.hints['fields'], 'primary')]
    except (AttributeError, KeyError):    pass
    else: _md['hints'].setdefault('dimensions', dimensions)
    steps = np.linspace(start, stop, num)
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
    scan_id = h.start['scan_id']    
    det = [det.name for det in detectors]
    det_name = ''
    for i in range(len(det)):
        det_name += det[i]
        det_name += ', '
    det_name = '[' + det_name[:-2] + ']'
    txt1 = get_scan_parameter()
    txt2 = f'detectors = {det_name}'
    txt = txt1 + '\n' + txt2
    insert_text(txt)
    print(txt) 



def fly_scan(exposure_time=0.1, relative_rot_angle = 180, period=0.15, chunk_size=20, out_x=0, out_y=2000, out_z=0,  out_r=0, rs=1, note='', simu=False, md=None):

    motor_x_ini = zps.sx.position
    motor_x_out = motor_x_ini + out_x
    motor_y_ini = zps.sy.position
    motor_y_out = motor_y_ini + out_y
    motor_z_ini = zps.sz.position
    motor_z_out = motor_z_ini + out_z
    motor_r_ini = zps.pi_r.position
    motor_r_out = motor_r_ini + out_r

    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    detectors = [Andor, ic3]
    offset_angle = -2.0 * rs
    current_rot_angle = zps.pi_r.position

    target_rot_angle = current_rot_angle + relative_rot_angle
    _md = {'detectors': ['Andor'],
           'motors': [mot.name for mot in motor],
           'XEng': XEng.position,
           'ion_chamber': ic3.name,
           'plan_args': {'exposure_time': exposure_time,
                         'relative_rot_angle': relative_rot_angle,
                         'period': period,
                         'chunk_size': chunk_size,
                         'out_x': out_x,
                         'out_y': out_y,
                         'out_z': out_z,
                         'out_r': out_r,
                         'rs': rs,
                         'note': note if note else 'None',
                        },
           'plan_name': 'fly_scan',
           'num_bkg_images': chunk_size,
           'num_dark_images': chunk_size,
           'chunk_size': chunk_size,
           'plan_pattern': 'linspace',
           'plan_pattern_module': 'numpy',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos': wh_pos(print_on_screen=0),
            }
    _md.update(md or {})
    try:  dimensions = [(zps.pi_r.hints['fields'], 'primary')]
    except (AttributeError, KeyError):    pass
    else: _md['hints'].setdefault('dimensions', dimensions)

    yield from _set_andor_param(exposure_time=exposure_time, period=period, chunk_size=chunk_size)
    yield from _set_rotation_speed(rs=rs)
    print('set rotation speed: {} deg/sec'.format(rs))


    @stage_decorator(list(detectors) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        #close shutter, dark images: numer=chunk_size (e.g.20)
        print('\nshutter closed, taking dark images...')
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

        #open shutter, tomo_images
        yield from _open_shutter(simu=simu)
        print ('\nshutter opened, taking tomo images...')
        yield from mv(zps.pi_r, current_rot_angle + offset_angle)
        status = yield from abs_set(zps.pi_r, target_rot_angle, wait=False)
        yield from bps.sleep(2)
        while not status.done:
            yield from trigger_and_read(list(detectors) + motor)
        # bkg images
        print ('\nTaking background images...')
        yield from _set_rotation_speed(rs=30)
        yield from _take_bkg_image(out_x, out_y, out_z, out_r, detectors, motor, num_bkg=1, simu=False)
        yield from _close_shutter(simu=simu)
        yield from _move_sample_in(motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini)

    uid = yield from fly_inner_scan()
    print('scan finished')
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)
    return uid




def grid2D_rel(motor1, start1, stop1, num1, motor2, start2, stop2, num2, exposure_time=0.05, delay_time=0, note='', md=None):
    # detectors=[ic3, ic4]

    detectors=[Andor, ic3]
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.image_mode, 0)
    yield from mv(Andor.cam.num_images, 1)
    yield from mv(detectors[0].cam.acquire_time, exposure_time)
    Andor.cam.acquire_period.put(exposure_time)


    motor1_ini = motor1.position
    motor2_ini = motor2.position

    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor1.name, motor2.name],
           'XEng': XEng.position,
           'plan_name': 'grid2D_rel',
           'plan_args': {'motor1': motor1.name,'start1':start1, 'stop1':stop1, 'num1':num1,
                         'motor2': motor2.name,'start2':start2, 'stop2':stop2, 'num2':num2,
                         'exposure_time':exposure_time, 'delay_time': delay_time,
                         'note': note if note else 'None',
                         },   
           'plan_pattern': 'linspace',
           'plan_pattern_module': 'numpy',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos':  wh_pos(print_on_screen=0),
            }
    _md.update(md or {})
    try:  dimensions = [(motor1.hints['fields'], 'primary')]
    except (AttributeError, KeyError):    pass
    else: _md['hints'].setdefault('dimensions', dimensions)

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
    scan_id = h.start['scan_id']    
    det = [det.name for det in detectors]
    det_name = ''
    for i in range(len(det)):
        det_name += det[i]
        det_name += ', '
    det_name = '[' + det_name[:-2] + ']'
    txt1 = get_scan_parameter()
    txt2 = f'detectors = {det_name}'
    txt = txt1 + '\n' + txt2 + '\n'
    insert_text(txt)
    print(txt) 




def delay_count(detectors, num=1, delay=None, *, note='', md=None):
    """
    same function as the default "count", 
    re_write it in order to add auto-logging
    """
    if num is None:
        num_intervals = None
    else:
        num_intervals = num - 1
    _md = {'detectors': [det.name for det in detectors],
           'num_points': num,
           'XEng': XEng.position,
           'num_intervals': num_intervals,
           'plan_args': {'detectors': 'detectors', 'num': num, 'delay': delay},
           'plan_name': 'delay_count',
           'hints': {},
           'note': note if note else 'None',
           }
    _md.update(md or {})
    _md['hints'].setdefault('dimensions', [(('time',), 'primary')])

    @bpp.stage_decorator(detectors)
    @bpp.run_decorator(md=_md)
    def inner_count():
        return (yield from bps.repeat(partial(bps.trigger_and_read, detectors),
                                      num=num, delay=delay))
    uid = yield from inner_count()
    h = db[-1]
    scan_id = h.start['scan_id']
    det = [det.name for det in detectors]
    det_name = ''
    for i in range(len(det)):
        det_name += det[i]
        det_name += ', '
    det_name = '[' + det_name[:-2] + ']'

    txt1 = get_scan_parameter()
    txt2 = f'detectors = {det_name}'
    txt = txt1 + '\n' + txt2
    insert_text(txt)
    print(txt)
    return uid    


def delay_scan(detectors, motor, start, stop, steps, exposure_time=0.1,  sleep_time=1.0, plot_flag=0, note='', md=None):
    '''
    add sleep_time to regular 'scan' for each scan_step
    '''
    if Andor in detectors:
        yield from _set_andor_param(exposure_time, period=exposure_time, chunk_size=1)

    #motor = dcm.th2
    #motor = pzt_dcm_th2.setpos
    motor_ini = motor.position
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'XEng': XEng.position,
           'plan_args': {'detectors': 'detectors',
                         'motor': motor.name,
                         'start': start,
                         'stop': stop,
                         'steps': steps,  
                         'exposure_time':exposure_time,                       
                         'sleep_time': sleep_time,
                         'plot_flag': plot_flag,
                         'note': note if note else 'None'
                         },
           'plan_name': 'delay_scan',
           'hints': {},
           'motor_pos':  wh_pos(print_on_screen=0),
           'operator': 'FXI'
            }
    _md.update(md or {})
    try:
        dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)

    my_var = np.linspace(start, stop, steps)
    @stage_decorator(list(detectors) + [motor])
    @run_decorator(md=_md)
    def delay_inner_scan():
        for x in my_var:
            yield from mv_stage(motor, x)
            yield from bps.sleep(sleep_time)
            yield from trigger_and_read(list(detectors + [motor]))
        yield from mv(motor, motor_ini)
    uid = yield from delay_inner_scan()
    h = db[-1]
    if plot_flag:        
        x = np.linspace(start, stop, steps)
        y = list(h.data(detectors[0].name))
        plt.figure();
        plt.plot(x,y);plt.xlabel(motor.name);plt.ylabel(detectors[0].name)
        plt.title('scan# {}'.format(h.start['scan_id']))
    scan_id = h.start['scan_id']
    det = [det.name for det in detectors]
    det_name = ''
    for i in range(len(det)):
        det_name += det[i]
        det_name += ', '
    det_name = '[' + det_name[:-2] + ']'

    txt1 = get_scan_parameter()
    txt2 = f'detectors = {det_name}'
    txt = txt1 + '\n' + txt2
    insert_text(txt)
    print(txt)
    return uid


'''
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

'''



def raster_2D_scan(x_range=[-1,1],y_range=[-1,1],exposure_time=0.1, out_x=0, out_y=0, out_z=0, out_r=0, img_sizeX=2560,img_sizeY=2160,pxl=17.2, note='', simu=False, md=None):

    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]
    detectors = [Andor, ic3]
    yield from _set_andor_param(exposure_time=exposure_time, period=exposure_time, chunk_size=1)

    x_initial = zps.sx.position    
    y_initial = zps.sy.position
    z_initial = zps.sz.position
    r_initial = zps.pi_r.position

    img_sizeX = np.int(img_sizeX)
    img_sizeY = np.int(img_sizeY)
    x_range = np.int_(x_range)
    y_range = np.int_(y_range)
    
    print('hello1')
    _md = {'detectors': [det.name for det in detectors],
           'motors': [mot.name for mot in motor],
           'num_bkg_images': 5,
           'num_dark_images': 5,
           'x_range': x_range,
           'y_range': y_range,
           'out_x': out_x,
           'out_y': out_y,
           'out_z': out_z,
           'exposure_time': exposure_time,
           'XEng': XEng.position,
           'plan_args': {'x_range': x_range,
                         'y_range': y_range,
                         'exposure_time': exposure_time,                         
                         'out_x': out_x,
                         'out_y': out_y,
                         'out_z': out_z,
                         'out_r': out_r,
                         'img_sizeX': img_sizeX,
                         'img_sizeY': img_sizeY,
                         'pxl': pxl,
                         'note': note if note else 'None'
                        },     
           'plan_name': 'raster_2D',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos':  wh_pos(print_on_screen=0),
            }
    _md.update(md or {})
    try:   dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):  pass
    else:   _md['hints'].setdefault('dimensions', dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def raster_2D_inner():
        # take dark image
        print('take 5 dark image')
        yield from _take_dark_image(detectors, motor, num_dark=5, simu=False)

        print('open shutter ...')
        yield from _open_shutter(simu)

        print('taking mosaic image ...')
        for ii in np.arange(x_range[0],x_range[1]+1):
            yield from mv(zps.sx, x_initial + ii*img_sizeX*pxl*1.0/1000)
            yield from mv(zps.sx, x_initial + ii*img_sizeX*pxl*1.0/1000)
            sleep_time = (x_range[-1] - x_range[0]) * img_sizeX*pxl*1.0/1000 / 600
            yield from bps.sleep(sleep_time)
            for jj in np.arange(y_range[0], y_range[1]+1):
                yield from mv(zps.sy, y_initial + jj*img_sizeY*pxl*1.0/1000)
                yield from _take_image(detectors, motor, 1)
#                yield from trigger_and_read(list(detectors) + motor)


        print('moving sample out to take 5 background image')
        yield from _take_bkg_image(out_x, out_y, out_z, out_r, detectors, motor, num_bkg=5, simu=simu)
        
        # move sample in
        yield from _move_sample_in(x_initial, y_initial, z_initial, r_initial, repeat=1)

        print('closing shutter')
        yield from _close_shutter(simu)
    
    yield from raster_2D_inner()
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)






def multipos_2D_xanes_scan(eng_list, x_list, y_list, z_list, r_list, out_x, out_y, out_z, out_r, exposure_time=0.1, repeat_num=1, sleep_time=0, note=''):
    num = len(x_list)
    txt = f'Multipos_2D_xanes_scan(eng_list, x_list, y_list, z_list, out_x={out_x}, out_y={out_y}, out_z={out_z}, out_r={out_r}, exposure_time={exposure_time}, note={note})'
    insert_text(txt)
    txt = 'Take 2D_xanes at multiple position, containing following scans:'
    insert_text(txt)
    for rep in range(repeat_num):
        print(f'round: {rep}')
        for i in range(num):
            print(f'current position[{i}]: x={x_list[i]}, y={y_list[i]}, z={z_list[i]}\n')
            my_note = note + f'_position_{i}: x={x_list[i]}, y={y_list[i]}, z={z_list[i]}'
            yield from mv(zps.sx, x_list[i], zps.sy, y_list[i], zps.sz, z_list[i], zps.pi_r, r_list[i])
            yield from xanes_scan2(eng_list, exposure_time=exposure_time, chunk_size=5, out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r, note=my_note)
        yield from bps.sleep(sleep_time)
    yield from mv(zps.sx, x_list[0], zps.sy, y_list[0], zps.sz, z_list[0], zps.pi_r, r_list[0])
    insert_text('Finished the multipos_2D_xanes_scan')




def multipos_2D_xanes_scan2(eng_list, x_list, y_list, z_list, r_list, out_x=0, out_y=0, out_z=0, out_r=0, exposure_time=0.1, repeat_num=1, sleep_time=0, chunk_size=5, simu=False, note='', md=None):
    '''
    Different from multipos_2D_xanes_scan. In the current scan, it take image at all locations and then move out sampel to take background image.

    For example:
    RE(multipos_2D_xanes_scan2(Ni_eng_list, x_list=[0,1,2], y_list=[2,3,4], z_list=[0,0,0], r_list=[0,0,0], out_x=1000, out_y=0, out_z=0, out_r=90, repeat_num=2, sleep_time=60, note='sample')
    
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

    out_x: float
           relative move amount of zps.sx motor
           (in unit of um, to move out sample)

    out_y: float
           relative move amount of zps.sy motor
           (in unit of um, to move out sample)

    out_z: float
           relative move amount of zps.sz motor
           (in unit of um, to move out sample)

    out_r: float
           relative move amount of zps.pi_r motor
           (in unit of degree, to move out sample)

    exposure_time: float
           in unit of seconds

    repeat_num: int
           nums to repeat xanes scan

    sleep_time: float(int)
           in unit of seconds

    chunk_size: int
           number of background images == num of dark images ==  num of image for each energy

    note: string
    
    '''    
    detectors=[Andor, ic3]
    period = max(0.05, exposure_time)
    yield from _set_andor_param(exposure_time, period=period, chunk_size=chunk_size)

    
    eng_ini = XEng.position

    motor_x = zps.sx # move sample y
    motor_x_ini = motor_x.position # initial position of motor_x
    motor_x_out = motor_x_ini + out_x  # 'out position' of motor_x
    motor_y = zps.sy # move sample y
    motor_y_ini = motor_y.position # initial position of motor_y
    motor_y_out = motor_y_ini + out_y  # 'out position' of motor_y
    motor_z = zps.sz
    motor_z_ini = motor_z.position # initial position of motor_y
    motor_z_out = motor_z_ini + out_z  # 'out position' of motor_y
    motor_r = zps.pi_r
    motor_r_ini = motor_r.position # initial position of motor_y
    motor_r_out = motor_r_ini + out_r  # 'out position' of motor_y

    motor = [XEng, motor_x, motor_y, motor_z, motor_r]

    _md = {'detectors': [det.name for det in detectors],
           'motors': [mot.name for mot in motor],
           'num_eng': len(eng_list),
           'num_bkg_images': chunk_size,
           'num_dark_images': chunk_size,
           'chunk_size': chunk_size,
           'out_x': out_x,
           'out_y': out_y,
           'exposure_time': exposure_time,
           'eng_list': eng_list,
           'num_pos': len(x_list),
           'XEng': XEng.position,
           'plan_args': {'eng_list': 'eng_list',
                         'x_list': x_list, 'y_list': y_list, 'z_list': z_list, 'r_list': r_list,                    
                         'out_x': out_x, 'out_y': out_y, 'out_z': out_z, 'out_r': out_r,
                         'exposure_time': exposure_time,  
                         'repeat_num': repeat_num,  
                         'sleep_time': sleep_time,                      
                         'chunk_size': chunk_size,
                         'note': note if note else 'None',
                        },     
           'plan_name': 'multipos_2D_xanes_scan2',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos':  wh_pos(print_on_screen=0),
            }
    _md.update(md or {})
    try:   dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):  pass
    else:   _md['hints'].setdefault('dimensions', dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def inner_scan():
        # close shutter and take dark image
        num = len(x_list) # num of position points
        print(f'\ntake {chunk_size} dark images...')
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=False)

        # start repeating xanes scan
        print(f'\nopening shutter, and start xanes scan: {chunk_size} images per each energy... ')
        print(f'start to take repeating xanes scan at {num} positions for {repeat_num} times and sleep for {sleep_time} seconds interval')
        for rep in range(repeat_num):
            # open shutter
            yield from _open_shutter(simu)
            print(f'round: {rep}')
        
            for eng in eng_list:
                yield from move_zp_ccd(eng, move_flag=1, info_flag=0)
                yield from _open_shutter(simu)
                for i in range(num):
                    # take image at multiple positions
                    yield from mv(zps.sx, x_list[i], zps.sy, y_list[i], zps.sz, z_list[i], zps.pi_r, r_list[i])
                    yield from mv(zps.sx, x_list[i], zps.sy, y_list[i], zps.sz, z_list[i], zps.pi_r, r_list[i])
                    yield from trigger_and_read(list(detectors) + motor)
                # move sample out to take background
                yield from _take_bkg_image(out_x, out_y, out_z, out_r, detectors, motor, num_bkg=1, simu=simu)
                # move sample in to the first position    
                yield from _move_sample_in(motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini)   
            
            # end of eng_list
            # close shutter and sleep
            yield from _close_shutter(simu)
            print(f'sleep for {sleep_time} sec...')
            yield from bps.sleep(sleep_time)
        # end of rep        
        yield from mv(zps.sx, x_list[0], zps.sy, y_list[0], zps.sz, z_list[0], zps.pi_r, r_list[0])

    yield from inner_scan()
    txt1 = get_scan_parameter()
    eng_list = np.round(eng_list, 5)
    if len(eng_list) > 10:
        txt2 = f'eng_list: {eng_list[0:10]}, ... {eng_list[-5:]}\n'
    else:
        txt2 = f'eng_list: {eng_list}'
    txt = txt1 + '\n' + txt2



def multipos_count(x_list, y_list, z_list,  out_x, out_y, out_z, out_r, exposure_time=0.1, repeat_num=1, sleep_time=0, note='', simu=False, md=None):

    detectors = [Andor, ic3]
    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    _md = {'detectors': ['Andor'],
           'motors': 'zps_sx, zps_sy, zps_sz',
           'XEng': XEng.position,
           'ion_chamber': ic3.name,
           'plan_args': {'x_list': f'{x_list}',
                         'y_list': f'{y_list}',
                         'z_list': f'{z_list}',                  
                         'out_x': out_x,
                         'out_y': out_y,
                         'out_z': out_z,
                         'exposure_time': exposure_time,
                         'repeat_num': repeat_num,
                         'sleep_time': sleep_time,
                         'note': note if note else 'None',
                        },
           'plan_name': 'multipos_count',
           'num_dark_images': 10,
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos': wh_pos(print_on_screen=0),    
        }

    period = max(0.05, exposure_time)
    yield from _set_andor_param(exposure_time, period=period, chunk_size=1)

    txt = f'multipos_count(x_list, y_list, z_list,  out_x={out_x}, out_y={out_y}, out_z={out_z}, out_r={out_r}, exposure_time={exposure_time}, repeat_num={repeat_num}, sleep_time={sleep_time}, note={note})'
    insert_text(txt)
    insert_text(f'x_list={x_list}')
    insert_text(f'y_list={y_list}')
    insert_text(f'z_list={z_list}')
    num = len(x_list)

    _md.update(md or {})
    try:  dimensions = [(zps.sx.hints['fields'], 'primary')]
    except (AttributeError, KeyError):    pass
    else: _md['hints'].setdefault('dimensions', dimensions)


    @stage_decorator(list([Andor, ic3])+[zps.sx, zps.sy, zps.sz, zps.pi_r])
    @run_decorator(md=_md)
    def inner_scan():
        print('\nshutter closed, taking 10 dark images...')
        yield from _close_shutter(simu=simu)

        yield from _take_image(detectors, motor, num=10)

        for repeat in range(repeat_num):
            print('\nshutter open ...')
            for i in range(num):
                yield from _open_shutter(simu=simu)
                print(f'\n\nmove to position[{i+1}]: x={x_list[i]}, y={y_list[i]}, z={z_list[i]}\n\n')
                yield from mv(zps.sx, x_list[i], zps.sy, y_list[i], zps.sz, z_list[i])
                x_ini = zps.sx.position
                y_ini = zps.sy.position
                z_ini = zps.sz.position
                r_ini = zps.pi_r.position
                x_target = x_ini + out_x
                y_target = y_ini + out_y
                z_target = z_ini + out_z
                r_target = r_ini + out_r
                yield from trigger_and_read(list([Andor, ic3])+[zps.sx, zps.sy, zps.sz, zps.pi_r])
                yield from mv(zps.pi_r, r_target)
                yield from mv(zps.sx, x_target, zps.sy, y_target, zps.sz, z_target)
                yield from trigger_and_read(list([Andor, ic3])+[zps.sx, zps.sy, zps.sz, zps.pi_r])
                yield from mv(zps.sx, x_ini, zps.sy, y_ini, zps.sz, z_ini)
                yield from mv(zps.pi_r, r_ini)
            yield from _close_shutter(simu=simu)
            print(f'sleep for {sleep_time} sec ...')
            yield from bps.sleep(sleep_time)    
    yield from inner_scan()
    print('scan finished')
    txt = get_scan_parameter()
    insert_text(txt) 




def xanes_3D(eng_list, exposure_time=0.05, relative_rot_angle=180, period=0.05, out_x=0, out_y=0, out_z=0, out_r=0, rs=2, simu=False, note=''):
    txt = 'start 3D xanes scan, containing following fly_scan:\n'
    insert_text(txt)
    for eng in eng_list:        
        yield from move_zp_ccd(eng, move_flag=1)
        my_note = note + f'_energy={eng}'
        yield from bps.sleep(1)
        print(f'current energy: {eng}')
        yield from fly_scan(exposure_time, relative_rot_angle=relative_rot_angle, period=period, out_x=out_x, out_y=out_y, out_z=out_z, out_r= out_r, rs=rs, note=my_note, simu=simu)
        yield from bps.sleep(1)
    export_pdf(1)









