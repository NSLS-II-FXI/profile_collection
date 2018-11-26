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





def xanes_scan(eng_list, exposure_time=0.1, chunk_size=5, out_x=0, out_y=0, out_z=0, out_r=0, note='', md=None):
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
    detectors=[Andor]
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.image_mode, 0)
    yield from mv(Andor.cam.num_images, chunk_size)
    yield from mv(detectors[0].cam.acquire_time, exposure_time)
    detectors[0].cam.acquire_period.put(exposure_time)
    motor = XEng
    eng_ini = XEng.position
    motor_ini = motor.position
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

    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'x_ray_energy': motor.position,
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

    @stage_decorator(list(detectors) + [motor, motor_x, motor_y, motor_z, motor_r])
    @run_decorator(md=_md)
    def xanes_inner_scan():
        print('\ntake {} dark images...'.format(chunk_size))
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)
        yield from trigger_and_read(list(detectors) + [motor, motor_x, motor_y, motor_z, motor_r])


        print('\nopening shutter, and start xanes scan: {} images per each energy... '.format(chunk_size))
        yield from abs_set(shutter_open, 1, wait=True)
        time.sleep(2)
        yield from abs_set(shutter_open, 1, wait=True)
        time.sleep(1)
        for eng in eng_list:
            yield from move_zp_ccd(eng, move_flag=1, info_flag=0)
            yield from trigger_and_read(list(detectors) + [motor, motor_x, motor_y, motor_z, motor_r])

        yield from mv(motor_r, motor_r_out)
        yield from mv(motor_x, motor_x_out, motor_y, motor_y_out, motor_z, motor_z_out)

        print('\ntake bkg image after xanes scan, {} per each energy...'.format(chunk_size))
        for eng in eng_list:
#            yield from mv_stage(motor, eng)
            yield from move_zp_ccd(eng, info_flag=0)
            yield from trigger_and_read(list(detectors) + [motor, motor_x, motor_y, motor_z, motor_r])
#            for i in range(bkg_num):
#                yield from trigger_and_read(list(detectors) + [motor])


        yield from mv(motor_x, motor_x_ini, motor_y, motor_y_ini, motor_z, motor_z_ini)
        yield from mv(motor_r, motor_r_ini)
        
        yield from move_zp_ccd(eng_ini, info_flag=0)

        print('closing shutter')
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)
        yield from abs_set(shutter_close, 1, wait=True)
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
    



def xanes_scan2(eng_list, exposure_time=0.1, chunk_size=5, out_x=0, out_y=0, out_z=0, out_r=0, note='', md=None):
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
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.image_mode, 0)
    yield from mv(Andor.cam.num_images, chunk_size)
    yield from mv(detectors[0].cam.acquire_time, exposure_time)
    detectors[0].cam.acquire_period.put(exposure_time)
    motor = XEng
    eng_ini = XEng.position
    motor_ini = motor.position
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

    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'x_ray_energy': motor.position,
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

    @stage_decorator(list(detectors) + [motor, motor_x, motor_y, motor_z, motor_r])
    @run_decorator(md=_md)
    def xanes_inner_scan():
        yield from abs_set(motor_r.velocity, 30)
        print('\ntake {} dark images...'.format(chunk_size))
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)
        yield from trigger_and_read(list(detectors) + [motor, motor_x, motor_y, motor_z, motor_r])


        print('\nopening shutter, and start xanes scan: {} images per each energy... '.format(chunk_size))
        yield from abs_set(shutter_open, 1, wait=True)
        time.sleep(2)
        yield from abs_set(shutter_open, 1, wait=True)
        time.sleep(1)
        for eng in eng_list:
#            yield from mv_stage(motor, eng)
            yield from move_zp_ccd(eng, move_flag=1, info_flag=0)
            yield from bps.sleep(0.1)
            yield from trigger_and_read(list(detectors) + [motor, motor_x, motor_y, motor_z, motor_r])
#            yield from bps.sleep(0.1)
            yield from mv(motor_r, motor_r_out)
            yield from mv(motor_x, motor_x_out, motor_y, motor_y_out, motor_z, motor_z_out)
            yield from mv(motor_r, motor_r_out)
            yield from mv(motor_x, motor_x_out, motor_y, motor_y_out, motor_z, motor_z_out)
#            yield from mv_stage(motor_x, motor_x_out)
#            yield from mv_stage(motor_y, motor_y_out)
#            yield from mv_stage(motor_z, motor_z_out)
#            yield from mv_stage(motor_r, motor_r_out)
            yield from bps.sleep(0.1)
            yield from trigger_and_read(list(detectors) + [motor, motor_x, motor_y, motor_z, motor_r])
#            yield from bps.sleep(0.1)
            yield from mv(motor_x, motor_x_ini, motor_y, motor_y_ini, motor_z, motor_z_ini)
            yield from mv(motor_r, motor_r_ini)
            yield from mv(motor_x, motor_x_ini, motor_y, motor_y_ini, motor_z, motor_z_ini)
            yield from mv(motor_r, motor_r_ini)
#            yield from mv_stage(motor_r, motor_r_ini)
#            yield from mv_stage(motor_x, motor_x_ini)
#            yield from mv_stage(motor_y, motor_y_ini)
#            yield from mv_stage(motor_z, motor_z_ini)

        yield from mv(motor_r, motor_r_ini)
        yield from mv(motor_x, motor_x_ini, motor_y, motor_y_ini, motor_z, motor_z_ini)
#        yield from mv_stage(motor_x, motor_x_ini) # move sample stage back to orginal position
#        yield from mv_stage(motor_y, motor_y_ini)
#        yield from mv_stage(motor_z, motor_z_ini)

        yield from move_zp_ccd(eng_ini, move_flag=1, info_flag=0)

        print('closing shutter')
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)
        yield from abs_set(shutter_close, 1, wait=True)
        yield from abs_set(motor_r.velocity, rs_ini)
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


#    yield from mv(Andor.cam.num_images, 1)
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



def fly_scan(exposure_time=0.1, relative_rot_angle = 180, period=0.15, chunk_size=20, out_x=0, out_y=2000, out_z=0, rs=1, parkpos=None, note='', md=None):
    motor_rot = zps.pi_r
    motor_x = zps.sx
    motor_y = zps.sy
    motor_z = zps.sz
    motor_x_ini = motor_x.position
    motor_x_out = motor_x_ini + out_x
    motor_y_ini = motor_y.position
    motor_y_out = motor_y_ini + out_y
    motor_z_ini = motor_z.position
    motor_z_out = motor_z_ini + out_z
    detectors = [Andor, ic3]
#    detectors = [detA1, ic3]
    offset_angle = -2.0 * rs
    current_rot_angle = motor_rot.position
    if parkpos == None:
        parkpos = current_rot_angle
    target_rot_angle = current_rot_angle + relative_rot_angle
#    assert (exposure_time <= 0.2), "Exposure time is too long, not suitable to run fly-scan. \nScan aborted."
    _md = {'detectors': ['Andor'],
           'motors': [motor_rot.name],
           'XEng': XEng.position,
           'ion_chamber': ic3.name,
           'plan_args': {'exposure_time': exposure_time,
                         'relative_rot_angle': relative_rot_angle,
                         'period': period,
                         'chunk_size': chunk_size,
                         'out_x': out_x,
                         'out_y': out_y,
                         'rs': rs,
                         'parkpos': parkpos,
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
#            'detectors': list(map(repr, detectors)),
            }
    _md.update(md or {})
    try:  dimensions = [(motor_rot.hints['fields'], 'primary')]
    except (AttributeError, KeyError):    pass
    else: _md['hints'].setdefault('dimensions', dimensions)
    yield from abs_set(motor_rot.velocity, rs)
    print('set rotation speed: {} deg/sec'.format(rs))
    yield from mv(detectors[0].cam.acquire, 0)
    yield from mv(detectors[0].cam.image_mode, 0)
    yield from mv(detectors[0].cam.num_images, chunk_size)
    yield from abs_set(detectors[0].cam.acquire_time, exposure_time)
#    yield from abs_set(detectors[0].cam.acquire_period, exposure_time)
    detectors[0].cam.acquire_period.put(period)
#    yield from abs_set(detectors[0].cam.acquire_period, exposure_time, wait=True)
    @stage_decorator(list(detectors) + [motor_rot])
    @bpp.monitor_during_decorator([motor_rot])
    @run_decorator(md=_md)
    def fly_inner_scan():
        #close shutter, dark images: numer=chunk_size (e.g.20)
        print('\nshutter closed, taking dark images...')
        yield from abs_set(shutter_close, 1)
        yield from sleep(1)
        yield from abs_set(shutter_close, 1)
        yield from sleep(2)
        yield from trigger_and_read(list(detectors) + [motor_rot])
        #open shutter, tomo_images
        yield from abs_set(shutter_open, 1)
        yield from sleep(2)
        yield from abs_set(shutter_open, 1)
        yield from sleep(1)
        print ('\nshutter opened, taking tomo images...')
        yield from mv(motor_rot, current_rot_angle + offset_angle)
        status = yield from abs_set(motor_rot, target_rot_angle, wait=False)
        yield from sleep(2)
        while not status.done:
            yield from trigger_and_read(list(detectors) + [motor_rot])
        #move out sample, taking bkg images
        yield from abs_set(motor_rot.velocity, 30)
        yield from mv(motor_rot, parkpos) # move sample stage back to parkposition to take bkg image
        yield from mv(motor_x, motor_x_out)    # move zps.sx stage to motor_x_out
        yield from mv(motor_y, motor_y_out)
        yield from mv(motor_z, motor_z_out)
        print ('\nTaking background images...')
        #print('pi_x position: {0}'\n.format(motor_x.position))
        yield from trigger_and_read(list(detectors) + [motor_rot])
        #move sample in
        yield from abs_set(shutter_close, 1)
        time.sleep(2)
        yield from abs_set(shutter_close, 1)
        yield from mv(motor_x, motor_x_ini)   # move zps.sx stage back to motor_x_start
        yield from mv(motor_y, motor_y_ini)
        yield from mv(motor_z, motor_z_ini)
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






def overnight_scan():
    for i in range(14):
        print('current run scan #{:2d}'.format(i))
        RE(eng_scan(8930, 9030, 200, 1))
        RE(eng_scan(8930, 9030, 200, 1))
        RE(eng_scan(8930, 9030, 200, 1))
        time.sleep(3600)



###############################################

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


def delay_scan(detectors, motor, start, stop, steps,  sleep_time=1.0, plot_flag=0, note='', md=None):
    '''
    add sleep_time to regular 'scan' for each scan_step
    '''
    if Andor in detectors:
        exposure_time = Andor.cam.acquire_time.value
        yield from mv(Andor.cam.acquire, 0)
        yield from mv(Andor.cam.image_mode, 0)
        yield from mv(Andor.cam.num_images, 1)
        Andor.cam.acquire_period.put(exposure_time)
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
#            yield from abs_set(motor,x)
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



def raster_2D_scan(x_range=[-1,1],y_range=[-1,1],exposure_time=0.1, out_x=0, out_y=0, out_z=0, out_r=0, img_sizeX=2560,img_sizeY=2160,pxl=17.2, note='', md=None):

    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]
    detectors = [Andor, ic3]
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.image_mode, 0)
    yield from mv(Andor.cam.num_images, 1)
    yield from mv(detectors[0].cam.acquire_time, exposure_time)
    detectors[0].cam.acquire_period.put(exposure_time)

    x_initial = zps.sx.position    
    y_initial = zps.sy.position
    z_initial = zps.sz.position
    r_initial = zps.pi_r.position

    img_sizeX = np.int(img_sizeX)
    img_sizeY = np.int(img_sizeY)
    x_range = np.int_(x_range)
    y_range = np.int_(y_range)
    
    
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
    
    @stage_decorator(list(detectors) + list(motor))
    @run_decorator(md=_md)
    def raster_2D_inner():
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)
        print('take 5 dark image')
        for i in range(5):
            yield from trigger_and_read(list(detectors) + list(motor))


        print('opening shutter ...')
        yield from abs_set(shutter_open, 1, wait=True)
        time.sleep(1)
        yield from abs_set(shutter_open, 1, wait=True)
        time.sleep(1)
        print('taking mosaic image ...')
        for ii in range(x_range[0], x_range[1]):
            yield from mv(zps.sx, x_initial + ii*img_sizeX*pxl*1.0/1000)
            for jj in range(y_range[0], y_range[1]):
                yield from mv(zps.sy, y_initial + jj*img_sizeY*pxl*1.0/1000)
                yield from trigger_and_read(list(detectors) + list(motor))


        print('moving sample out to take 5 background image')
        yield from mv(zps.pi_r, out_r)
        yield from mv(zps.sx, x_initial+out_x, zps.sy, y_initial+out_y, zps.sz, z_initial+out_z)
        for i in range(5):
            yield from trigger_and_read(list(detectors) + list(motor))


        yield from mv(zps.sx, x_initial, zps.sy, y_initial, zps.sz, z_initial)
        yield from mv(zps.pi_r, r_initial)
        print('closing shutter')
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)


    yield from raster_2d_inner()
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)
















