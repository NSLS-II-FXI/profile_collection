import uuid
import sys
import time
from functools import wraps
from warnings import warn
from bluesky.plans import mv, mvr

import numpy as np
try:
    from cytools import partition
except ImportError:
    from toolz import partition
from bluesky import plan_patterns
from bluesky.utils import (Msg, short_uid as _short_uid, make_decorator)


def timeit(method):    
    def timed(*args, **kw):
        ts = time.time()
        results = method(*args, **kw)
        te = time.time()        
        print ('Time for {0:s}: {1:2.2f}'.format(method.__name__, te-ts))
        return results
    return timed



def tomo_scan(start, stop, num, exposure_time=0.1, detectors=[detA1], bkg_num=10, md=None):
    '''
    Script for running Tomography scan
    Use as: RE(tomo_scan(start, stop, num, exposure_time=0.1, detectors=[detA1], bkg_num=10, md=None))

    Input:
    ------
    start: start angle
    stop: stop angle
    num: number of scan angles
    exposure time: second (default: 0.1)
    detectors: camera to capture image (default: detA1)
    md: metadate (default: None)
    '''
    abs_set(detectors[0].cam.acquire_time, exposure_time)
    
    motor_x = zps.pi_x
    motor_x_ini = motor_x.position # initial position of motor_x
    motor_x_out = 1  # 'out position' of motor_x
        
    motor_rot = zps.pi_r
    motor_rot_ini = motor_rot.position  # initial position of motor_x
    
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor_rot.name],
           'x_ray_energy': XEng.position,
           'num_points': num,
           'num_intervals': num - 1,
           'back_ground_images': bkg_num,
           'plan_args': {'detectors': list(map(repr, detectors)), 'num': num,
                         'motor': repr(motor_rot),
                         'start': start, 'stop': stop, 'exposure_time': exposure_time},
           'plan_name': 'tomo_scan',
           'plan_pattern': 'linspace',
           'plan_pattern_module': 'numpy',
           'other_motor_position': {'zp_x': zp.x.position, 'zp_y': zp.y.position, 'zp_z': zp.z.position,
                                    'aper_x': aper.x.position, 'aper_y': aper.y.position, 
                                    'aper_z': aper.z.position , 'cond_x': clens.x.position,
                                    'cond_y1': clens.y1.position, 'cond_y2': clens.y2.position, 
                                    'cond_z1': clens.z1.position, 'cond_z2': clens.z2.position, 
                                    'cond_pitch': clens.p.position,
                                    'phase_ring_x': phase_ring.x.position, 'phase_ring_y': phase_ring.y.position,
                                    'phase_ring_z': phase_ring.z.position, 'betr_x': betr.x.position,
                                    'betr_y': betr.y.position, 'betr_z': betr.z.position,
                                    'sample_x': zps.sx.position, 'sample_y': zps.sy.position,'sample_z': zps.sz.position, 
                                    'sample_pi_x': zps.pi_x.position, 'sample_pi_rotate': zps.pi_r.position,
                                    'DetU_x': DetU.x.position, 'DetU_y': DetU.y.position, 'DetU_z': DetU.z.position,
                                    'dcm_eng': XEng.position, 'dcm_th1': dcm.th1.position, 'dcm_dy2': dcm.dy2.position,
                                    'dcm_chi2': dcm.chi2.position},
           'hints': {},
           'operator': 'FXI'
            }
    _md.update(md or {})

    try:
        dimensions = [(motor_rot.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)

    steps = np.linspace(start, stop, num)
    
    @stage_decorator(list(detectors) + [motor_rot, motor_x])
    @run_decorator(md=_md)

    def tomo_inner_scan():

        # move pi_x stage to motor_x_out
        yield from mv_stage(motor_x, motor_x_out)
        print (('\n\nTaking background images...\npi_x position: {0}').format(motor_x.position))
        
        # take 10 background image when stage is at out position
        for num in range(10):
            yield from trigger_and_read(list(detectors))

        # move pi_x stage back to motor_x_start
        yield from mv_stage(motor_x, motor_x_ini)
        print (('\npi_x position: {0}\n\nstarting tomo_scan...').format(motor_x.position))

        # take tomography images
        for step in steps:
            yield from one_1d_step(detectors, motor_rot, step)

        yield from mv_stage(motor_rot, motor_rot_ini)

    yield from tomo_inner_scan()

def mv_stage(motor, pos):
        grp = _short_uid('set')
        yield Msg('checkpoint')
        yield Msg('set', motor, pos, group=grp)
        yield Msg('wait', None, group=grp)


def eng_scan(eng_start, eng_end, steps, dwell_time=1.):
    '''
    eng_start, eng_end are in unit of eV !!
    '''
    check_eng_range(eng_start, eng_end)
    set_ic_dwell_time(dwell_time=dwell_time)
    yield from scan([ic1, ic2], XEng, eng_start/1000, eng_end/1000, steps)
    h = db[-1]
    y0 = np.array(list(h.data(ic1.name)))
    y1 = np.array(list(h.data(ic2.name)))
    
    r = y0/y1
    x = np.linspace(eng_start, eng_end, steps)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.title.set_text('ratio of: {0}/{1}'.format(ic1.name, ic2.name))    
    ax1.plot(x, r, '.-')

    ax2 = fig.add_subplot(212)         
    ax2.title.set_text('differential of: {0}/{1}'.format(ic1.name, ic2.name))
    r_dif = np.array([0] + list(np.diff(r)))
    ax2.plot(x, r_dif, '.-')
    fig.subplots_adjust(hspace=.5)
    plt.show()





