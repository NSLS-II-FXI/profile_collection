import uuid
import sys
import time
from datetime import datetime
from functools import wraps
from warnings import warn
from bluesky.plan_stubs import mv, mvr
import subprocess
import pandas as pd

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



def tomo_scan(start, stop, num, exposure_time=0.1, detectors=[detA1], bkg_num=10, out_pos=1, md=None):
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
    yield from abs_set(detectors[0].cam.acquire_time, exposure_time)
    
    motor_x = zps.pi_x
    motor_x_ini = motor_x.position # initial position of motor_x
    motor_x_out = out_pos  # 'out position' of motor_x
        
    motor_rot = zps.pi_r
    motor_rot_ini = motor_rot.position  # initial position of motor_x

    print('test')    
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor_rot.name],
           'x_ray_energy': XEng.position,
           'num_angles': num,
           'num_bkg_images': bkg_num,
           'num_dk_images': 0,
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
        yield from mv_stage(motor_x, motor_x_out)    # move pi_x stage to motor_x_out
        print (('\n\nTaking background images...\npi_x position: {0}').format(motor_x.position))
        
        for num in range(bkg_num):    # take 10 background image when stage is at out position
            yield from trigger_and_read(list(detectors) + [motor_rot])
        
        yield from mv_stage(motor_x, motor_x_ini)   # move pi_x stage back to motor_x_start
        print (('\npi_x position: {0}\n\nstarting tomo_scan...').format(motor_x.position))
     
        for step in steps:  # take tomography images
            yield from one_1d_step(detectors, motor_rot, step)

        yield from mv_stage(motor_rot, motor_rot_ini)

    return (yield from tomo_inner_scan())


def xanes_scan(eng_list, exposure_time=0.1, detectors=[detA1], bkg_num=10, out_pos=1, md=None):

    yield from mv(detectors[0].cam.acquire_time, exposure_time)
    motor = XEng
    motor_x = zps.pi_x
    motor_x_ini = motor_x.position
    
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'x_ray_energy': motor.position,
           'num_eng': len(eng_list),
           'num_bkg_images': bkg_num,
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motor': repr(motor),
                         'exposure_time': exposure_time},
           'plan_name': 'xanes_scan',
           'hints': {},
           'operator': 'FXI'
            }
    _md.update(md or {})
    
    try:
        dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)
    
    @stage_decorator(list(detectors) + [motor, motor_x])
    @run_decorator(md=_md)

    def xanes_inner_scan():
        print('\nstart xanes scan ... ')
        
        for eng in eng_list:
            yield from mv_stage(motor, eng)
            yield from trigger_and_read(list(detectors) + [motor])
        
        yield from mv(motor_x, out_pos)
        print('\ntake bkg image after xanes scan, {} per each energy...'.format(bkg_num))
        for eng in eng_list:
            for i in range(bkg_num):
                yield from mv_stage(motor, eng)
                yield from trigger_and_read(list(detectors) + [motor])
        yield from mv(motor_x, motor_x_ini)
    return (yield from xanes_inner_scan())


def mv_stage(motor, pos):
        grp = _short_uid('set')
        yield Msg('checkpoint')
        yield Msg('set', motor, pos, group=grp)
        yield Msg('wait', None, group=grp)





def eng_scan(eng_start, eng_end, steps, dwell_time=1.):
    '''
    eng_start, eng_end are in unit of eV !!
    '''
    check_eng_range([eng_start, eng_end])
    set_ic_dwell_time(dwell_time=dwell_time)
    yield from scan([ic3, ic4], XEng, eng_start/1000, eng_end/1000, steps)
    h = db[-1]
    y0 = np.array(list(h.data(ic3.name)))
    y1 = np.array(list(h.data(ic4.name)))
    
    r = y0/y1
    x = np.linspace(eng_start, eng_end, steps)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.title.set_text('ratio of: {0}/{1}'.format(ic4.name, ic3.name))    
    ax1.plot(x, r, '.-')

    ax2 = fig.add_subplot(212)         
    ax2.title.set_text('differential of: {0}/{1}'.format(ic4.name, ic3.name))
    r_dif = np.array([0] + list(np.diff(r)))
    ax2.plot(x, r_dif, '.-')
    fig.subplots_adjust(hspace=.5)
    plt.show()




def overnight_scan():
    for i in range(14):
        print('current run scan #{:2d}'.format(i))
        RE(eng_scan(8930, 9030, 200, 1))
        RE(eng_scan(8930, 9030, 200, 1))
        RE(eng_scan(8930, 9030, 200, 1))
        time.sleep(3600)

##################
def cond_scan(detectors=[detA1], *, md=None):
    motor = clens.x
    
    _md = {'detectors': [det.name for det in detectors],
           'motors': [clens.x.name],
 
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motor': repr(motor),
                         },
           'plan_name': 'cond_scan',
           'hints': {},
           'operator': 'FXI'
            }
    _md.update(md or {})
    
    try:
        dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):
        pass
    else:
        _md['hints'].setdefault('dimensions', dimensions)   
    
    @stage_decorator(list(detectors))
    @run_decorator(md=_md)


    def cond_inner_scan():
        for x in range(6000, 7100, 100):
            for z1 in range(-800, 800, 20):
                for p in range(-600, 600, 10):
                    yield from mv_stage(clens.x, x)
                    yield from mv_stage(clens.z1, z1)
                    yield from mv_stage(clens.p, p)
                    yield from trigger_and_read(list(detectors))  
    return (yield from cond_inner_scan())


###############################################
def delay_scan(start, stop, steps, detectors=[Vout2], md=None):
    motor = dcm.th2
    
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
 
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motor': repr(motor),
                         'num': steps,
                         'start': start,
                         'stop': stop,
                         },
           'plan_name': 'delay_scan',
           'hints': {},
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
            time.sleep(1)
            yield from trigger_and_read(list(detectors))  
    return (yield from delay_inner_scan())

################  PZT  ##########################

def pzt_scan(moving_pzt, start, stop, steps, read_back_dev, record_dev, delay_time=5, print_flag=1, overlay_flag=0):
    '''
    Input:
    -------
    moving_pzt: pv name of the pzt device, e.g. 'XF:18IDA-OP{Mir:DCM-Ax:Th2Fine}SET_POSITION.A'

    read_back_dev: device (encoder) that changes with moving_pzt, e.g., dcm.th2

    record_dev: signal you want to record, e.g. Vout2     

    delay_time: waiting time for device to response
    '''

    current_pos = moving_pzt.pos

    my_set_cmd = 'caput ' + moving_pzt.setting_pv + ' ' + str(start)
    subprocess.Popen(my_set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(delay_time)
     
    my_var = np.linspace(start, stop, steps)
    pzt_readout = []
    motor_readout = []
    signal_readout = []
    for x in my_var:        
        my_set_cmd = 'caput ' + moving_pzt.setting_pv + ' ' + str(x) 
        subprocess.Popen(my_set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
        time_start = time.time()        
        while True:
            pos = subprocess.check_output(['caget', pzt_dcm_th2.getting_pv, '-t']).rstrip()
            pos = np.float(str_convert(pos))
            if np.abs(pos-x) < 1e-2 or (time.time()-time_start > delay_time):
                break
        time.sleep(1)

        pzt_readout.append(pos)
        y = read_back_dev.position
        motor_readout.append(y)
        z = record_dev.value
        signal_readout.append(z)

        prt1 = moving_pzt.name + ': {:2.4f}'.format(x)
        prt2 = 'pzt read_back: {:2.4f}'.format(pos)  
        prt3 = read_back_dev.name + ': {:2.4f}'.format(y)
        prt4 = record_dev.name + ': {:2.4f}'.format(z)        
        print(prt1 + '   ' + prt2 + '   ' + prt3 + '   ' + prt4)

    my_set_cmd = 'caput ' + moving_pzt.setting_pv + ' ' + str(current_pos)
    subprocess.Popen(my_set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    print('\nmoving {0} back to original position: {1:2.4f}'.format(moving_pzt.name, current_pos)) 

    if not overlay_flag:
        plt.figure()
    if print_flag:    
         
        plt.subplot(221);plt.plot(my_var, pzt_readout, '.-')
        plt.xlabel(moving_pzt.name+' set_point');plt.ylabel(read_back_dev.name+ ' read_out')

        plt.subplot(222); plt.plot(pzt_readout, motor_readout, '.-')
        plt.xlabel(moving_pzt.name);plt.ylabel(read_back_dev.name)
    
        plt.subplot(223); plt.plot(pzt_readout, signal_readout, '.-')
        plt.xlabel(moving_pzt.name);plt.ylabel(record_dev.name)
    
        plt.subplot(224); plt.plot(motor_readout, signal_readout, '.-')
        plt.xlabel(read_back_dev.name);plt.ylabel(record_dev.name)
        
    return pzt_readout, motor_readout, signal_readout


def pzt_scan_multiple(moving_pzt, start, stop, steps, read_back_dev, record_dev, repeat_num=10, delay_time=5, save_file_dir='/home/xf18id/Documents/FXI_commision/DCM_scan/'):
    
    current_eng = XEng.position
        
    df = pd.DataFrame(data = [])
    col_x_prefix = read_back_dev.name
    col_y_prefix = record_dev.name
    xx, yy, zz = [], [], []
    for num in range(repeat_num):
        print('\nscan #' + str(num))
        pzt_readout, motor_readout, signal_readout = pzt_scan(moving_pzt, start, stop, steps, read_back_dev, record_dev, delay_time=delay_time, overlay_flag=1)
        col_x = col_x_prefix + ' #' + str(num)
        col_y = col_y_prefix + ' #' + str(num)
        df[col_x] = pd.Series(motor_readout)
        df[col_y] = pd.Series(signal_readout)
        time.sleep(1)
        plt.show()

    fig = plt.figure()
    for num in range(repeat_num):
        col_x = col_x_prefix + ' #' + str(num)
        col_y = col_y_prefix + ' #' + str(num)
        x, y = df[col_x], df[col_y]
        plt.plot(x, y)
    plt.title('X-ray Energy: {:2.1f}keV'.format(current_eng))

    now = datetime.now()
    year = np.str(now.year)
    mon  = '{:02d}'.format(now.month)
    day  = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minu = '{:02d}'.format(now.minute)
    current_date = year + '-' + mon + '-' + day
    fn = save_file_dir + 'pzt_scan_' + '{:2.1f}keV'.format(current_eng) + current_date + '_' + hour + '-' + minu
    fn_fig = fn + '.tiff'
    fn_file = fn + '.csv' 
    df.to_csv(fn_file, sep = '\t')
    print('save to: ' + fn_file)

    fig.savefig(fn_fig)
    return

#########
def pzt_overnigh_scan():
    shutter_open.put(1)
    time.sleep(5)
    print('shutter open')
    engs = np.arange(5, 13, 0.5)
    for eng in engs:
        RE(abs_set(XEng, eng))
        time.sleep(60)
        current_eng = XEng.position
        print('current X-ray Energy: {:2.1f}keV'.format(current_eng))
        pzt_scan_multiple(pzt_dcm_th2, -18, 18, 73, dcm.th2, Vout2, repeat_num=1)

    RE(abs_set(XEng, 9))
    time.sleep(300)
    current_eng = XEng.position
    print('current X-ray Energy: {:2.1f}keV'.format(current_eng))
    print('run 10 times each 1-hour')
    for i in range(12):
        pzt_scan_multiple(pzt_dcm_th2, -18, 18, 73, dcm.th2, Vout2, repeat_num=10)

    shutter_close.put(1)    
        



