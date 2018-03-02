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



def tomo_scan(start, stop, num, exposure_time=0.1, detectors=[detA1], bkg_num=10, dark_num=10, out_pos=1, note='', md=None):
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
    bkg_num: number of background image to be taken
    dark_num: number of dark image to be taken
    out_pos: position of pi_x stage where sample is out (absolute position: mm)
    md: metadate (default: None)
    '''
    yield from abs_set(detectors[0].cam.acquire_time, exposure_time)
    
    motor_x = phase_ring.x
    # motor_x = zps.pi_x
    motor_x_ini = motor_x.position # initial position of motor_x
    motor_x_out = out_pos  # 'out position' of motor_x
        
    # motor_rot = zps.pi_r
    motor_rot = zps.sx
    motor_rot_ini = motor_rot.position  # initial position of motor_x
    
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor_rot.name],
           'x_ray_energy': XEng.position,
           'num_angles': num,
           'num_bkg_images': bkg_num,
           'num_dark_images': dark_num,
           'plan_args': {'detectors': list(map(repr, detectors)), 'num': num,
                         'motor': repr(motor_rot),
                         'start': start, 'stop': stop, 'exposure_time': exposure_time},
           'plan_name': 'tomo_scan',
           'plan_pattern': 'linspace',
           'plan_pattern_module': 'numpy',
           'hints': {},
           'note': note,
           'operator': 'FXI'
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
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(2)
        for num in range(dark_num):   # close the shutter, and take 10(default) dark image when stage is at out position
            yield from trigger_and_read(list(detectors) + [motor_rot])        

        
        # Open shutter, tomo images  
#        yield from abs_set(shutter_open, 1, wait=True)
#        yield from abs_set(shutter_open, 1, wait=True)
        time.sleep(2)
        print ('shutter opened, pi_x position: {0}\n\nstarting tomo_scan...'.format(motor_x.position))

        for step in steps:  # take tomography images
            yield from one_1d_step(detectors, motor_rot, step)
#        yield from mv_stage(motor_rot, motor_rot_ini)
        yield from mov(motor_rot, motor_rot_ini)

        # move out sample, bkg image
#        yield from mv_stage(motor_x, motor_x_out)    # move pi_x stage to motor_x_out
        print ('\n\nTaking background images...\npi_x position: {0}'.format(motor_x.position))
        for num in range(bkg_num):    # take 10 background image when stage is at out position
            yield from trigger_and_read(list(detectors) + [motor_rot])

        # close shutter, move in the sample
#        yield from mov(motor_x, motor_x_ini)   # move pi_x stage back to motor_x_start
#        yield from abs_set(shutter_close, 1, wait=True)
#        yield from abs_set(shutter_close, 1, wait=True)
    return (yield from tomo_inner_scan())





def xanes_scan(eng_list, exposure_time=0.1, detectors=[detA1], bkg_num=10, dark_num=10, out_pos=1, md=None):
    yield from mv(detectors[0].cam.acquire_time, exposure_time)
    motor = XEng
    motor_x = zps.pi_x
    motor_x_ini = motor_x.position    
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'x_ray_energy': motor.position,
           'num_eng': len(eng_list),
           'num_bkg_images': bkg_num,
           'num_dark_images': dark_num,
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motor': repr(motor),
                         'exposure_time': exposure_time},
           'plan_name': 'xanes_scan',
           'hints': {},
           'operator': 'FXI'
            }
    _md.update(md or {})    
    try:   dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):  pass
    else:   _md['hints'].setdefault('dimensions', dimensions)
    
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

        print('\ntake dark image after xanes scan, {} per each energy...'.format(dark_num))
        for i in range(dark_num):
            yield from trigger_and_read(list(detectors) + [motor])

        yield from mv(motor_x, motor_x_ini) # move sample stage back to orginal position
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




def fly_scan(exposure_time=0.1, relative_rot_angle = 1000, chunk_size=20, note='', md=None):
    # motor_rot = zps.pi_r
    motor_rot = zps.sx   
    # motor_x = zps.pi_x
    #motor_x = phase_ring.x
    detectors = [Andor]
    current_rot_angle = motor_rot.position
    target_rot_angle = current_rot_angle + relative_rot_angle 

    assert (exposure_time <= 0.2), "Exposure time is too long, not suitable to run fly-scan. \nScan aborted."   


    _md = {'detectors': ['Andor'],
           'motors': [motor_rot.name],
           'x_ray_energy': XEng.position,
           'plan_args': {'detectors': list(map(repr, detectors)), 
                         'motor': repr(motor_rot),
                         'exposure_time': exposure_time},
           'plan_name': 'fly_scan',
           'num_bkg_images': chunk_size,
           'num_dark_images': chunk_size,
           'chunk_size': chunk_size,
           'plan_pattern': 'linspace',
           'plan_pattern_module': 'numpy',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos': wh_pos(),
            }
    _md.update(md or {})

    try:  dimensions = [(motor_rot.hints['fields'], 'primary')]
    except (AttributeError, KeyError):    pass
    else: _md['hints'].setdefault('dimensions', dimensions)

    yield from mv(Andor.cam.num_images, chunk_size)
    yield from abs_set(detectors[0].cam.acquire_time, exposure_time, wait=True) 

    @stage_decorator(list(detectors) + [motor_rot])
    @bpp.monitor_during_decorator([motor_rot])
    @run_decorator(md=_md)
    def fly_inner_scan(): 

        #close shutter, dark images: numer=chunk_size (e.g.20)
        print('\nshutter closed, taking dark images...')
        # yield from abs_set(shutter_close, 1, wait=True)
        # yield from abs_set(shutter_close, 1, wait=True)        
        yield from trigger_and_read(list(detectors)) 
        
        #open shutter, tomo_images   
        # yield from abs_set(shutter_open, 1, wait=True)
        # yield from abs_set(shutter_open, 1, wait=True)
        print ('\nshutter opened, taking tomo images...')
        status = yield from abs_set(motor_rot, target_rot_angle, wait=False)
        while not status.done:
            yield from trigger_and_read(list(detectors))   

        #move out sample, taking bkg images
        # yield from mv_stage(motor_x, motor_x_out)    # move pi_x stage to motor_x_out
        print ('\nTaking background images...')
        #print('pi_x position: {0}'.format(motor_x.position))
        yield from trigger_and_read(list(detectors))


        #move sample in       
        # yield from mv_stage(motor_x, motor_x_ini)   # move pi_x stage back to motor_x_start
        # yield from abs_set(shutter_close, 1, wait=True)
        # yield from abs_set(shutter_close, 1, wait=True)  

    uid = yield from fly_inner_scan()
    print('scan finished')
    return uid



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
        col_x = col_x_prefix + ' #' + '{:2.1f}'.format(num)
        col_y = col_y_prefix + ' #' + '{:2.1f}'.format(num)
        df[col_x] = pd.Series(motor_readout)
        df[col_y] = pd.Series(signal_readout)
        time.sleep(1)
        plt.show()

    fig = plt.figure()
    for num in range(repeat_num):
        col_x = col_x_prefix + ' #' + '{:2.1f}'.format(num)
        col_y = col_y_prefix + ' #' + '{:2.1f}'.format(num)
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
    engs = np.arange(7, 13.5, 0.5)
    for eng in engs:
        RE(abs_set(XEng, eng))
        time.sleep(30)
        current_eng = XEng.position
        print('current X-ray Energy: {:2.1f}keV'.format(current_eng))
        pzt_scan_multiple(pzt_dcm_th2, -18, 18, 73, dcm.th2, Vout2, repeat_num=1)
    
    RE(abs_set(XEng, 9))
    time.sleep(300)
    current_eng = XEng.position
    print('current X-ray Energy: {:2.1f}keV'.format(current_eng))
    print('run 10 times each 1-hour')
    for i in range(16):
        pzt_scan_multiple(pzt_dcm_th2, -18, 18, 73, dcm.th2, Vout2, repeat_num=10)
        time.sleep(3600)
    
    shutter_close.put(1)    
        



