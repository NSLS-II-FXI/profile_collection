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
           'plan_args': {'detectors': list(map(repr, detectors)), 'num': num,
                         'motor': repr(motor_rot),
                         'start': start, 'stop': stop, 'exposure_time': exposure_time},
           'plan_name': 'tomo_scan',
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
    return (yield from tomo_inner_scan())





def xanes_scan(eng_list, exposure_time=0.1, chunk_size=5, out_x=0, out_y=0, note='', md=None):
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
 
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor.name],
           'x_ray_energy': motor.position,
           'num_eng': len(eng_list),
           'num_bkg_images': chunk_size,
           'num_dark_images': chunk_size,
           'chunk_size': chunk_size,
           'plan_args': {'detectors': list(map(repr, detectors)),
                         'motor': repr(motor),
                         'exposure_time': exposure_time},
           'plan_name': 'xanes_scan',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos': wh_pos(),
            }
    _md.update(md or {})    
    try:   dimensions = [(motor.hints['fields'], 'primary')]
    except (AttributeError, KeyError):  pass
    else:   _md['hints'].setdefault('dimensions', dimensions)
    
    @stage_decorator(list(detectors) + [motor, motor_x])
    @run_decorator(md=_md)
    def xanes_inner_scan(): 
        print('\ntake {} dark images...'.format(chunk_size))
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)        
        yield from abs_set(shutter_close, 1, wait=True) 
        time.sleep(1)
        yield from trigger_and_read(list(detectors) + [motor])
               
               
        print('\nopening shutter, and start xanes scan: {} images per each energy... '.format(chunk_size))
        yield from abs_set(shutter_open, 1, wait=True)
        time.sleep(2)        
        yield from abs_set(shutter_open, 1, wait=True) 
        time.sleep(1)
        for eng in eng_list:
#            yield from mv_stage(motor, eng)
            yield from move_zp_ccd(eng*1000, info_flag=0)
            yield from trigger_and_read(list(detectors) + [motor])
        
        yield from mv_stage(motor_x, motor_x_out)
        yield from mv_stage(motor_y, motor_y_out)
        print('\ntake bkg image after xanes scan, {} per each energy...'.format(chunk_size))
        for eng in eng_list:
#            yield from mv_stage(motor, eng)
            yield from move_zp_ccd(eng*1000, info_flag=0)
            yield from trigger_and_read(list(detectors) + [motor])
#            for i in range(bkg_num):
#                yield from trigger_and_read(list(detectors) + [motor])

        yield from mv_stage(motor_x, motor_x_ini) # move sample stage back to orginal position
        yield from mv_stage(motor_y, motor_y_ini)
        yield from move_zp_ccd(eng_ini*1000, info_flag=0)

        print('closing shutter')
        yield from abs_set(shutter_close, 1, wait=True)
        time.sleep(1)        
        yield from abs_set(shutter_close, 1, wait=True) 
    return (yield from xanes_inner_scan())


def mv_stage(motor, pos):
        grp = _short_uid('set')
        yield Msg('checkpoint')
        yield Msg('set', motor, pos, group=grp)
        yield Msg('wait', None, group=grp)


def eng_scan(eng_start, eng_end, steps, num=10, delay_time=1):
    '''
    eng_start, eng_end are in unit of eV !!
    '''
#    yield from mv(Andor.cam.num_images, 1)
    check_eng_range([eng_start, eng_end])
#    set_ic_dwell_time(dwell_time=dwell_time)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)  
    for i in range(num):
  #      yield from scan([ic3, ic4], XEng, eng_start/1000, eng_end/1000, steps)
        yield from eng_scan_delay(eng_start/1000, eng_end/1000, steps, delay_time=delay_time)
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

def eng_scan_delay(start, stop, num, delay_time=1, note='', md=None):

    detectors=[ic3, ic4]
    motor_x = XEng 
    motor_x_ini = motor_x.position # initial position of motor_x
    
    _md = {'detectors': [det.name for det in detectors],
           'motors': [motor_x.name],
           'x_ray_energy': XEng.position,
           'plan_name': 'eng_scan',
           'plan_pattern': 'linspace',
           'plan_pattern_module': 'numpy',
           'hints': {},
           'operator': 'FXI',
           'note': note if note else 'None',
           'motor_pos': wh_pos(),
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
        yield from abs_set(motor_x, motor_x_ini, wait=True)
    return (yield from eng_inner_scan())

def fly_scan(exposure_time=0.1, relative_rot_angle = 180, period=0.15, chunk_size=20, out_x=0, out_y=2000, rs=1, parkpos=None, note='', md=None):
    motor_rot = zps.pi_r
    motor_x = zps.sx
    motor_y = zps.sy
    motor_x_ini = motor_x.position
    motor_x_out = motor_x_ini + out_x
    motor_y_ini = motor_y.position
    motor_y_out = motor_y_ini + out_y
    detectors = [Andor, ic3]

    offset_angle = -2.0 * rs
    
    current_rot_angle = motor_rot.position
    if parkpos == None:
        parkpos = current_rot_angle
    
    target_rot_angle = current_rot_angle + relative_rot_angle 

#    assert (exposure_time <= 0.2), "Exposure time is too long, not suitable to run fly-scan. \nScan aborted."   
    

    _md = {'detectors': ['Andor'],
           'motors': [motor_rot.name],
           'x_ray_energy': XEng.position,
           'ion_chamber': ic3.name,
           'detectors': list(map(repr, detectors)), 
           'plan_args': {'exposure_time': exposure_time,
                         'relative_rot_angle': relative_rot_angle,
                         'chunk_size': chunk_size,
                         'out_x': out_x,
                         'out_y': out_y,
                         'rs': rs,
                         'note': note,
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
           'motor_pos': wh_pos(),
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

        print ('\nTaking background images...')
        #print('pi_x position: {0}'\n.format(motor_x.position))
        yield from trigger_and_read(list(detectors) + [motor_rot])


        #move sample in    
        yield from abs_set(shutter_close, 1)
        time.sleep(2)
        yield from abs_set(shutter_close, 1)    
        
        yield from mv(motor_x, motor_x_ini)   # move zps.sx stage back to motor_x_start
        yield from mv(motor_y, motor_y_ini)        

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
           'operator': 'FXI',
           'motor_pos': wh_pos(),
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
def delay_scan(motor, start, stop, steps, detectors=[Vout2], sleep_time=1.0, plot_flag=0, md=None):
    '''
    add sleep_time to regular 'scan' for each scan_step
    '''

    #motor = dcm.th2
    #motor = pzt_dcm_th2.setpos
    
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
           'motor_pos': wh_pos(),
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
    uid = yield from delay_inner_scan()
    if plot_flag:
        h = db[-1]
        x = np.linspace(start, stop, steps)
        y = list(h.data(detectors[0].name))
        plt.figure();
        plt.plot(x,y);plt.xlabel(motor.name);plt.ylabel(detectors[0].name)
        plt.title('scan# {}'.format(h.start['scan_id']))
    return uid
    
    
################  PZT  ##########################

def pzt_scan(pzt_motor, start, stop, steps, detectors=[Vout2], sleep_time=1, md=None):

    motor = pzt_motor.setpos
    motor_readback = pzt_motor.pos    
    motor_ini_pos = motor_readback.get()

    detector_set_read = [motor, motor_readback]
    detector_all = detectors + detector_set_read

    

    _md = {'detectors': [det.name for det in detectors],

           'detector_set_read': [det.name for det in detector_set_read],
       'motors': [motor.name],
 
       'plan_args': {'detectors': [det.name for det in detectors],
                     'detector_set_read': [det.name for det in detector_set_read],
                     'motor': repr(motor),
                     'num': steps,
                     'start': start,
                     'stop': stop,
                     'sleep_time': sleep_time,
                     },
       'plan_name': 'pzt_scan',
       'hints': {},
       'motor_pos': wh_pos(),
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

    @stage_decorator(list(detector_all))
    @run_decorator(md=_md)
    def pzt_inner_scan():
        for x in my_var:
            yield from mv(motor, x)
            yield from bps.sleep(sleep_time)
            yield from trigger_and_read(list(detector_all))  
        yield from mv(motor, motor_ini_pos)
    uid = yield from pzt_inner_scan()
    return uid



#def pzt_scan(moving_pzt, start, stop, steps, read_back_dev, record_dev, delay_time=5, print_flag=1, overlay_flag=0):
    '''
    Input:
    -------
    moving_pzt: pv name of the pzt device, e.g. 'XF:18IDA-OP{Mir:DCM-Ax:Th2Fine}SET_POSITION.A'

    read_back_dev: device (encoder) that changes with moving_pzt, e.g., dcm.th2

    record_dev: signal you want to record, e.g. Vout2     

    delay_time: waiting time for device to response
    '''


#    current_pos = moving_pzt.pos

#    my_set_cmd = 'caput ' + moving_pzt.setting_pv + ' ' + str(start)
#    subprocess.Popen(my_set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#    time.sleep(delay_time)
     
#    my_var = np.linspace(start, stop, steps)
#    pzt_readout = []
#    motor_readout = []
#    signal_readout = []
#    for x in my_var:        
#        my_set_cmd = 'caput ' + moving_pzt.setting_pv + ' ' + str(x) 
#        subprocess.Popen(my_set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
#        time_start = time.time()        
#        while True:
#            pos = subprocess.check_output(['caget', pzt_dcm_th2.getting_pv, '-t']).rstrip()
#            pos = np.float(str_convert(pos))
#            if np.abs(pos-x) < 1e-2 or (time.time()-time_start > delay_time):
#                break
#        time.sleep(1)

#        pzt_readout.append(pos)
#        y = read_back_dev.position

#        motor_readout.append(y)
#        z = record_dev.value
#        signal_readout.append(z)

#        prt1 = moving_pzt.name + ': {:2.4f}'.format(x)
#        prt2 = 'pzt read_back: {:2.4f}'.format(pos)  
#        prt3 = read_back_dev.name + ': {:2.4f}'.format(y)
#        prt4 = record_dev.name + ': {:2.4f}'.format(z)        

#        print(prt1 + '   ' + prt2 + '   ' + prt3 + '   ' + prt4)

#    my_set_cmd = 'caput ' + moving_pzt.setting_pv + ' ' + str(current_pos)
#    subprocess.Popen(my_set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

#    print('\nmoving {0} back to original position: {1:2.4f}'.format(moving_pzt.name, current_pos)) 

#    if not overlay_flag:
#        plt.figure()
#    if print_flag:    
         
#        plt.subplot(221);plt.plot(my_var, pzt_readout, '.-')
#        plt.xlabel(moving_pzt.name+' set_point');plt.ylabel(read_back_dev.name+ ' read_out')

#        plt.subplot(222); plt.plot(pzt_readout, motor_readout, '.-')
#        plt.xlabel(moving_pzt.name);plt.ylabel(read_back_dev.name)
    
#        plt.subplot(223); plt.plot(pzt_readout, signal_readout, '.-')
#        plt.xlabel(moving_pzt.name);plt.ylabel(record_dev.name)
    
#        plt.subplot(224); plt.plot(motor_readout, signal_readout, '.-')
#        plt.xlabel(read_back_dev.name);plt.ylabel(record_dev.name)
        
#    return pzt_readout, motor_readout, signal_readout


def pzt_scan_multiple(moving_pzt, start, stop, steps, detectors=[Vout2], repeat_num=2, sleep_time=1, save_file_dir='/home/xf18id/Documents/FXI_commision/DCM_scan/'):
    '''
    Repeat scanning the pzt (e.g. pzt_dcm_ch2, pzt_dcm_th2), and read the detector outputs. Images and .csv data file will be saved
    Use as:
    e.g., pzt_scan_multiple(pzt_dcm_th2,-4, -2, 3, [Vout2, Andor,ic1, dcm.th2], repeat_num=2, save_file_dir='/home/xf18id/Documents/FXI_commision/DCM_scan/')

    Inputs:
    ---------

    moving_pzt: pzt device
        e.g., pzt_dcm_th2, pzt_dcm_chi   
    start: float
        start position of pzt stage
    stop: float
        stop position of pzt stage
    steps:  int
        steps of scanning
    detectors: list of detector device or signals
        e.g., [dcm.th2, Andor, ic3, Vout2]
    repeat_num: int
        repeat scanning for "repeat_num" times
    save_file_dir: str
        directory to save files and images    

    '''
    current_eng = XEng.position
    df = pd.DataFrame(data = [])
    
    for num in range(repeat_num):
        yield from pzt_scan(moving_pzt, start, stop, steps, detectors = detectors, sleep_time=sleep_time)
    yield from abs_set(XEng, current_eng, wait=True)
    print('\nscan finished, ploting and saving data...')
    fig = plt.figure()
    for num in reversed(range(repeat_num)):
        h = db[-1-num]
        scan_id = h.start['scan_id']
        detector_set_read = h.start['detector_set_read']
        col_x_prefix = detector_set_read[1]
        col_x = col_x_prefix + ' #' + '{}'.format(scan_id)

        motor_readout = np.array(list(h.data(col_x_prefix)))           
        df[col_x] = pd.Series(motor_readout)

        detector_signal = h.start['detectors']

        for i in range(len(detector_signal)):
            det = detector_signal[i]

            if (det == 'Andor') or (det == 'detA1'):
                det =  det +'_stats1_total'
            det_readout = np.array(list(h.data(det)))
            col_y_prefix = det            
            col_y = col_y_prefix + ' #' + '{}'.format(scan_id)
            df[col_y] = pd.Series(det_readout)
            plt.subplot(len(detector_signal), 1, i+1)
            plt.plot(df[col_x], df[col_y])     
            plt.ylabel(det)

    
    plt.subplot(len(detector_signal),1,len(detector_signal))    
    plt.xlabel(col_x_prefix)
    plt.subplot(len(detector_signal),1,1)
    plt.title('X-ray Energy: {:2.1f}keV'.format(current_eng))
        
    now = datetime.now()
    year = np.str(now.year)
    mon  = '{:02d}'.format(now.month)
    day  = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minu = '{:02d}'.format(now.minute)
    current_date = year + '-' + mon + '-' + day
    fn = save_file_dir + 'pzt_scan_' + '{:2.1f}keV_'.format(current_eng) + current_date + '_' + hour + '-' + minu
    fn_fig = fn + '.tiff'
    fn_file = fn + '.csv' 
    df.to_csv(fn_file, sep = '\t')
    fig.savefig(fn_fig)
    print('save to: ' + fn_file)

    

    
######################

def pzt_energy_scan(moving_pzt, start, stop, steps, eng_list, detectors=[dcm.th2, Vout2], repeat_num=1,sleep_time=1, save_file_dir='/home/xf18id/Documents/FXI_commision/DCM_scan/'):
    '''
    With given energy list, scan the pzt multiple times and record the signal from various detectors, file will be saved to local folder.
    
    Inputs:
    ---------
    moving_pzt: pzt device
        e.g., pzt_dcm_th2, pzt_dcm_chi   
    start: float
        start position of pzt stage
    stop: float
        stop position of pzt stage
    steps:  int
        steps of scanning
    eng_list: list or array(float)
        e.g. [8.9, 9.0]
    detectors: list of detector device or signals
        e.g., [dcm.th2, Andor, ic3, Vout2]
    repeat_num: int
        repeat scanning for "repeat_num" times
    save_file_dir: str
        directory to save files and images    

    '''
    eng_ini = XEng.position
    yield from abs_set(shutter_open, 1)
    yield from bps.sleep(1)
    yield from abs_set(shutter_open, 1)
    print('shutter open')
    for eng in eng_list:
        yield from abs_set(XEng, eng, wait=True)
        current_eng = XEng.position
        yield from bps.sleep(1)
        print('current X-ray Energy: {:2.1f}keV'.format(current_eng))
        yield from pzt_scan_multiple(pzt_dcm_th2, start, stop, steps, detectors, repeat_num=repeat_num, sleep_time=sleep_time, save_file_dir=save_file_dir)
    yield from abs_set(XEng, eng_ini, wait=True)
    yield from abs_set(shutter_close, 1)
    yield from bps.sleep(1)
    yield from abs_set(shutter_close, 1)

def pzt_overnight_scan(moving_pzt, start, stop, steps, detectors=[dcm.th2, Vout2], repeat_num=10, sleep_time=1, night_sleep_time=3600, scan_num=12,  save_file_dir='/home/xf18id/Documents/FXI_commision/DCM_scan/'):
    '''
    At current energy, repeating scan the pzt multiple times and record the signal from various detectors, file will be saved to local folder.
    
    Inputs:
    ---------
    moving_pzt: pzt device
        e.g., pzt_dcm_th2, pzt_dcm_chi   
    start: float
        start position of pzt stage
    stop: float
        stop position of pzt stage
    steps:  int
        steps of scanning
    eng_list: list or array(float)
        e.g. [8.9, 9.0]
    detectors: list of detector device or signals
        e.g., [dcm.th2, Andor, ic3, Vout2]
    repeat_num: int
        single step: repeat scanning for "repeat_num" times
    sleep_time: float
        time interval (seconds) for successive scans, e.g. 3600 seconds
    scan_num: int
        repeat multiple step scanning, e.g. 12 times, at 1 hour interval (set sleep_time=3600)
    save_file_dir: str
        directory to save files and images    

    '''

    eng_ini = XEng.position
    print('current X-ray Energy: {:2.1f}keV'.format(current_def)) 
    print('run {0:d} times at {1:d} seconds interval'.format(repeat_num, scan_num))
    for i in range(scan_num):
        print('scan num: {:d}'.format(i))
        yield from pzt_scan_multiple(pzt_dcm_th2, start, stop, steps, detectors, repeat_num=repeat_num, sleep_time=sleep_time,  save_file_dir=save_file_dir)
        yield from bps.sleep(night_sleep_time)   
        

####### 
def test_scan(out_x=-100, out_y=-100, num=10, num_bd=10, fn='/home/xf18id/zp_30nm_02s_Linear_off_rotary_on_20180402.h5'):
    RE(count([Andor], num))
    img = get_img(db[-1])

    y_ini = zps.sy.position
    y_out = y_ini + out_y
    RE(mv(zps.sy, y_out))

    x_ini = zps.sx.position
    x_out = x_ini + out_x
    RE(mv(zps.sx, x_out))


    RE(count([Andor], num_bd))
    img_bkg = get_img(db[-1])

    RE(abs_set(shutter_close, 1, wait=True))
    time.sleep(2) 
    RE(abs_set(shutter_close, 1, wait=True))
    time.sleep(2)
    RE(count([Andor], num_bd))
    img_dark = get_img(db[-1])

    img_dark_avg = np.mean(img_dark, axis=0)
    img_bkg_avg = np.mean(img_bkg, axis=0)
    RE(mv(zps.sy, y_ini))
    RE(mv(zps.sx, x_ini))
    img_norm = (img - img_dark_avg)/(img_bkg_avg - img_dark_avg)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0
    img_norm_avg = np.mean(img_norm, axis=0).reshape(1, img_norm.shape[1], img_norm.shape[2])
    with h5py.File(fn, 'w') as hf:
        hf.create_dataset('img_norm', data = img_norm)
        hf.create_dataset('img_raw', data = img)
        hf.create_dataset('img_dark_avg', data = img_dark_avg)
        hf.create_dataset('img_bkg_avg', data = img_bkg_avg)
        hf.create_dataset('img_norm_avg', data = img_norm_avg)
    RE(abs_set(shutter_open, 1, wait=True))
    return 0

def z_scan(start=-0.03, end=0.03, step=25, out_y=-100, fn='/home/xf18id/Documents/FXI_commision/star_pattern/30nm_starpattern/z_scan_motor_on_1.h5'):
    z_ini = zp.z.position    
    RE(scan([Andor], zp.z, z_ini+start, z_ini+end, step))
    y_ini = zps.sy.position
    y_out = y_ini + out_y
    img = get_img(db[-1])
    RE(mv(zps.sy, y_out))
    RE(count([Andor], 10))
    img_bkg = get_img(db[-1])
    RE(abs_set(shutter_close, 1, wait=True))
    time.sleep(1)
    RE(abs_set(shutter_close, 1))
    RE(count([Andor], 10))
    img_dark = get_img(db[-1])
    img_dark_avg = np.mean(img_dark, axis=0)
    img_bkg_avg = np.mean(img_bkg, axis=0)
    RE(mv(zps.sy, y_ini))
    RE(mv(zp.z, z_ini))
    img_norm = (img - img_dark_avg)/(img_bkg_avg - img_dark_avg)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0
    with h5py.File(fn, 'w') as hf:
        hf.create_dataset('img_norm', data = img_norm)
        hf.create_dataset('img_raw', data = img)
        hf.create_dataset('img_dark_avg', data = img_dark_avg)
        hf.create_dataset('img_bkg_avg', data = img_bkg_avg)
    RE(abs_set(shutter_open, 1, wait=True))


#####################a

def load_cell_scan(bender_pos_list, pbsl_pos_list, num, eng_start, eng_end, steps, delay_time=0.5):
    check_eng_range([eng_start, eng_end])

    num_pbsl_pos = len(pbsl_pos_list)

    for bender_pos in bender_pos_list:
        yield from mv(pzt_cm.setpos, bender_pos)
        yield from bps.sleep(2)
        load_cell_force = pzt_cm_loadcell.value
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)  
        for pbsl_pos in pbsl_pos_list:
            yield from mv(pbsl.y_ctr, pbsl_pos)
            for i in range(num):
                #      yield from scan([ic3, ic4], XEng, eng_start/1000, eng_end/1000, steps)
                yield from eng_scan_delay(eng_start/1000, eng_end/1000, steps, delay_time=delay_time)
                h = db[-1]
                y0 = np.array(list(h.data(ic3.name)))
                y1 = np.array(list(h.data(ic4.name)))
                r = np.log(y0/y1)
                x = np.linspace(eng_start, eng_end, steps)
                   
                ax1.plot(x, r, '.-')
                r_dif = np.array([0] + list(np.diff(r)))
                ax2.plot(x, r_dif, '.-')
#        
        ax1.title.set_text('scan_id: {}-{}, ratio of: {}/{}'.format(h.start['scan_id']-num*num_pbsl_pos+1, h.start['scan_id'], ic3.name, ic4.name))
        ax2.title.set_text('load_cell: {}, bender_pos: {}'.format(load_cell_force, bender_pos))
        fig.subplots_adjust(hspace=.5)
        plt.show()
        
###########################
def ssa_scan(bender_pos_list, ssa_motor, ssa_start, ssa_end, ssa_steps):
# scanning ssa, with different pzt_tm position

    pzt_motor = pzt_tm.setpos
    x = np.linspace(ssa_start, ssa_end, ssa_steps)
    for bender_pos in bender_pos_list:
        yield from mv(pzt_motor, bender_pos)
        yield from bps.sleep(2)
        load_cell_force = pzt_tm_loadcell.value
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
#        yield from scan([ic3, ic4, Vout2], ssa_motor, ssa_start, ssa_end, ssa_steps)
        yield from delay_scan(ssa_motor, ssa_start, ssa_end, ssa_steps, detectors=[ic3, ic4, Vout2], sleep_time=1.2, md=None)
        h = db[-1]
        y0 = np.array(list(h.data(ic3.name)))
        y1 = np.array(list(h.data(ic4.name)))      
        y2 = np.array(list(h.data(Vout2.name))) 
        ax1.plot(x, y0, '.-')
#            r_dif = np.array([0] + list(np.diff(r)))
        ax2.plot(x, y1, '.-')
        ax3.plot(x, y2, '.-')
        ax1.title.set_text('scan_id: {}, ic3'.format(h.start['scan_id']))
        ax2.title.set_text('ic4, load_cell: {}, bender_pos: {}'.format(load_cell_force, bender_pos))
        ax3.title.set_text('Vout2')
        fig.subplots_adjust(hspace=.5)
        plt.show()
        
        
def ssa_scan_tm_yaw(tm_yaw_pos_list, ssa_motor, ssa_start, ssa_end, ssa_steps):
# scanning ssa, with different pzt_tm position

    motor = tm.yaw
    x = np.linspace(ssa_start, ssa_end, ssa_steps)
    for tm_yaw_pos in tm_yaw_pos_list:
        yield from mv(motor, tm_yaw_pos)
        yield from bps.sleep(2)
        load_cell_force = pzt_tm_loadcell.value
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
#        yield from scan([ic3, ic4, Vout2], ssa_motor, ssa_start, ssa_end, ssa_steps)
        yield from delay_scan(ssa_motor, ssa_start, ssa_end, ssa_steps, detectors=[ic3, ic4, Vout2], sleep_time=1.2, md=None)
        h = db[-1]
        y0 = np.array(list(h.data(ic3.name)))
        y1 = np.array(list(h.data(ic4.name)))      
        y2 = np.array(list(h.data(Vout2.name))) 
        ax1.plot(x, y0, '.-')
#            r_dif = np.array([0] + list(np.diff(r)))
        ax2.plot(x, y1, '.-')
        ax3.plot(x, y2, '.-')
        ax1.title.set_text('scan_id: {}, ic3'.format(h.start['scan_id']))
        ax2.title.set_text('ic4, load_cell: {}'.format(load_cell_force))
        ax3.title.set_text('Vout2, tm_yaw = {}'.format(tm_yaw_pos))
        fig.subplots_adjust(hspace=.5)
        plt.show()
        
def ssa_scan_pbsl_x_gap(pbsl_x_gap_list, ssa_motor, ssa_start, ssa_end, ssa_steps):
# scanning ssa, with different pzt_tm position

    motor = pbsl.x_gap
    x = np.linspace(ssa_start, ssa_end, ssa_steps)
    for pbsl_x_gap in pbsl_x_gap_list:
        yield from mv(motor, pbsl_x_gap)
        yield from bps.sleep(2)
        load_cell_force = pzt_tm_loadcell.value
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
#        yield from scan([ic3, ic4, Vout2], ssa_motor, ssa_start, ssa_end, ssa_steps)
        yield from delay_scan(ssa_motor, ssa_start, ssa_end, ssa_steps, detectors=[ic3, ic4, Vout2], sleep_time=1.2, md=None)
        h = db[-1]
        y0 = np.array(list(h.data(ic3.name)))
        y1 = np.array(list(h.data(ic4.name)))      
        y2 = np.array(list(h.data(Vout2.name))) 
        ax1.plot(x, y0, '.-')
#            r_dif = np.array([0] + list(np.diff(r)))
        ax2.plot(x, y1, '.-')
        ax3.plot(x, y2, '.-')
        ax1.title.set_text('scan_id: {}, ic3'.format(h.start['scan_id']))
        ax2.title.set_text('ic4, load_cell: {}'.format(load_cell_force))
        ax3.title.set_text('Vout2, pbsl_x_gap = {}'.format(pbsl_x_gap))
        fig.subplots_adjust(hspace=.5)
        plt.show()
        
        
def ssa_scan_pbsl_y_gap(pbsl_y_gap_list, ssa_motor, ssa_start, ssa_end, ssa_steps):
# scanning ssa, with different pzt_tm position

    motor = pbsl.y_gap
    x = np.linspace(ssa_start, ssa_end, ssa_steps)
    for pbsl_y_gap in pbsl_y_gap_list:
        yield from mv(motor, pbsl_y_gap)
        yield from bps.sleep(2)
        load_cell_force = pzt_tm_loadcell.value
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
#        yield from scan([ic3, ic4, Vout2], ssa_motor, ssa_start, ssa_end, ssa_steps)
        yield from delay_scan(ssa_motor, ssa_start, ssa_end, ssa_steps, detectors=[ic3, ic4, Vout2], sleep_time=1.2, md=None)
        h = db[-1]
        y0 = np.array(list(h.data(ic3.name)))
        y1 = np.array(list(h.data(ic4.name)))      
        y2 = np.array(list(h.data(Vout2.name))) 
        ax1.plot(x, y0, '.-')
#            r_dif = np.array([0] + list(np.diff(r)))
        ax2.plot(x, y1, '.-')
        ax3.plot(x, y2, '.-')
        ax1.title.set_text('scan_id: {}, ic3'.format(h.start['scan_id']))
        ax2.title.set_text('ic4, load_cell: {}'.format(load_cell_force))
        ax3.title.set_text('Vout2, pbsl_y_gap = {}'.format(pbsl_y_gap))
        fig.subplots_adjust(hspace=.5)
        plt.show()
        
def repeat_scan(detectors, motor, start, stop, steps, num=1, sleep_time=1.2):
    for i in range(num):
        yield from delay_scan(motor, start, stop, steps, detectors, sleep_time=1.2)
      

def xanes_3d_scan(eng_list, exposure_time, relative_rot_angle, period, chunk_size=20, out_x=0, out_y=0, rs=3, parkpos=None, note=''):
    '''
    eng is in KeV
    
    '''
    id_list=[]
    
    my_eng_list = eng_list * 1000.0
    for eng in my_eng_list:
        RE(move_zp_ccd(eng))
        RE(fly_scan(exposure_time, relative_rot_angle, period, chunk_size, out_x, out_y, rs, parkpos, note))
        scan_id=db[-1].start['scan_id']
        id_list.append(int(scan_id))
        print('current energy: {} --> scan_id: {}\n'.format(eng, scan_id))
    return my_eng_list, id_list
