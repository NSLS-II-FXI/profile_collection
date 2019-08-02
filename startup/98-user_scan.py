'''
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_email(subject,
               body,
               hostname,
               port,
               user,
               password,
               recipients,
               attachment_path=None):
    """Sends an email, and possibly an attachment, to the given recipients.
    Args:
        subject: The email subject text.
        body: The email body text.
        host: Hostname of the SMTP email server.
        port: Port on the host to connect on.
        user: Email address to send the email from.
        password: Password of the sending address.
        recipients: A list of email addresses to send the message to.
        attachment_path (optional): Path to the attachment file.
    """
    # Create message and add body
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From']    = user
    msg['To'] = ', '.join(recipients)
    msg.attach(MIMEText(body))
    # Add attachment to message
    if attachment_path != None:
        attachment = open(attachment_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename="{}"'.format(attachment_path))
        msg.attach(part)
    # Send the message
    server = smtplib.SMTP(hostname, port)
    server.starttls()
    server.login(user, password)
    server.sendmail(from_addr = user,
                    to_addrs = recipients,
                    msg = msg.as_string())
    server.quit()

'''


def user_scan(exposure_time, period, out_x, out_y, out_z, rs=1, out_r=0, xanes_flag=False, xanes_angle=0, note=''):
# Ni
    angle_ini = 0
    yield from mv(zps.pi_r, angle_ini)
    print('start taking tomo and xanes of Ni')
    yield from move_zp_ccd(8.35, move_flag=1)  
    yield from fly_scan(exposure_time, relative_rot_angle=180, period=period, out_x=out_x, out_y=out_y,out_z=out_z, rs=rs, parkpos=out_r, note=note+'_8.35keV')
    yield from bps.sleep(2)
    yield from move_zp_ccd(8.3, move_flag=1)
    yield from fly_scan(exposure_time, relative_rot_angle=180, period=period, out_x=out_x, out_y=out_y,out_z=out_z, rs=rs, parkpos=out_r, note=note+'8.3keV')
    yield from mv(zps.pi_r, xanes_angle)
    if xanes_flag:
        yield from xanes_scan2(eng_list_Ni, exposure_time, chunk_size=5, out_x=out_x, out_y=out_y,out_z=out_z, out_r=out_r, note=note+'_xanes') 
    yield from mv(zps.pi_r, angle_ini)

    '''
    # Co
    print('start taking tomo and xanes of Co')
    yield from mv(zps.pi_r, angle_ini)
    yield from move_zp_ccd(7.75, move_flag=1) 
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)

    yield from move_zp_ccd(7.66, move_flag=1)  
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)
    yield from mv(zps.pi_r, xanes_angle)
    if xanes_flag:
        yield from xanes_scan2(eng_list_Co, 0.05, chunk_size=5, out_x=out_x, out_y=out_y,note=note)
    yield from mv(zps.pi_r, angle_ini)

    # Mn

    print('start taking tomo and xanes of Mn')
    yield from mv(zps.pi_r, angle_ini)
    yield from move_zp_ccd(6.59, move_flag=1)
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)

    yield from move_zp_ccd(6.49, move_flag=1)  
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)
    yield from mv(zps.pi_r, xanes_angle)
    if xanes_flag:
        yield from xanes_scan2(eng_list_Mn, 0.1, chunk_size=5, out_x=out_x, out_y=out_y,note=note) 
    yield from mv(zps.pi_r, angle_ini)

    '''
def user_xanes(out_x, out_y, note=''):
    '''
    yield from move_zp_ccd(7.4, move_flag=1, xanes_flag='2D')  
    yield from bps.sleep(1)
    yield from xanes_scan2(eng_list_Co, 0.05, chunk_size=5, out_x=out_x, out_y=out_y, note=note)
    yield from bps.sleep(5)
    '''
    print('please wait for 5 sec...starting Ni xanes')
    yield from move_zp_ccd(8.3, move_flag=1)  
    yield from bps.sleep(1)
    yield from xanes_scan2(eng_list_Ni, 0.05, chunk_size=5, out_x=out_x, out_y=out_y,note=note)

'''
def user_flyscan(out_x, out_y, note=''):
    yield from move_zp_ccd(8.35, move_flag=1, xanes_flag='2D') 
    yield from bps.sleep(1)
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)
    yield from move_zp_ccd(8.3, move_flag=1, xanes_flag='2D') 
    yield from bps.sleep(1)
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)

    yield from move_zp_ccd(7.75, move_flag=1, xanes_flag='2D') 
    yield from bps.sleep(1)
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)
    yield from move_zp_ccd(7.66, move_flag=1, xanes_flag='2D') 
    yield from bps.sleep(1)
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)

    yield from move_zp_ccd(6.59, move_flag=1, xanes_flag='2D') 
    yield from bps.sleep(1)
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)
    yield from move_zp_ccd(6.49, move_flag=1, xanes_flag='2D') 
    yield from bps.sleep(1)
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)
'''





def overnight_fly():
    insert_text('start William Zhou in-situ scan at 10min interval for 70 times:')
    for i in range(70):
        print(f'current scan# {i}')
        yield from abs_set(shutter_open, 1)
        yield from sleep(1)
        yield from abs_set(shutter_open, 1)
        yield from sleep(2)
        yield from fly_scan(exposure_time=0.05, relative_rot_angle=180, period=0.05, chunk_size=20, out_x=0, out_y=0, out_z=1000, out_r=0, rs=3,simu=False, note='WilliamZhou_DOW_Water_drying_insitu_scan@8.6keV,w/filter 1&2')
           
        yield from abs_set(shutter_close, 1)
        yield from sleep(1)
        yield from abs_set(shutter_close, 1)
        yield from sleep(2)
        yield from bps.sleep(520) 
    insert_text('finished pin-situ scan')

    
   
def insitu_xanes_scan(eng_list, exposure_time=0.2, out_x=0, out_y=0, out_z=0, out_r=0, repeat_num=1, sleep_time=1, note='None'):
    insert_text('start from now on, taking in-situ NMC charge/discharge xanes scan:')
    for i in range(repeat_num):
        print(f'scan #{i}\n')
        yield from xanes_scan2(eng_list, exposure_time=exposure_time, chunk_size=2, out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r, note=f'{note}_#{i}')
        current_time = str(datetime.now().time())[:8]
        print(f'current time is {current_time}')
        insert_text(f'current scan finished at: {current_time}')
        yield from abs_set(shutter_close, 1)
        yield from bps.sleep(1)
        yield from abs_set(shutter_close, 1)
        print(f"\nI'm sleeping for {sleep_time} sec ...\n")
        yield from bps.sleep(sleep_time)  
    insert_text('finished in-situ xanes scan !!')
    
    


def user_fly_scan(exposure_time=0.1, period=0.1, chunk_size=20, rs=1, note='', simu=False, md=None):

    '''
    motor_x_ini = zps.pi_x.position
  #  motor_x_out = motor_x_ini + txm_out_x
    motor_y_ini = zps.sy.position
    motor_y_out = motor_y_ini + out_y
    motor_z_ini = zps.sz.position
    motor_z_out = motor_z_ini + out_z
    motor_r_ini = zps.pi_r.position
    motor_r_out = motor_r_ini + out_r
    '''
    motor_r_ini = zps.pi_r.position
    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r, zps.pi_x]

    detectors = [Andor, ic3]
    offset_angle = -2.0 * rs
    current_rot_angle = zps.pi_r.position

  #  target_rot_angle = current_rot_angle + relative_rot_angle
    _md = {'detectors': ['Andor'],
           'motors': [mot.name for mot in motor],
           'XEng': XEng.position,
           'ion_chamber': ic3.name,
           'plan_args': {'exposure_time': exposure_time,
                         'period': period,
                         'chunk_size': chunk_size,
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
    
    print('set rotation speed: {} deg/sec'.format(rs))


    @stage_decorator(list(detectors) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def inner_scan():
        #close shutter, dark images: numer=chunk_size (e.g.20)
        print('\nshutter closed, taking dark images...')
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

        yield from mv(zps.pi_x, 0)
        yield from mv(zps.pi_r, -50)
        yield from _set_rotation_speed(rs=rs)
        #open shutter, tomo_images
        yield from _open_shutter(simu=simu)
        print ('\nshutter opened, taking tomo images...')
        yield from mv(zps.pi_r, -50+offset_angle)
        status = yield from abs_set(zps.pi_r, 50, wait=False)
        yield from bps.sleep(2)
        while not status.done:
            yield from trigger_and_read(list(detectors) + motor)
        # bkg images
        print ('\nTaking background images...')
        yield from _set_rotation_speed(rs=30)
        yield from mv(zps.pi_r, 0)
        
        
        yield from mv(zps.pi_x, 12)
        yield from mv(zps.pi_r, 70)
        yield from trigger_and_read(list(detectors) + motor)

        yield from _close_shutter(simu=simu)
        yield from mv(zps.pi_r, 0)
        yield from mv(zps.pi_x, 0)
        yield from mv(zps.pi_x, 0)
        #yield from mv(zps.pi_r, motor_r_ini)

    uid = yield from inner_scan()
    print('scan finished')
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)
    return uid

def tmp_scan():
    x = np.array([0,1,2,3])*0.015*2560 + zps.sx.position
    y = np.array([0,1,2,3])*0.015*2160 + zps.sy.position

    i=0; j=0;
    for xx in x:
        i += 1
        for yy in y:
            j += 1
            print(f'current {i}_{j}: x={xx}, y={yy}')
            yield from mv(zps.sx, xx, zps.sy, yy)
            yield from xanes_scan2(eng_Ni_list_xanes, 0.05, chunk_size=4, out_x=2000, out_y=0, out_z=0, out_r=0, simu=False, note='NCM532_72cycle_discharge_{i}_{j}')



def mosaic_fly_scan(x_list, y_list, z_list, r_list, exposure_time=0.1, relative_rot_angle = 150, period=0.1, chunk_size=20, out_x=None, out_y=None, out_z=4400,  out_r=90, rs=1, note='', simu=False, relative_move_flag=0, traditional_sequence_flag=0):
    txt = 'start mosaic_fly_scan, containing following fly_scan\n'
    insert_text(txt)
    insert_text('x_list = ')
    insert_text(str(x_list))
    insert_text('y_list = ')
    insert_text(str(y_list))
    insert_text('z_list = ')
    insert_text(str(z_list))
    insert_text('r_list = ')
    insert_text(str(r_list))
    
    nx = len(x_list)
    ny = len(y_list)
    for i in range(ny):
        for j in range(nx):
            success = False
            count = 1        
            while not success and count < 20:        
                try:
                    RE(mv(zps.sx, x_list[j], zps.sy, y_list[i], zps.sz, z_list[i], zps.pi_r, r_list[i]))
                    RE(fly_scan(exposure_time, relative_rot_angle, period, chunk_size, out_x, out_y, out_z,  out_r, rs, note, simu, relative_move_flag, traditional_sequence_flag, md=None))
                    success = True
                except:
                    count += 1
                    RE.abort()
                    Andor.unstage()
                    print('sleeping for 30 sec')
                    RE(bps.sleep(30))
                    txt = f'Redo scan at x={x_list[i]}, y={y_list[i]}, z={z_list[i]} for {count} times'
                    print(txt)
                    insert_text(txt)                
    txt = 'mosaic_fly_scan finished !!\n'
    insert_text(txt)
        
def mosaic2d_lists(x_start, x_end, x_step, y_start, y_end, y_step, z, r):
    x_range = list(range(x_start, x_end + x_step, x_step))
    y_range = list(range(y_start, y_end + y_step, y_step))
    x_list = x_range * len(y_range)
    y_list = []
    for y in y_range:
        y_list.extend([y] * len(x_range))
    z_list = [z] * len(x_list)
    r_list = [r] * len(x_list)
    return x_list, y_list, z_list, r_list


    

def multi_pos_3D_xanes(eng_list, x_list=[0], y_list=[0], z_list=[0], r_list=[0], exposure_time= 0.05, relative_rot_angle=182, rs=2,):
    '''
    the sample_out position is in its absolute value:
    will move sample to out_x (um) out_y (um) out_z(um) and out_r (um) to take background image

    to run:

    RE(multi_pos_3D_xanes(Ni_eng_list, x_list=[a, b, c], y_list=[aa,bb,cc], z_list=[aaa,bbb, ccc], r_list=[0, 0, 0], exposure_time=0.05, relative_rot_angle=185, rs=3, out_x=1500, out_y=-1500, out_z=-770, out_r=0, note='NC')
    '''
    num_pos = len(x_list)
    for i in range(num_pos):
        print(f'currently, taking 3D xanes at position {i}\n')
        yield from mv(zps.sx, x_list[i], zps.sy, y_list[i], zps.sz, z_list[i], zps.pi_r, r_list[i])
        yield from bps.sleep(2)
        note_pos = note + f'position_{i}'
        yield from xanes_3D(eng_list, exposure_time=exposure_time, relative_rot_angle=relative_rot_angle, period=exposure_time, out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r, rs=rs, simu=False, relative_move_flag=0, traditional_sequence_flag=1, note=note_pos)
        insert_text(f'finished 3D xanes scan for {note_pos}')



def multi_pos_2D_and_3D_xanes(elements=['Ni'], sam_in_pos_list_2D=[[0, 0, 0, 0]], sam_out_pos_list_2D=[[0, 0, 0, 0]], sam_in_pos_list_3D=[[0, 0, 0, 0]], sam_out_pos_list_3D=[[0, 0, 0, 0]], exposure_time=[0.05], relative_rot_angle=182, rs=1, note=''):
    sam_in_pos_list_2D = np.asarray(sam_in_pos_list_2D)
    sam_out_pos_list_2D = np.asarray(sam_out_pos_list_2D)    
    sam_in_pos_list_3D = np.asarray(sam_in_pos_list_3D)
    sam_out_pos_list_3D = np.asarray(sam_out_pos_list_3D)
    exposure_time = np.asarray(exposure_time)
    if exposure_time.shape[0] == 1:
        exposure_time = np.ones(len(elements))*exposure_time[0]
    elif len(elements) != exposure_time.shape[0]:
        # to do in bs manner
        pass

    eng_list = []
    for ii in elements:
        eng_list.append(list(np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/'+ii+'/eng_list_'+ii+'_xanes_standard.txt')))

    for ii in range(sam_in_pos_list_2D.shape[0]):
        for jj in range(len(elements)):
            x_list = [sam_in_pos_list_2D[ii, 0]]
            y_list = [sam_in_pos_list_2D[ii, 1]]
            z_list = [sam_in_pos_list_2D[ii, 2]]
            r_list = [sam_in_pos_list_2D[ii, 3]]
            out_x = sam_out_pos_list_2D[ii, 0]
            out_y = sam_out_pos_list_2D[ii, 1]
            out_z = sam_out_pos_list_2D[ii, 2]
            out_r = sam_out_pos_list_2D[ii, 3]
            yield from multipos_2D_xanes_scan2(eng_list[jj], x_list, y_list, z_list, r_list, 
                                               out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r, 
                                               exposure_time=exposure_time[jj], chunk_size=5,
                                               simu=False, relative_move_flag=True, note=note, md=None, sleep_time=0, repeat_num=1)

    for ii in range(sam_in_pos_list_3D.shape[0]):
        for jj in range(len(elements)):
            x_list = [sam_in_pos_list_3D[ii, 0]]
            y_list = [sam_in_pos_list_3D[ii, 1]]
            z_list = [sam_in_pos_list_3D[ii, 2]]
            r_list = [sam_in_pos_list_3D[ii, 3]]
            out_x = sam_out_pos_list_3D[ii, 0]
            out_y = sam_out_pos_list_3D[ii, 1]
            out_z = sam_out_pos_list_3D[ii, 2]
            out_r = sam_out_pos_list_3D[ii, 3]
            yield from multi_pos_3D_xanes(eng_list[jj], x_list, y_list, z_list, r_list, 
                                          exposure_time=exposure_time[jj], relative_rot_angle=relative_rot_angle, rs=rs, 
                                          out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r, note=note, simu=False, 
                                          relative_move_flag=1, traditional_sequence_flag=1, sleep_time=0, repeat=1) 


def zps_motor_scan_with_Andor(motors, starts, ends, num_steps, out_x=100, out_y=0, out_z=0, out_r=0, exposure_time=None, period=None, chunk_size=1, note='', relative_move_flag=1, simu=False, rot_first_flag=0, md=None):
    global ZONE_PLATE 
    detectors = [Andor, ic3]

#    if len(out_x) != len(motors):
#        out_x = [out_x[0]] * len(motors)
#        
#    if len(out_y) != len(motors):
#        out_y = [out_y[0]] * len(motors)  
#        
#    if len(out_z) != len(motors):
#        out_z = [out_z[0]] * len(motors)
#
#    if len(out_r) != len(motors):
#        out_r = [out_r[0]] * len(motors)        

    def _set_andor_param():
        yield from mv(Andor.cam.acquire, 0)
        yield from mv(Andor.cam.image_mode, 0)
        yield from mv(Andor.cam.num_images, chunk_size)
        yield from mv(Andor.cam.acquire_time, exposure_time)
        Andor.cam.acquire_period.put(period)
        
    if exposure_time is not None:
        yield from _set_andor_param()

    mot_ini = []
    mot_start = []
    mot_end = []
    for start, end, motor in zip(starts, ends, motors):
        mot_ini.append(getattr(motor, 'position'))
        mot_start.append(getattr(motor, 'position')+start)
        mot_end.append(getattr(motor, 'position')+end)
     
    mot_num_step = np.int_(num_steps)    
#        
#
#    motor_out = []        
#    if relative_move_flag:  
#        for motor in motors:
#            motor_out.append(motor_ini + out)

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
    
    
    
    print('hello1')
    _md = {'detectors': [det.name for det in detectors],
           'motors': [mot.name for mot in motors],
           'num_bkg_images': 5,
           'num_dark_images': 5,
           'mot_start': starts,
           'motor_end': ends,
           'motor_num_step': mot_num_step,
           'out_x': out_x,
           'out_y': out_y,
           'out_z': out_z,
           'out_r': out_r,
           'exposure_time': exposure_time,
           'chunk_size': chunk_size,
           'XEng': XEng.position,
           'plan_args': {'mot_start': mot_start,
                         'mot_end': mot_end,
                         'mot_num_step': mot_num_step,
                         'exposure_time': exposure_time, 
                         'chunk_size': chunk_size,
                         'out_x': out_x,
                         'out_y': out_y,
                         'out_z': out_z,
                         'out_r': out_r,
                         'note': note if note else 'None',
                         'relative_move_flag': relative_move_flag,
                         'rot_first_flag': rot_first_flag,
                         'note': note if note else 'None',
                         'zone_plate': ZONE_PLATE,
                        },     
           'plan_name': 'zps_motor_scan_with_Andor',
           'hints': {},
           'operator': 'FXI',
           'zone_plate': ZONE_PLATE,
           'note': note if note else 'None',
           #'motor_pos':  wh_pos(print_on_screen=0),
            }
    _md.update(md or {})
    try:   dimensions = [(motors.hints['fields'], 'primary')]
    except (AttributeError, KeyError):  pass
    else:   _md['hints'].setdefault('dimensions', dimensions)

    @stage_decorator(list(detectors) + motors)
    @run_decorator(md=_md)
    def zps_motor_scan_inner():
        # take dark image
        print('take 5 dark image')
        yield from _take_dark_image(detectors, motors, num_dark=5)

        print('open shutter ...')
        yield from _open_shutter(simu)

        print('taking mosaic image ...')
        if len(motors) == 1:
            mot_pos = np.linspace(mot_start[0], mot_end[0], mot_num_step[0], endpoint=False)
        elif len(motors) == 2:
            mot_pos_coor1, mot_pos_coor2 = np.meshgrid(np.linspace(mot_start[0], mot_end[0], mot_num_step[0], endpoint=False),
                                                       np.linspace(mot_start[1], mot_end[1], mot_num_step[1], endpoint=False))
            mot_pos = np.array([mot_pos_coor1.flatten(), mot_pos_coor2.flatten()])                                                       
        elif len(motors) == 3:
            mot_pos_coor1, mot_pos_coor2, mot_pos_coor3 = np.meshgrid(np.linspace(mot_start[0], mot_end[0], mot_num_step[0], endpoint=False),
                                                                      np.linspace(mot_start[1], mot_end[1], mot_num_step[1], endpoint=False),
                                                                      np.linspace(mot_start[2], mot_end[2], mot_num_step[2], endpoint=False))   
            mot_pos = np.array([mot_pos_coor1.flatten(), mot_pos_coor2.flatten(), mot_pos_coor3.flatten()])                                                                      
        elif len(motors) == 4:
            mot_pos_coor1, mot_pos_coor2, mot_pos_coor3, mot_pos_coor4 = np.meshgrid(np.linspace(mot_start[0], mot_end[0], mot_num_step[0], endpoint=False),
                                                                                     np.linspace(mot_start[1], mot_end[1], mot_num_step[1], endpoint=False),
                                                                                     np.linspace(mot_start[2], mot_end[2], mot_num_step[2], endpoint=False),
                                                                                     np.linspace(mot_start[3], mot_end[3], mot_num_step[3], endpoint=False))  
            mot_pos = np.array([mot_pos_coor1.flatten(), mot_pos_coor2.flatten(), mot_pos_coor3.flatten(), mot_pos_coor4.flatten()])    

        for jj in range(mot_pos.shape[1]): 
#            yield from mv(motors, mot_pos[:, jj])
            for ii in range(len(motors)):
                yield from mv(motors[ii], mot_pos[ii, jj])                                                                                  
            yield from _take_image(detectors, motors, 1) 

        print('moving sample out to take 5 background image')
        yield from _take_bkg_image(motor_x_out, motor_y_out, motor_z_out, motor_r_out, detectors, motors, num_bkg=5, simu=simu,traditional_sequence_flag=rot_first_flag)    

        # move sample in
        yield from _move_sample_in(motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini, repeat=1,trans_first_flag=1-rot_first_flag)

        print('closing shutter')
        yield from _close_shutter(simu) 
  
    yield from zps_motor_scan_inner()
    yield from mv(Andor.cam.image_mode, 1)
    print('scan finished')
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    