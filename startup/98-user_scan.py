def user_scan(out_x, out_y, xanes_flag=False, xanes_angle=0, note=''):
# Ni
    angle_ini = 0
    yield from mv(zps.pi_r, angle_ini)
    print('start taking tomo and xanes of Ni')
    yield from move_zp_ccd(8.35, move_flag=1)  
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)

    yield from move_zp_ccd(8.3, move_flag=1)
    yield from fly_scan(0.05, relative_rot_angle=180, period=0.05, out_x=out_x, out_y=out_y,out_z=0, rs=2, parkpos=0, note=note)
    yield from mv(zps.pi_r, xanes_angle)
    if xanes_flag:
        yield from xanes_scan2(eng_list_Ni, 0.05, chunk_size=5, out_x=out_x, out_y=out_y,note=note) 
    yield from mv(zps.pi_r, angle_ini)

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

def xanes_3D(eng_list, exposure_time=0.05, relative_rot_angle=180, period=0.05, out_x=0, out_y=0, out_z=0, out_r=0, rs=2, note=''):
    txt = 'start 3D xanes scan, containing following fly_scan:\n'
    insert_text(txt)
    for eng in eng_list:        
        yield from move_zp_ccd(eng, move_flag=1)
        my_note = note + f'_energy={eng}'
        yield from bps.sleep(1)
        print(f'current energy: {eng}')
        yield from fly_scan(exposure_time, relative_rot_angle=relative_rot_angle, period=period, out_x=out_x, out_y=out_y,out_z=out_z, rs=rs, parkpos=out_r, note=my_note)
        yield from bps.sleep(1)
    export_pdf(1)



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


def multipos_count(x_list, y_list, z_list,  out_x, out_y, out_z, out_r, exposure_time=0.1, repeat_num=1, sleep_time=0, note='', md=None):

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



    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.image_mode, 0)
    yield from mv(Andor.cam.num_images, 1)
    yield from abs_set(Andor.cam.acquire_time, exposure_time)
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
        yield from abs_set(shutter_close, 1)
        yield from sleep(1)
        yield from abs_set(shutter_close, 1)
        yield from sleep(2)
        for i in range(10):
            yield from trigger_and_read(list([Andor, ic3])+[zps.sx, zps.sy, zps.sz, zps.pi_r])

        for repeat in range(repeat_num):
            print('\nshutter open ...')
            for i in range(num):

                yield from abs_set(shutter_open, 1)
                yield from sleep(1)
                yield from abs_set(shutter_open, 1)
                yield from sleep(2)
            
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

            yield from abs_set(shutter_close, 1)
            yield from sleep(1)
            yield from abs_set(shutter_close, 1)
            yield from sleep(2)
            print(f'sleep for {sleep_time} sec ...')
            yield from bps.sleep(sleep_time)
    
    yield from inner_scan()
    print('scan finished')
    txt = get_scan_parameter()
    insert_text(txt) 

        
        























