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

import numpy as np
import sys


def select_filters(flts=[]):
    yield from _close_shutter(simu=False)
    for key, item in filters.items():
        yield from mv(item, 0)
    for ii in flts:
        yield from mv(filters["filter" + str(ii)], 1)


def user_scan(
    exposure_time,
    period,
    out_x,
    out_y,
    out_z,
    rs=1,
    out_r=0,
    xanes_flag=False,
    xanes_angle=0,
    note="",
):
    # Ni
    angle_ini = 0
    yield from mv(zps.pi_r, angle_ini)
    print("start taking tomo and xanes of Ni")
    yield from move_zp_ccd(8.35, move_flag=1)
    yield from fly_scan(
        exposure_time,
        relative_rot_angle=180,
        period=period,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
        rs=rs,
        parkpos=out_r,
        note=note + "_8.35keV",
    )
    yield from bps.sleep(2)
    yield from move_zp_ccd(8.3, move_flag=1)
    yield from fly_scan(
        exposure_time,
        relative_rot_angle=180,
        period=period,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
        rs=rs,
        parkpos=out_r,
        note=note + "8.3keV",
    )
    yield from mv(zps.pi_r, xanes_angle)
    if xanes_flag:
        yield from xanes_scan2(
            eng_list_Ni,
            exposure_time,
            chunk_size=5,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            note=note + "_xanes",
        )
    yield from mv(zps.pi_r, angle_ini)

    """
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

    """


def user_xanes(out_x, out_y, note=""):
    """
    yield from move_zp_ccd(7.4, move_flag=1, xanes_flag='2D')
    yield from bps.sleep(1)
    yield from xanes_scan2(eng_list_Co, 0.05, chunk_size=5, out_x=out_x, out_y=out_y, note=note)
    yield from bps.sleep(5)
    """
    print("please wait for 5 sec...starting Ni xanes")
    yield from move_zp_ccd(8.3, move_flag=1)
    yield from bps.sleep(1)
    yield from xanes_scan2(
        eng_list_Ni, 0.05, chunk_size=5, out_x=out_x, out_y=out_y, note=note
    )


"""
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
"""


def overnight_fly():
    insert_text("start William Zhou in-situ scan at 10min interval for 70 times:")
    for i in range(70):
        print(f"current scan# {i}")
        yield from abs_set(shutter_open, 1)
        yield from sleep(1)
        yield from abs_set(shutter_open, 1)
        yield from sleep(2)
        yield from fly_scan(
            exposure_time=0.05,
            relative_rot_angle=180,
            period=0.05,
            chunk_size=20,
            out_x=0,
            out_y=0,
            out_z=1000,
            out_r=0,
            rs=3,
            simu=False,
            note="WilliamZhou_DOW_Water_drying_insitu_scan@8.6keV,w/filter 1&2",
        )

        yield from abs_set(shutter_close, 1)
        yield from sleep(1)
        yield from abs_set(shutter_close, 1)
        yield from sleep(2)
        yield from bps.sleep(520)
    insert_text("finished pin-situ scan")


def insitu_xanes_scan(
    eng_list,
    exposure_time=0.2,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    repeat_num=1,
    sleep_time=1,
    note="None",
):
    insert_text("start from now on, taking in-situ NMC charge/discharge xanes scan:")
    for i in range(repeat_num):
        print(f"scan #{i}\n")
        yield from xanes_scan2(
            eng_list,
            exposure_time=exposure_time,
            chunk_size=2,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            note=f"{note}_#{i}",
        )
        current_time = str(datetime.now().time())[:8]
        print(f"current time is {current_time}")
        insert_text(f"current scan finished at: {current_time}")
        yield from abs_set(shutter_close, 1)
        yield from bps.sleep(1)
        yield from abs_set(shutter_close, 1)
        print(f"\nI'm sleeping for {sleep_time} sec ...\n")
        yield from bps.sleep(sleep_time)
    insert_text("finished in-situ xanes scan !!")


def user_fly_scan(
    exposure_time=0.1, period=0.1, chunk_size=20, rs=1, note="", simu=False, md=None
):

    """
    motor_x_ini = zps.pi_x.position
  #  motor_x_out = motor_x_ini + txm_out_x
    motor_y_ini = zps.sy.position
    motor_y_out = motor_y_ini + out_y
    motor_z_ini = zps.sz.position
    motor_z_out = motor_z_ini + out_z
    motor_r_ini = zps.pi_r.position
    motor_r_out = motor_r_ini + out_r
    """
    motor_r_ini = zps.pi_r.position
    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r, zps.pi_x]

    detectors = [Andor, ic3]
    offset_angle = -2.0 * rs
    current_rot_angle = zps.pi_r.position

    #  target_rot_angle = current_rot_angle + relative_rot_angle
    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motor],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exposure_time,
            "period": period,
            "chunk_size": chunk_size,
            "rs": rs,
            "note": note if note else "None",
        },
        "plan_name": "fly_scan",
        "num_bkg_images": chunk_size,
        "num_dark_images": chunk_size,
        "chunk_size": chunk_size,
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        "motor_pos": wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(zps.pi_r.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    yield from _set_andor_param(
        exposure_time=exposure_time, period=period, chunk_size=chunk_size
    )

    print("set rotation speed: {} deg/sec".format(rs))

    @stage_decorator(list(detectors) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def inner_scan():
        # close shutter, dark images: numer=chunk_size (e.g.20)
        print("\nshutter closed, taking dark images...")
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

        yield from mv(zps.pi_x, 0)
        yield from mv(zps.pi_r, -50)
        yield from _set_rotation_speed(rs=rs)
        # open shutter, tomo_images
        yield from _open_shutter(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from mv(zps.pi_r, -50 + offset_angle)
        status = yield from abs_set(zps.pi_r, 50, wait=False)
        yield from bps.sleep(2)
        while not status.done:
            yield from trigger_and_read(list(detectors) + motor)
        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=30)
        yield from mv(zps.pi_r, 0)

        yield from mv(zps.pi_x, 12)
        yield from mv(zps.pi_r, 70)
        yield from trigger_and_read(list(detectors) + motor)

        yield from _close_shutter(simu=simu)
        yield from mv(zps.pi_r, 0)
        yield from mv(zps.pi_x, 0)
        yield from mv(zps.pi_x, 0)
        # yield from mv(zps.pi_r, motor_r_ini)

    uid = yield from inner_scan()
    print("scan finished")
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)
    return uid


def tmp_scan():
    x = np.array([0, 1, 2, 3]) * 0.015 * 2560 + zps.sx.position
    y = np.array([0, 1, 2, 3]) * 0.015 * 2160 + zps.sy.position

    i = 0
    j = 0
    for xx in x:
        i += 1
        for yy in y:
            j += 1
            print(f"current {i}_{j}: x={xx}, y={yy}")
            yield from mv(zps.sx, xx, zps.sy, yy)
            yield from xanes_scan2(
                eng_Ni_list_xanes,
                0.05,
                chunk_size=4,
                out_x=2000,
                out_y=0,
                out_z=0,
                out_r=0,
                simu=False,
                note="NCM532_72cycle_discharge_{i}_{j}",
            )


def mosaic_fly_scan(
    x_list,
    y_list,
    z_list,
    r_list,
    exposure_time=0.1,
    relative_rot_angle=150,
    period=0.1,
    chunk_size=20,
    out_x=None,
    out_y=None,
    out_z=4400,
    out_r=90,
    rs=1,
    note="",
    simu=False,
    relative_move_flag=0,
    traditional_sequence_flag=0,
):
    txt = "start mosaic_fly_scan, containing following fly_scan\n"
    insert_text(txt)
    insert_text("x_list = ")
    insert_text(str(x_list))
    insert_text("y_list = ")
    insert_text(str(y_list))
    insert_text("z_list = ")
    insert_text(str(z_list))
    insert_text("r_list = ")
    insert_text(str(r_list))

    nx = len(x_list)
    ny = len(y_list)
    for i in range(ny):
        for j in range(nx):
            success = False
            count = 1
            while not success and count < 20:
                try:
                    RE(
                        mv(
                            zps.sx,
                            x_list[j],
                            zps.sy,
                            y_list[i],
                            zps.sz,
                            z_list[i],
                            zps.pi_r,
                            r_list[i],
                        )
                    )
                    RE(
                        fly_scan(
                            exposure_time,
                            relative_rot_angle,
                            period,
                            chunk_size,
                            out_x,
                            out_y,
                            out_z,
                            out_r,
                            rs,
                            note,
                            simu,
                            relative_move_flag,
                            traditional_sequence_flag,
                            md=None,
                        )
                    )
                    success = True
                except:
                    count += 1
                    RE.abort()
                    Andor.unstage()
                    print("sleeping for 30 sec")
                    RE(bps.sleep(30))
                    txt = f"Redo scan at x={x_list[i]}, y={y_list[i]}, z={z_list[i]} for {count} times"
                    print(txt)
                    insert_text(txt)
    txt = "mosaic_fly_scan finished !!\n"
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


def multi_pos_3D_xanes(
    eng_list,
    x_list=[0],
    y_list=[0],
    z_list=[0],
    r_list=[0],
    exposure_time=0.05,
    relative_rot_angle=182,
    rs=2,
):
    """
    the sample_out position is in its absolute value:
    will move sample to out_x (um) out_y (um) out_z(um) and out_r (um) to take background image

    to run:

    RE(multi_pos_3D_xanes(Ni_eng_list, x_list=[a, b, c], y_list=[aa,bb,cc], z_list=[aaa,bbb, ccc], r_list=[0, 0, 0], exposure_time=0.05, relative_rot_angle=185, rs=3, out_x=1500, out_y=-1500, out_z=-770, out_r=0, note='NC')
    """
    num_pos = len(x_list)
    for i in range(num_pos):
        print(f"currently, taking 3D xanes at position {i}\n")
        yield from mv(
            zps.sx, x_list[i], zps.sy, y_list[i], zps.sz, z_list[i], zps.pi_r, r_list[i]
        )
        yield from bps.sleep(2)
        note_pos = note + f"position_{i}"
        yield from xanes_3D(
            eng_list,
            exposure_time=exposure_time,
            relative_rot_angle=relative_rot_angle,
            period=exposure_time,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            rs=rs,
            simu=False,
            relative_move_flag=0,
            traditional_sequence_flag=1,
            note=note_pos,
        )
        insert_text(f"finished 3D xanes scan for {note_pos}")


def mk_eng_list(elem):
    if elem.split("_")[-1] == "wl":
        eng_list = np.genfromtxt(
            "/NSLS2/xf18id1/SW/xanes_ref/"
            + elem.split("_")[0]
            + "/eng_list_"
            + elem.split("_")[0]
            + "_s_xanes_standard_21pnt.txt")
    elif  elem.split("_")[-1] == "101":
        eng_list = np.genfromtxt(
            "/NSLS2/xf18id1/SW/xanes_ref/"
            + elem.split("_")[0]
            + "/eng_list_"
            + elem.split("_")[0]
            + "_xanes_standard_101pnt.txt")
    elif  elem.split("_")[-1] == "63":
        eng_list = np.genfromtxt(
            "/NSLS2/xf18id1/SW/xanes_ref/"
            +  elem.split("_")[0]
            + "/eng_list_"
            +  elem.split("_")[0]
            + "_xanes_standard_63pnt.txt")
    return eng_list
            
def sort_in_pos(in_pos_list):
    x_list = []
    y_list = []
    z_list = []
    r_list = []
    for ii in range(len(in_pos_list)):
        if in_pos_list[ii][0] is None:
            x_list.append(zps.sx.position)
        else:
            x_list.append(in_pos_list[ii][0])
            
        if in_pos_list[ii][1] is None:
            y_list.append(zps.sy.position)
        else:
            y_list.append(in_pos_list[ii][1])
            
        if in_pos_list[ii][2] is None:
            z_list.append(zps.sz.position)
        else:
            z_list.append(in_pos_list[ii][2])
            
        if in_pos_list[ii][3] is None:
            r_list.append(zps.pi_r.position)
        else:
            r_list.append(in_pos_list[ii][3])
            
    return (x_list, y_list, z_list, r_list)

def multi_edge_xanes(
    elements=["Ni_wl"],
    scan_type = '3D',
    filters={"Ni_filters": [1, 2, 3]},    
    exposure_time={"Ni_exp": 0.05},
    relative_rot_angle=185,
    rs=1,
    in_pos_list = [[0, 0, 0, 0]],
    out_pos=[0, 0, 0, 0],
    note="",
    relative_move_flag=0,
    binning = [2, 2],
    simu=False):
        
    x_list, y_list, z_list, r_list = sort_in_pos(in_pos_list)
    for elem in elements:
        for key in filters.keys():
            if elem.split("_")[0] == key.split("_")[0]:
               yield from select_filters(filters[key])
               break
            else:
                yield from select_filters([])
        for key in exposure_time.keys():
            if elem.split("_")[0] == key.split("_")[0]:                
                exposure = exposure_time[key]
                print(elem, exposure)
                break
            else:
                exposure =  0.05
                print('use default exposure time 0.05s')
        eng_list = mk_eng_list(elem)
        if scan_type == '2D':
            yield from multipos_2D_xanes_scan2(eng_list,
                                                x_list,
                                                y_list,
                                                z_list,
                                                r_list,
                                                out_x=out_pos[0],
                                                out_y=out_pos[1],
                                                out_z=out_pos[2],
                                                out_r=out_pos[3],
                                                exposure_time=exposure,
                                                chunk_size=5,
                                                simu=simu,
                                                relative_move_flag=relative_move_flag,
                                                note=note,
                                                md=None,
                                                sleep_time=0,
                                                binning = [2, 2],
                                                repeat_num=1)
        elif scan_type == '3D':
            yield from multi_pos_xanes_3D(eng_list,
                                        x_list,
                                        y_list,
                                        z_list,
                                        r_list,
                                        exposure_time=exposure,
                                        relative_rot_angle=relative_rot_angle,
                                        rs=rs,
                                        out_x=out_pos[0],
                                        out_y=out_pos[1],
                                        out_z=out_pos[2],
                                        out_r=out_pos[3],
                                        note=note,
                                        simu=simu,
                                        relative_move_flag=relative_move_flag,
                                        traditional_sequence_flag=1,
                                        sleep_time=0,
                                        binning = [2, 2],
                                        repeat=1)
        else:
            print('wrong scan type')
            return


def fly_scan2(
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

    motors = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    detectors = [Andor, ic3]
    offset_angle = -2 * rs
    current_rot_angle = zps.pi_r.position

    target_rot_angle = current_rot_angle + relative_rot_angle
    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motors],
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
        "plan_name": "fly_scan2",
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
    yield from abs_set(Andor.cam.acquire_time, exposure_time, wait=True)
#    yield from abs_set(Andor.cam.acquire_period, period, wait=True)
    Andor.cam.acquire_period.put(period)
    
#    yield from _set_andor_param(
#        exposure_time=exposure_time, period=period, chunk_size=chunk_size
#    )
    yield from _set_rotation_speed(rs=rs)
    print("set rotation speed: {} deg/sec".format(rs))

    # We manually stage the Andor detector below. See there for why....
    # Stage everything else here in the usual way.
    @stage_decorator([ic3] + motors)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        # set filters
        for flt in filters:
            yield from mv(flt, 1)
            yield from mv(flt, 1)
        yield from abs_set(Andor.cam.num_images, chunk_size, wait=True)

        # Manually stage the Andor. This creates a Resource document that
        # contains the path to the HDF5 file where the detector writes. It also
        # encodes the so-called 'frame_per_point' which here is what this plan
        # calls chunk_size. The chunk_size CANNOT BE CHANGED later in the scan
        # unless we unstage and re-stage the detector and generate a new
        # Resource document.

        # This approach imposes some unfortunate overhead (closing the HDF5
        # file, opening a new one, going through all the steps to set the Area
        # Detector's filepath PV, etc.). A better approach has been sketched
        # in https://github.com/bluesky/area-detector-handlers/pull/11. It
        # allows a single HDF5 file to contain multiple chunk_sizes.

        yield from bps.stage(Andor)
        yield from bps.sleep(1)
        
        # open shutter, tomo_images
        yield from _open_shutter(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from mv(zps.pi_r, current_rot_angle + offset_angle)
        status = yield from abs_set(zps.pi_r, target_rot_angle, wait=False)
        yield from bps.sleep(2)
        while not status.done:
            yield from trigger_and_read(list(detectors) + motors)

        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=rot_back_velo)        
        yield from  abs_set(Andor.cam.num_images, 20, wait=True)

        # Now that the new chunk_size has been set (20) create a new Resource
        # document by unstage and re-staging the detector.
        yield from bps.unstage(Andor)
        yield from bps.stage(Andor)

        yield from bps.sleep(1)
        yield from _take_bkg_image(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            detectors,
            motors,
            num_bkg=1,
            simu=False,
            traditional_sequence_flag=rot_first_flag,
        )
        
        # dark images
        yield from _close_shutter(simu=simu)
        print("\nshutter closed, taking dark images...")
        yield from _take_dark_image(detectors, motors, num_dark=1, simu=simu)

        yield from bps.unstage(Andor)
        
        # restore fliters
        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            trans_first_flag=rot_first_flag,
        )
        for flt in filters:
            yield from mv(flt, 0)

    yield from fly_inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    print("scan finished")
    txt = get_scan_parameter(print_flag=0)
    insert_text(txt)
    print(txt)
    
def dummy_scan( exposure_time=0.1,
    start_angle = None,
    relative_rot_angle=180,
    period=0.15,
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
    repeat=1):   
        
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

    motors = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    detectors = [Andor, ic3]
    offset_angle = -2 * rs
    current_rot_angle = zps.pi_r.position

    target_rot_angle = current_rot_angle + relative_rot_angle
    _md={'dummy scan':'dummy scan'}

    yield from mv(Andor.cam.acquire, 0)
    yield from _set_andor_param(
        exposure_time=exposure_time, period=period
    )
    yield from mv(Andor.cam.image_mode, 1)
    yield from mv(Andor.cam.acquire, 1)
    
    yield from _set_rotation_speed(rs=rs)
    print("set rotation speed: {} deg/sec".format(rs))

    @stage_decorator(motors)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        # open shutter, tomo_images
        yield from _open_shutter(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from mv(zps.pi_r, current_rot_angle + offset_angle)
        status = yield from abs_set(zps.pi_r, target_rot_angle, wait=False)
        while not status.done:
            yield from bps.sleep(1)
        status = yield from abs_set(zps.pi_r, current_rot_angle + offset_angle, wait=False)
        while not status.done:
            yield from bps.sleep(1)
    for ii in range(repeat):
        yield from fly_inner_scan()
    yield from _set_rotation_speed(rs=rot_back_velo)
    print("dummy scan finished")

def radiographic_record(exp_t=0.1, period=0.1, t_span=10, stop=True, 
                        out_x=None, out_y=None, out_z=None, out_r=None, 
                        filters=[], md={}, note="", simu=False,
                        rot_first_flag=1, relative_move_flag=1):
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

    motors = [zps.sx, zps.sy, zps.sz, zps.pi_r]
                        
    detectors = [Andor, ic3]
    _md = {
        "detectors": ["Andor"],
#        "motors": [mot.name for mot in motors],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exp_t,
            "period": period,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "time_span": t_span,
            "filters": [filt.name for filt in filters] if filters else "None",
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "radiographic_record",
        "num_bkg_images": 20,
        "num_dark_images": 20,
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        "zone_plate": ZONE_PLATE
    }
    _md.update(md or {})

    yield from mv(Andor.cam.acquire, 0)
    yield from _set_andor_param(
        exposure_time=exp_t, period=period
    )
    yield from mv(Andor.cam.image_mode, 0)    
    
    @stage_decorator(list(detectors))
#    @bpp.monitor_during_decorator([Andor.cam.num_images_counter])
    @run_decorator(md=_md)
    def rad_record_inner():
        yield from _open_shutter(simu=simu)        
        for flt in filters:
            yield from mv(flt, 1)
            yield from mv(flt, 1)
        yield from bps.sleep(1)        
        
        yield from mv(Andor.cam.num_images, int(t_span/period))
        yield from trigger_and_read([Andor])
        
        yield from mv(zps.sx, motor_x_out, 
                      zps.sy, motor_y_out, 
                      zps.sz, motor_z_out,
                      zps.pi_r, motor_r_out)  
        yield from mv(Andor.cam.num_images,20)
        yield from trigger_and_read([Andor])
        yield from _close_shutter(simu=simu)
        yield from mv(zps.sx, motor_x_ini, 
                      zps.sy, motor_y_ini, 
                      zps.sz, motor_z_ini,
                      zps.pi_r, motor_r_ini) 
        yield from trigger_and_read([Andor])
        yield from mv(Andor.cam.image_mode, 1)
        for flt in filters:
            yield from mv(flt, 0)
        
    yield from rad_record_inner()
    
# def multi_pos_2D_and_3D_xanes(elements=['Ni'], sam_in_pos_list_2D=[[[0, 0, 0, 0],]], sam_out_pos_list_2D=[[[0, 0, 0, 0],]], sam_in_pos_list_3D=[[[0, 0, 0, 0],]], sam_out_pos_list_3D=[[[0, 0, 0, 0],]], exposure_time=[0.05], relative_rot_angle=182, relative_move_flag=False, rs=1, note=''):
#    sam_in_pos_list_2D = np.asarray(sam_in_pos_list_2D)
#    sam_out_pos_list_2D = np.asarray(sam_out_pos_list_2D)
#    sam_in_pos_list_3D = np.asarray(sam_in_pos_list_3D)
#    sam_out_pos_list_3D = np.asarray(sam_out_pos_list_3D)
#    exposure_time = np.asarray(exposure_time)
#    if exposure_time.shape[0] == 1:
#        exposure_time = np.ones(len(elements))*exposure_time[0]
#    elif len(elements) != exposure_time.shape[0]:
#        # to do in bs manner
#        pass
#
#    eng_list = []
#    for ii in elements:
#        eng_list.append(list(np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/'+ii+'/eng_list_'+ii+'_xanes_standard.txt')))
#
#    for ii in range(sam_in_pos_list_2D.shape[0]):
#        for jj in range(len(elements)):
#            x_list = [sam_in_pos_list_2D[ii, :, 0]]
#            y_list = [sam_in_pos_list_2D[ii, :, 1]]
#            z_list = [sam_in_pos_list_2D[ii, :, 2]]
#            r_list = [sam_in_pos_list_2D[ii, :, 3]]
#            out_x = sam_out_pos_list_2D[ii, :, 0]
#            out_y = sam_out_pos_list_2D[ii, :, 1]
#            out_z = sam_out_pos_list_2D[ii, :, 2]
#            out_r = sam_out_pos_list_2D[ii, :, 3]
#            yield from multipos_2D_xanes_scan2(eng_list[jj], x_list, y_list, z_list, r_list,
#                                               out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r,
#                                               exposure_time=exposure_time[jj], chunk_size=5,
#                                               simu=False, relative_move_flag=relative_move_flag, note=note, md=None, sleep_time=0, repeat_num=1)
#
#    for ii in range(sam_in_pos_list_3D.shape[0]):
#        for jj in range(len(elements)):
#            x_list = [sam_in_pos_list_3D[ii, :, 0]]
#            y_list = [sam_in_pos_list_3D[ii, :, 1]]
#            z_list = [sam_in_pos_list_3D[ii, :, 2]]
#            r_list = [sam_in_pos_list_3D[ii, :, 3]]
#            out_x = sam_out_pos_list_3D[ii, :, 0]
#            out_y = sam_out_pos_list_3D[ii, :, 1]
#            out_z = sam_out_pos_list_3D[ii, :, 2]
#            out_r = sam_out_pos_list_3D[ii, :, 3]
#            yield from multi_pos_3D_xanes(eng_list[jj], x_list, y_list, z_list, r_list,
#                                          exposure_time=exposure_time[jj], relative_rot_angle=relative_rot_angle, rs=rs,
#                                          out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r, note=note, simu=False,
#                                          relative_move_flag=relative_move_flag, traditional_sequence_flag=1, sleep_time=0, repeat=1)


# def multi_pos_2D_xanes_and_3D_tomo(elements=['Ni'], sam_in_pos_list_2D=[[[0, 0, 0, 0]]], sam_out_pos_list_2D=[[[0, 0, 0, 0]]], sam_in_pos_list_3D=[[[0, 0, 0, 0]]], sam_out_pos_list_3D=[[[0, 0, 0, 0]]],
#                                  exposure_time_2D=[0.05], exposure_time_3D=[0.05], relative_rot_angle=182, rs=1, eng_3D=[8.4], note='', relative_move_flag=False):
#    sam_in_pos_list_2D = np.asarray(sam_in_pos_list_2D)
#    sam_out_pos_list_2D = np.asarray(sam_out_pos_list_2D)
#    sam_in_pos_list_3D = np.asarray(sam_in_pos_list_3D)
#    sam_out_pos_list_3D = np.asarray(sam_out_pos_list_3D)
#    exposure_time_2D = np.asarray(exposure_time_2D)
#    exposure_time_3D = np.asarray(exposure_time_3D)
#    if exposure_time_2D.shape[0] == 1:
#        exposure_time_2D = np.ones(len(elements))*exposure_time_2D[0]
#    elif len(elements) != exposure_time_2D.shape[0]:
#        # to do in bs manner
#        pass
#
#    if exposure_time_3D.shape[0] == 1:
#        exposure_time_3D = np.ones(len(elements))*exposure_time_3D[0]
#    elif len(elements) != exposure_time_3D.shape[0]:
#        # to do in bs manner
#        pass
#
#    eng_list = []
#    for ii in elements:
#        eng_list.append(list(np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/'+ii+'/eng_list_'+ii+'_xanes_standard.txt')))
#
#    for ii in range(sam_in_pos_list_2D.shape[0]):
#        for jj in range(len(elements)):
#            x_list = sam_in_pos_list_2D[ii, :, 0]
#            y_list = sam_in_pos_list_2D[ii, :, 1]
#            z_list = sam_in_pos_list_2D[ii, :, 2]
#            r_list = sam_in_pos_list_2D[ii, :, 3]
#            out_x = sam_out_pos_list_2D[ii, 0]
#            out_y = sam_out_pos_list_2D[ii, 1]
#            out_z = sam_out_pos_list_2D[ii, 2]
#            out_r = sam_out_pos_list_2D[ii, 3]
#            print(x_list)
#            print(y_list)
#            print(z_list)
#            print(r_list)
#            print(out_x)
#            print(out_y)
#            print(out_z)
#            print(out_r)
#            yield from multipos_2D_xanes_scan2(eng_list[jj], x_list, y_list, z_list, r_list,
#                                               out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r,
#                                               exposure_time=exposure_time_2D[jj], chunk_size=5,
#                                               simu=False, relative_move_flag=relative_move_flag, note=note, md=None, sleep_time=0, repeat_num=1)
#
#    for ii in range(sam_in_pos_list_3D.shape[0]):
#        for jj in range(len(elements)):
#            x_list = sam_in_pos_list_3D[ii, :, 0]
#            y_list = sam_in_pos_list_3D[ii, :, 1]
#            z_list = sam_in_pos_list_3D[ii, :, 2]
#            r_list = sam_in_pos_list_3D[ii, :, 3]
#            out_x = sam_out_pos_list_3D[ii, 0]
#            out_y = sam_out_pos_list_3D[ii, 1]
#            out_z = sam_out_pos_list_3D[ii, 2]
#            out_r = sam_out_pos_list_3D[ii, 3]
#            yield from multi_pos_xanes_3D(eng_3D, x_list, y_list, z_list, r_list,
#                                          exposure_time=exposure_time_3D[jj], relative_rot_angle=relative_rot_angle, rs=rs,
#                                          out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r, note=note, simu=False,
#                                          relative_move_flag=relative_move_flag, traditional_sequence_flag=1, sleep_time=0, repeat=1)


############ old routine: 2D routine works but 3D routine has some bugs -- start
# def  multi_pos_2D_and_3D_xanes(elements=['Ni_short'], filters=[[1, 2, 3]], sam_in_pos_list_2D=[[[0, 0, 0, 0]]], sam_out_pos_list_2D=[[[0, 0, 0, 0]]], sam_in_pos_list_3D=[[[0, 0, 0, 0]]], sam_out_pos_list_3D=[[[0, 0, 0, 0]]],
#                                  exposure_time_2D=[0.05], exposure_time_3D=[0.05], relative_rot_angle=182, rs=1, sleep_time=0, repeat_num=1, note='', relative_move_flag=0, simu=False):
#    """
#    pos_list layer structure: 1st layer -> energy
#                              2nd layer -> multiple positions at the given energy
#                              3rd layer -> individual postion in the multiple poistion list
#    """
#    for kk in range(repeat_num):
#        sam_in_pos_list_2D = np.asarray(sam_in_pos_list_2D)
#        sam_out_pos_list_2D = np.asarray(sam_out_pos_list_2D)
#        sam_in_pos_list_3D = np.asarray(sam_in_pos_list_3D)
#        sam_out_pos_list_3D = np.asarray(sam_out_pos_list_3D)
#        exposure_time_2D = np.asarray(exposure_time_2D)
#        exposure_time_3D = np.asarray(exposure_time_3D)
#        if exposure_time_2D.shape[0] == 1:
#            exposure_time_2D = np.ones(len(elements))*exposure_time_2D[0]
#        elif len(elements) != exposure_time_2D.shape[0]:
#            # to do in bs manner
#            pass
#
#        if exposure_time_3D.shape[0] == 1:
#            exposure_time_3D = np.ones(len(elements))*exposure_time_3D[0]
#        elif len(elements) != exposure_time_3D.shape[0]:
#            # to do in bs manner
#            pass
#
#        eng_list = []
#        for ii in elements:
#            if ii.split('_')[1] == 'wl':
#                eng_list.append(list(np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/'+ii.split('_')[0]+'/eng_list_'+ii.split('_')[0]+'_s_xanes_standard_21pnt.txt')))
#            elif ii.split('_')[1] == '101':
#                eng_list.append(list(np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/'+ii+'/eng_list_'+ii+'_xanes_standard_101pnt.txt')))
#            elif ii.split('_')[1] == '63':
#                eng_list.append(list(np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/'+ii+'/eng_list_'+ii+'_xanes_standard_63pnt.txt')))
#
#        eng_list = np.array(eng_list)
#
#        if  sam_in_pos_list_2D.size != 0:
#            for ii in range(sam_in_pos_list_2D.shape[0]):
#                for jj in range(len(elements)):
#                    if filters[jj]:
#                        select_filters(filters[jj])
##                    yield from _close_shutter(simu=simu)
##                    yield from mv(filter1, 0)
##                    yield from mv(filter2, 0)
##                    yield from mv(filter3, 0)
##                    yield from mv(filter4, 0)
##                    for flt in filters[jj]:
##                        if flt == 'filter1':
##                            yield from mv(filter1, 1)
##                        elif flt == 'filter2':
##                            yield from mv(filter2, 1)
##                        elif flt == 'filter3':
##                            yield from mv(filter3, 1)
##                        elif flt == 'filter4':
##                            yield from mv(filter4, 1)
#                    x_list = sam_in_pos_list_2D[ii, :, 0]
#                    y_list = sam_in_pos_list_2D[ii, :, 1]
#                    z_list = sam_in_pos_list_2D[ii, :, 2]
#                    r_list = sam_in_pos_list_2D[ii, :, 3]
#                    out_x = sam_out_pos_list_2D[ii, :, 0]
#                    out_y = sam_out_pos_list_2D[ii, :, 1]
#                    out_z = sam_out_pos_list_2D[ii, :, 2]
#                    out_r = sam_out_pos_list_2D[ii, :, 3]
#                    print(x_list)
#                    print(y_list)
#                    print(z_list)
#                    print(r_list)
#                    print(out_x)
#                    print(out_y)
#                    print(out_z)
#                    print(out_r)
#                    yield from multipos_2D_xanes_scan2(eng_list[jj], x_list, y_list, z_list, r_list,
#                                                       out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r,
#                                                       exposure_time=exposure_time_2D[jj], chunk_size=5,
#                                                       simu=simu, relative_move_flag=relative_move_flag, note=note, md=None, sleep_time=0, repeat_num=1)
#
#        if sam_in_pos_list_3D.size != 0:
#            for ii in range(sam_in_pos_list_3D.shape[0]):
#                for jj in range(len(elements)):
#                    if filters[jj]:
#                        select_filters(filters[jj])
##                    yield from _close_shutter(simu=simu)
##                    yield from mv(filter1, 0)
##                    yield from mv(filter2, 0)
##                    yield from mv(filter3, 0)
##                    yield from mv(filter4, 0)
##                    for flt in filters[jj]:
##                        if flt == 'filter1':
##                            yield from mv(filter1, 1)
##                        elif flt == 'filter2':
##                            yield from mv(filter2, 1)
##                        elif flt == 'filter3':
##                            yield from mv(filter3, 1)
##                        elif flt == 'filter4':
##                            yield from mv(filter4, 1)
#                    x_list = sam_in_pos_list_3D[ii, :, 0]
#                    y_list = sam_in_pos_list_3D[ii, :, 1]
#                    z_list = sam_in_pos_list_3D[ii, :, 2]
#                    r_list = sam_in_pos_list_3D[ii, :, 3]
#                    out_x = sam_out_pos_list_3D[ii, :, 0]
#                    out_y = sam_out_pos_list_3D[ii, :, 1]
#                    out_z = sam_out_pos_list_3D[ii, :, 2]
#                    out_r = sam_out_pos_list_3D[ii, :, 3]
#                    print(x_list, out_x, out_y, out_z, out_r)
#                    yield from multi_pos_xanes_3D(eng_list[jj], x_list, y_list, z_list, r_list,
#                                                  exposure_time=exposure_time_3D[jj], relative_rot_angle=relative_rot_angle, rs=rs,
#                                                  out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r, note=note, simu=simu,
#                                                  relative_move_flag=relative_move_flag, traditional_sequence_flag=1, sleep_time=0, repeat=1)
#        if kk != (repeat_num-1):
#            print(f'We are in multi_pos_2D_and_3D_xanes cycle # {kk}; we are going to sleep for {sleep_time} seconds ...')
#            yield from bps.sleep(sleep_time)
############ old routine: 2D routine works but 3D routine has some bugs -- end


def multi_pos_2D_and_3D_xanes(
    elements=["Ni_wl"],
    filters={"Ni_filters": [1, 2, 3]},
    sam_in_pos_list_2D={"Ni_2D_in_pos_list": [[0, 0, 0, 0]]},
    sam_out_pos_list_2D={"Ni_2D_out_pos_list": [[0, 0, 0, 0]]},
    sam_in_pos_list_3D={"Ni_3D_in_pos_list": [[0, 0, 0, 0]]},
    sam_out_pos_list_3D={"Ni_3D_out_pos_list": [[0, 0, 0, 0]]},
    exposure_time_2D={"Ni_2D_exp": 0.05},
    exposure_time_3D={"Ni_3D_exp": 0.05},
    relative_rot_angle=185,
    rs=1,
    sleep_time=0,
    repeat_num=1,
    note="",
    relative_move_flag=0,
    simu=False):

    xanes2D = {}
    xanes3D = {}
    for kk in range(repeat_num):
        for elem in elements:
            ### if there is a filter combination is defined for the element
            for key, item in sam_in_pos_list_2D.items():
                if elem.split("_")[0] == key.split("_")[0]:
                    xanes2D[elem+'_2D'] = {}
                    xanes2D[elem+'_2D']['eng'] = elem
                    xanes2D[elem+'_2D']['in_pos'] =  item
                    xanes2D[elem+'_2D']['in_pos_defined'] =  True
            for key, item in filters.items():
                if elem.split("_")[0] == key.split("_")[0]:
                    xanes2D[elem+'_2D']['filter'] = item
                else:
                    xanes2D[elem+'_2D']['filter'] = []
            for key, item in sam_out_pos_list_2D.items():
                if elem.split("_")[0] == key.split("_")[0]:
                    xanes2D[elem+'_2D']['out_pos'] =  item
                    xanes2D[elem+'_2D']['out_pos_defined'] =  True
            for key, item in exposure_time_2D.items():
                if elem.split("_")[0] == key.split("_")[0]:
                    xanes2D[elem+'_2D']['exposure'] =  item
                    xanes2D[elem+'_2D']['exposure_defined'] =  True
            if not (xanes2D[elem+'_2D']['in_pos_defined'] &
                    xanes2D[elem+'_2D']['out_pos_defined'] &
                    xanes2D[elem+'_2D']['exposure_defined']):
                print(elem+' 2D scan setup is not correct. Quit.')
                sys.exit()
        for elem in elements:
            ### if there is a filter combination is defined for the element
            for key, item in sam_in_pos_list_3D.items():
                if elem.split("_")[0] == key.split("_")[0]:
                    xanes3D[elem+'_3D'] = {}
                    xanes3D[elem+'_3D']['eng'] = elem
                    xanes3D[elem+'_3D']['in_pos'] =  item
                    xanes3D[elem+'_3D']['in_pos_defined'] =  True
            for key, item in filters.items():
                if elem.split("_")[0] == key.split("_")[0]:
                    xanes3D[elem+'_3D']['filter'] = item
                else:
                    xanes3D[elem+'_3D']['filter'] = []
            for key, item in sam_out_pos_list_3D.items():
                if elem.split("_")[0] == key.split("_")[0]:
                    xanes3D[elem+'_3D']['out_pos'] =  item
                    xanes3D[elem+'_3D']['out_pos_defined'] =  True
            for key, item in exposure_time_3D.items():
                if elem.split("_")[0] == key.split("_")[0]:
                    xanes3D[elem+'_3D']['exposure'] =  item
                    xanes3D[elem+'_3D']['exposure_defined'] =  True
            if not (xanes3D[elem+'_3D']['in_pos_defined'] &
                    xanes3D[elem+'_3D']['out_pos_defined'] &
                    xanes3D[elem+'_3D']['exposure_defined']):
                print(elem+' 3D scan setup is not correct. Quit.')
                sys.exit()
    for elem2D in xanes2D:
        x_list_2D = []
        y_list_2D = []
        z_list_2D = []
        r_list_2D = []
        out_x_2D = []
        out_y_2D = []
        out_z_2D = []
        out_r_2D = []
        for inpos in elem2D['in_pos']:
            x_list_2D.append(inpos[0])
            y_list_2D.append(inpos[1])
            z_list_2D.append(inpos[2])
            r_list_2D.append(inpos[3])
        for outpos in elem2D['out_pos']:
            out_x_2D.append(outpos[0])
            out_y_2D.append(outpos[1])
            out_z_2D.append(outpos[2])
            out_r_2D.append(outpos[3])
        if len(x_list_2D) != len(out_x_2D):
            print('x_list_2D and out_x_2D are not equal in length. Quit.')
            sys.exit()

        select_filters(elem2D['filter'])

        if elem2D['eng'].split("_")[-1] == "wl":
            eng_list = np.genfromtxt(
                "/NSLS2/xf18id1/SW/xanes_ref/"
                + elem2D['eng'].split("_")[0]
                + "/eng_list_"
                + elem2D['eng'].split("_")[0]
                + "_s_xanes_standard_21pnt.txt")
        elif  elem2D['eng'].split("_")[-1] == "101":
            eng_list = np.genfromtxt(
                "/NSLS2/xf18id1/SW/xanes_ref/"
                + elem2D['eng'].split("_")
                + "/eng_list_"
                + elem2D['eng'].split("_")
                + "_xanes_standard_101pnt.txt")
        elif  elem2D['eng'].split("_")[-1] == "63":
            eng_list = np.genfromtxt(
                "/NSLS2/xf18id1/SW/xanes_ref/"
                + elem2D['eng'].split("_")
                + "/eng_list_"
                + elem2D['eng'].split("_")
                + "_xanes_standard_63pnt.txt")

        yield from multipos_2D_xanes_scan2(
                    eng_list,
                    x_list_2D,
                    y_list_2D,
                    z_list_2D,
                    r_list_2D,
                    out_x=out_x_2D,
                    out_y=out_y_2D,
                    out_z=out_z_2D,
                    out_r=out_r_2D,
                    exposure_time=elem2D['exposure'],
                    chunk_size=5,
                    simu=simu,
                    relative_move_flag=relative_move_flag,
                    note=note,
                    md=None,
                    sleep_time=0,
                    repeat_num=1)

    for elem3D in xanes3D:
        x_list_3D = []
        y_list_3D = []
        z_list_3D = []
        r_list_3D = []
        out_x_3D = []
        out_y_3D = []
        out_z_3D = []
        out_r_3D = []
        for inpos in elem3D['in_pos']:
            x_list_3D.append(inpos[0])
            y_list_3D.append(inpos[1])
            z_list_3D.append(inpos[2])
            r_list_3D.append(inpos[3])
        for outpos in elem3D['out_pos']:
            out_x_3D.append(outpos[0])
            out_y_3D.append(outpos[1])
            out_z_3D.append(outpos[2])
            out_r_3D.append(outpos[3])
        if len(x_list_3D) != len(out_x_3D):
            print('x_list_3D and out_x_3D are not equal in length. Quit.')
            sys.exit()

        select_filters(elem3D['filter'])

        if elem3D['eng'].split("_")[-1] == "wl":
            eng_list = np.genfromtxt(
                "/NSLS2/xf18id1/SW/xanes_ref/"
                + elem3D['eng'].split("_")[0]
                + "/eng_list_"
                + elem3D['eng'].split("_")[0]
                + "_s_xanes_standard_21pnt.txt")
        elif  elem3D['eng'].split("_")[-1] == "101":
            eng_list = np.genfromtxt(
                "/NSLS2/xf18id1/SW/xanes_ref/"
                + elem3D['eng'].split("_")
                + "/eng_list_"
                + elem3D['eng'].split("_")
                + "_xanes_standard_101pnt.txt")
        elif  elem3D['eng'].split("_")[-1] == "63":
            eng_list = np.genfromtxt(
                "/NSLS2/xf18id1/SW/xanes_ref/"
                + elem3D['eng'].split("_")
                + "/eng_list_"
                + elem3D['eng'].split("_")
                + "_xanes_standard_63pnt.txt")

        yield from multi_pos_xanes_3D(
            eng_list,
            x_list_3D,
            y_list_3D,
            z_list_3D,
            r_list_3D,
            exposure_time==elem3D['exposure'],
            relative_rot_angle=relative_rot_angle,
            rs=rs,
            out_x=out_x_3D,
            out_y=out_y_3D,
            out_z=out_z_3D,
            out_r=out_r_3D,
            note=note,
            simu=simu,
            relative_move_flag=relative_move_flag,
            traditional_sequence_flag=1,
            sleep_time=0,
            repeat=1)



#            find = False
#            defined = False
#            for flt_elem in filters.keys():
#                if elem.split("_")[0] == flt_elem.split("_")[0]:
#                    find = True
#            if find is False:
#                print("There is not filters defined for ", elem, "!")
#                sys.exit(1)
#
#            ### if there are 2D_sam_in and 2D_sam_out positions defined for the element
#            find = False
#            for in_elem in sam_in_pos_list_2D.keys():
#                if elem.split("_")[0] == in_elem.split("_")[0]:
#                    find = True
#            if find:
#                find = False
#                for out_elem in sam_out_pos_list_2D.keys():
#                    if elem.split("_")[0] == out_elem.split("_")[0]:
#                        find = True
#                if find is False:
#                    print(
#                        elem, "2D_in_pos_list and", elem, "2D_in_pos_list dont match!"
#                    )
#                    sys.exit(1)
#            if find:
#                find = False
#                for exp_elem in exposure_time_2D.keys():
#                    print(1, elem.split("_"), exp_elem.split("_"), find)
#                    if elem.split("_")[0] == exp_elem.split("_")[0]:
#                        find = True
#                if find is False:
#                    print(2, elem.split("_"), exp_elem.split("_"))
#                    print("There is not exposure_time_2D defined for", elem)
#                    sys.exit(1)
#            if find:
#                defined = True
#
#            ### if there are 3D_sam_in and 3D_sam_out positions defined for the element
#            find = False
#            for in_elem in sam_in_pos_list_3D.keys():
#                if elem.split("_")[0] == in_elem.split("_")[0]:
#                    find = True
#            if find:
#                find = False
#                for out_elem in sam_out_pos_list_3D.keys():
#                    if elem.split("_")[0] == out_elem.split("_")[0]:
#                        find = True
#                if find is False:
#                    print(
#                        elem, "3D_in_pos_list and", elem, "3D_in_pos_list dont match!"
#                    )
#                    sys.exit(1)
#            if find:
#                find = False
#                for exp_elem in exposure_time_3D.keys():
#                    if elem.split("_")[0] == exp_elem.split("_")[0]:
#                        find = True
#                if find is False:
#                    print("There is not exposure_time_3D defined for", elem)
#                    sys.exit(1)
#            if find:
#                defined = True
#
#            if not defined:
#                print("There is neither 2D nor 3D position list defined for", elem)
#                sys.exit()
#
#        for elem in elements:
#            select_filters(filters[elem.split("_")[0] + "_filters"])
#
#            if ii.split("_")[1] == "wl":
#                eng_list = np.genfromtxt(
#                    "/NSLS2/xf18id1/SW/xanes_ref/"
#                    + ii.split("_")[0]
#                    + "/eng_list_"
#                    + ii.split("_")[0]
#                    + "_s_xanes_standard_21pnt.txt"
#                )
#            elif ii.split("_")[1] == "101":
#                eng_list = np.genfromtxt(
#                    "/NSLS2/xf18id1/SW/xanes_ref/"
#                    + ii
#                    + "/eng_list_"
#                    + ii
#                    + "_xanes_standard_101pnt.txt"
#                )
#            elif ii.split("_")[1] == "63":
#                eng_list = np.genfromtxt(
#                    "/NSLS2/xf18id1/SW/xanes_ref/"
#                    + ii
#                    + "/eng_list_"
#                    + ii
#                    + "_xanes_standard_63pnt.txt"
#                )
#
#            if sam_in_pos_list_2D[elem.split("_")[0] + "_2D_in_pos_list"]:
#                x_list_2D = np.asarray(
#                    sam_in_pos_list_2D[elem.split("_")[0] + "_2D_in_pos_list"]
#                )[0, :]
#                y_list_2D = np.asarray(
#                    sam_in_pos_list_2D[elem.split("_")[0] + "_2D_in_pos_list"]
#                )[1, :]
#                z_list_2D = np.asarray(
#                    sam_in_pos_list_2D[elem.split("_")[0] + "_2D_in_pos_list"]
#                )[2, :]
#                r_list_2D = np.asarray(
#                    sam_in_pos_list_2D[elem.split("_")[0] + "_2D_in_pos_list"]
#                )[3, :]
#                if sam_out_pos_list_2D[elem.split("_")[0] + "_2D_out_pos_list"]:
#                    out_x_2D = np.asarray(
#                        sam_out_pos_list_2D[elem.split("_")[0] + "_2D_out_pos_list"]
#                    )[0, :]
#                    out_y_2D = np.asarray(
#                        sam_out_pos_list_2D[elem.split("_")[0] + "_2D_out_pos_list"]
#                    )[1, :]
#                    out_z_2D = np.asarray(
#                        sam_out_pos_list_2D[elem.split("_")[0] + "_2D_out_pos_list"]
#                    )[2, :]
#                    out_r_2D = np.asarray(
#                        sam_out_pos_list_2D[elem.split("_")[0] + "_2D_out_pos_list"]
#                    )[3, :]
#                else:
#                    print(elem, "_2D_out_pos_list is not defined!")
#                    sys.exit(1)
#
#                if exposure_time_2D[elem.split("_")[0] + "_2D_exp"]:
#                    exp_2D = exposure_time_2D[elem.split("_")[0] + "_2D_exp"]
#                else:
#                    print(elem, "_2D_exp is not defined!")
#                    sys.exit(1)
#
#                yield from multipos_2D_xanes_scan2(
#                    eng_list,
#                    x_list_2D,
#                    y_list_2D,
#                    z_list_2D,
#                    r_list_2D,
#                    out_x=out_x_2D,
#                    out_y=out_y_2D,
#                    out_z=out_z_2D,
#                    out_r=out_r_2D,
#                    exposure_time=exp_2D,
#                    chunk_size=5,
#                    simu=simu,
#                    relative_move_flag=relative_move_flag,
#                    note=note,
#                    md=None,
#                    sleep_time=0,
#                    repeat_num=1,
#                )
#
#            if sam_in_pos_list_3D[elem.split("_")[0] + "_3D_in_pos_list"]:
#                x_list_3D = np.asarray(
#                    sam_in_pos_list_3D[elem.split("_")[0] + "_3D_in_pos_list"]
#                )[0, :]
#                y_list_3D = np.asarray(
#                    sam_in_pos_list_3D[elem.split("_")[0] + "_3D_in_pos_list"]
#                )[1, :]
#                z_list_3D = np.asarray(
#                    sam_in_pos_list_3D[elem.split("_")[0] + "_3D_in_pos_list"]
#                )[2, :]
#                r_list_3D = np.asarray(
#                    sam_in_pos_list_3D[elem.split("_")[0] + "_3D_in_pos_list"]
#                )[3, :]
#                if sam_out_pos_list_3D[elem.split("_")[0] + "_3D_out_pos_list"]:
#                    out_x_3D = np.asarray(
#                        sam_out_pos_list_3D[elem.split("_")[0] + "_3D_out_pos_list"]
#                    )[0, :]
#                    out_y_3D = np.asarray(
#                        sam_out_pos_list_3D[elem.split("_")[0] + "_3D_out_pos_list"]
#                    )[1, :]
#                    out_z_3D = np.asarray(
#                        sam_out_pos_list_3D[elem.split("_")[0] + "_3D_out_pos_list"]
#                    )[2, :]
#                    out_r_3D = np.asarray(
#                        sam_out_pos_list_3D[elem.split("_")[0] + "_3D_out_pos_list"]
#                    )[3, :]
#                else:
#                    print(elem, "_3D_out_pos_list is not defined!")
#                    sys.exit(1)
#                if exposure_time_3D[elem.split("_")[0] + "_3D_exp"]:
#                    exp_3D = exposure_time_3D[elem.split("_")[0] + "_3D_exp"]
#                else:
#                    print(elem, "_3D_exp is not defined!")
#                    sys.exit(1)
#
#                yield from multi_pos_xanes_3D(
#                    eng_list,
#                    x_list_3D,
#                    y_list_3D,
#                    z_list_3D,
#                    r_list_3D,
#                    exposure_time=exp_3D,
#                    relative_rot_angle=relative_rot_angle,
#                    rs=rs,
#                    out_x=out_x_3D,
#                    out_y=out_y_3D,
#                    out_z=out_z_3D,
#                    out_r=out_r_3D,
#                    note=note,
#                    simu=simu,
#                    relative_move_flag=relative_move_flag,
#                    traditional_sequence_flag=1,
#                    sleep_time=0,
#                    repeat=1,
#                )
#
#        if kk != (repeat_num - 1):
#            print(
#                f"We are in multi_pos_2D_and_3D_xanes cycle # {kk}; we are going to sleep for {sleep_time} seconds ..."
#            )
#            yield from bps.sleep(sleep_time)


#    for kk in range(repeat_num):
#        for elem in elements:
#            ### if there is a filter combination is defined for the element
#            find = False
#            defined = False
#            for flt_elem in filters.keys():
#                if elem.split("_")[0] == flt_elem.split("_")[0]:
#                    find = True
#            if find is False:
#                print("There is not filters defined for ", elem, "!")
#                sys.exit(1)
#
#            ### if there are 2D_sam_in and 2D_sam_out positions defined for the element
#            find = False
#            for in_elem in sam_in_pos_list_2D.keys():
#                if elem.split("_")[0] == in_elem.split("_")[0]:
#                    find = True
#            if find:
#                find = False
#                for out_elem in sam_out_pos_list_2D.keys():
#                    if elem.split("_")[0] == out_elem.split("_")[0]:
#                        find = True
#                if find is False:
#                    print(
#                        elem, "2D_in_pos_list and", elem, "2D_in_pos_list dont match!"
#                    )
#                    sys.exit(1)
#            if find:
#                find = False
#                for exp_elem in exposure_time_2D.keys():
#                    print(1, elem.split("_"), exp_elem.split("_"), find)
#                    if elem.split("_")[0] == exp_elem.split("_")[0]:
#                        find = True
#                if find is False:
#                    print(2, elem.split("_"), exp_elem.split("_"))
#                    print("There is not exposure_time_2D defined for", elem)
#                    sys.exit(1)
#            if find:
#                defined = True
#
#            ### if there are 3D_sam_in and 3D_sam_out positions defined for the element
#            find = False
#            for in_elem in sam_in_pos_list_3D.keys():
#                if elem.split("_")[0] == in_elem.split("_")[0]:
#                    find = True
#            if find:
#                find = False
#                for out_elem in sam_out_pos_list_3D.keys():
#                    if elem.split("_")[0] == out_elem.split("_")[0]:
#                        find = True
#                if find is False:
#                    print(
#                        elem, "3D_in_pos_list and", elem, "3D_in_pos_list dont match!"
#                    )
#                    sys.exit(1)
#            if find:
#                find = False
#                for exp_elem in exposure_time_3D.keys():
#                    if elem.split("_")[0] == exp_elem.split("_")[0]:
#                        find = True
#                if find is False:
#                    print("There is not exposure_time_3D defined for", elem)
#                    sys.exit(1)
#            if find:
#                defined = True
#
#            if not defined:
#                print("There is neither 2D nor 3D position list defined for", elem)
#                sys.exit()
#
#        for elem in elements:
#            select_filters(filters[elem.split("_")[0] + "_filters"])
#
#            if ii.split("_")[1] == "wl":
#                eng_list = np.genfromtxt(
#                    "/NSLS2/xf18id1/SW/xanes_ref/"
#                    + ii.split("_")[0]
#                    + "/eng_list_"
#                    + ii.split("_")[0]
#                    + "_s_xanes_standard_21pnt.txt"
#                )
#            elif ii.split("_")[1] == "101":
#                eng_list = np.genfromtxt(
#                    "/NSLS2/xf18id1/SW/xanes_ref/"
#                    + ii
#                    + "/eng_list_"
#                    + ii
#                    + "_xanes_standard_101pnt.txt"
#                )
#            elif ii.split("_")[1] == "63":
#                eng_list = np.genfromtxt(
#                    "/NSLS2/xf18id1/SW/xanes_ref/"
#                    + ii
#                    + "/eng_list_"
#                    + ii
#                    + "_xanes_standard_63pnt.txt"
#                )
#
#            if sam_in_pos_list_2D[elem.split("_")[0] + "_2D_in_pos_list"]:
#                x_list_2D = np.asarray(
#                    sam_in_pos_list_2D[elem.split("_")[0] + "_2D_in_pos_list"]
#                )[0, :]
#                y_list_2D = np.asarray(
#                    sam_in_pos_list_2D[elem.split("_")[0] + "_2D_in_pos_list"]
#                )[1, :]
#                z_list_2D = np.asarray(
#                    sam_in_pos_list_2D[elem.split("_")[0] + "_2D_in_pos_list"]
#                )[2, :]
#                r_list_2D = np.asarray(
#                    sam_in_pos_list_2D[elem.split("_")[0] + "_2D_in_pos_list"]
#                )[3, :]
#                if sam_out_pos_list_2D[elem.split("_")[0] + "_2D_out_pos_list"]:
#                    out_x_2D = np.asarray(
#                        sam_out_pos_list_2D[elem.split("_")[0] + "_2D_out_pos_list"]
#                    )[0, :]
#                    out_y_2D = np.asarray(
#                        sam_out_pos_list_2D[elem.split("_")[0] + "_2D_out_pos_list"]
#                    )[1, :]
#                    out_z_2D = np.asarray(
#                        sam_out_pos_list_2D[elem.split("_")[0] + "_2D_out_pos_list"]
#                    )[2, :]
#                    out_r_2D = np.asarray(
#                        sam_out_pos_list_2D[elem.split("_")[0] + "_2D_out_pos_list"]
#                    )[3, :]
#                else:
#                    print(elem, "_2D_out_pos_list is not defined!")
#                    sys.exit(1)
#
#                if exposure_time_2D[elem.split("_")[0] + "_2D_exp"]:
#                    exp_2D = exposure_time_2D[elem.split("_")[0] + "_2D_exp"]
#                else:
#                    print(elem, "_2D_exp is not defined!")
#                    sys.exit(1)
#
#                yield from multipos_2D_xanes_scan2(
#                    eng_list,
#                    x_list_2D,
#                    y_list_2D,
#                    z_list_2D,
#                    r_list_2D,
#                    out_x=out_x_2D,
#                    out_y=out_y_2D,
#                    out_z=out_z_2D,
#                    out_r=out_r_2D,
#                    exposure_time=exp_2D,
#                    chunk_size=5,
#                    simu=simu,
#                    relative_move_flag=relative_move_flag,
#                    note=note,
#                    md=None,
#                    sleep_time=0,
#                    repeat_num=1,
#                )
#
#            if sam_in_pos_list_3D[elem.split("_")[0] + "_3D_in_pos_list"]:
#                x_list_3D = np.asarray(
#                    sam_in_pos_list_3D[elem.split("_")[0] + "_3D_in_pos_list"]
#                )[0, :]
#                y_list_3D = np.asarray(
#                    sam_in_pos_list_3D[elem.split("_")[0] + "_3D_in_pos_list"]
#                )[1, :]
#                z_list_3D = np.asarray(
#                    sam_in_pos_list_3D[elem.split("_")[0] + "_3D_in_pos_list"]
#                )[2, :]
#                r_list_3D = np.asarray(
#                    sam_in_pos_list_3D[elem.split("_")[0] + "_3D_in_pos_list"]
#                )[3, :]
#                if sam_out_pos_list_3D[elem.split("_")[0] + "_3D_out_pos_list"]:
#                    out_x_3D = np.asarray(
#                        sam_out_pos_list_3D[elem.split("_")[0] + "_3D_out_pos_list"]
#                    )[0, :]
#                    out_y_3D = np.asarray(
#                        sam_out_pos_list_3D[elem.split("_")[0] + "_3D_out_pos_list"]
#                    )[1, :]
#                    out_z_3D = np.asarray(
#                        sam_out_pos_list_3D[elem.split("_")[0] + "_3D_out_pos_list"]
#                    )[2, :]
#                    out_r_3D = np.asarray(
#                        sam_out_pos_list_3D[elem.split("_")[0] + "_3D_out_pos_list"]
#                    )[3, :]
#                else:
#                    print(elem, "_3D_out_pos_list is not defined!")
#                    sys.exit(1)
#                if exposure_time_3D[elem.split("_")[0] + "_3D_exp"]:
#                    exp_3D = exposure_time_3D[elem.split("_")[0] + "_3D_exp"]
#                else:
#                    print(elem, "_3D_exp is not defined!")
#                    sys.exit(1)
#
#                yield from multi_pos_xanes_3D(
#                    eng_list,
#                    x_list_3D,
#                    y_list_3D,
#                    z_list_3D,
#                    r_list_3D,
#                    exposure_time=exp_3D,
#                    relative_rot_angle=relative_rot_angle,
#                    rs=rs,
#                    out_x=out_x_3D,
#                    out_y=out_y_3D,
#                    out_z=out_z_3D,
#                    out_r=out_r_3D,
#                    note=note,
#                    simu=simu,
#                    relative_move_flag=relative_move_flag,
#                    traditional_sequence_flag=1,
#                    sleep_time=0,
#                    repeat=1,
#                )
#
#        if kk != (repeat_num - 1):
#            print(
#                f"We are in multi_pos_2D_and_3D_xanes cycle # {kk}; we are going to sleep for {sleep_time} seconds ..."
#            )
#            yield from bps.sleep(sleep_time)


def multi_pos_2D_xanes_and_3D_tomo(
    elements=["Ni"],
    sam_in_pos_list_2D=[[[0, 0, 0, 0]]],
    sam_out_pos_list_2D=[[[0, 0, 0, 0]]],
    sam_in_pos_list_3D=[[[0, 0, 0, 0]]],
    sam_out_pos_list_3D=[[[0, 0, 0, 0]]],
    exposure_time_2D=[0.05],
    exposure_time_3D=[0.05],
    relative_rot_angle=0,
    rs=1,
    eng_3D=[10, 60],
    note="",
    relative_move_flag=0,
    simu=False,
):
    sam_in_pos_list_2D = np.asarray(sam_in_pos_list_2D)
    sam_out_pos_list_2D = np.asarray(sam_out_pos_list_2D)
    sam_in_pos_list_3D = np.asarray(sam_in_pos_list_3D)
    sam_out_pos_list_3D = np.asarray(sam_out_pos_list_3D)
    exposure_time_2D = np.asarray(exposure_time_2D)
    exposure_time_3D = np.asarray(exposure_time_3D)
    if exposure_time_2D.shape[0] == 1:
        exposure_time_2D = np.ones(len(elements)) * exposure_time_2D[0]
    elif len(elements) != exposure_time_2D.shape[0]:
        # to do in bs manner
        pass

    if exposure_time_3D.shape[0] == 1:
        exposure_time_3D = np.ones(len(elements)) * exposure_time_3D[0]
    elif len(elements) != exposure_time_3D.shape[0]:
        # to do in bs manner
        pass

    eng_list = []
    for ii in elements:
        eng_list.append(
            list(
                np.genfromtxt(
                    "/NSLS2/xf18id1/SW/xanes_ref/"
                    + ii
                    + "/eng_list_"
                    + ii
                    + "_xanes_standard.txt"
                )
            )
        )
    eng_list = np.array(eng_list)

    if sam_in_pos_list_2D.size != 0:
        for ii in range(sam_in_pos_list_2D.shape[0]):
            for jj in range(len(elements)):
                x_list = sam_in_pos_list_2D[ii, :, 0]
                y_list = sam_in_pos_list_2D[ii, :, 1]
                z_list = sam_in_pos_list_2D[ii, :, 2]
                r_list = sam_in_pos_list_2D[ii, :, 3]
                out_x = sam_out_pos_list_2D[ii, :, 0]
                out_y = sam_out_pos_list_2D[ii, :, 1]
                out_z = sam_out_pos_list_2D[ii, :, 2]
                out_r = sam_out_pos_list_2D[ii, :, 3]
                print(x_list)
                print(y_list)
                print(z_list)
                print(r_list)
                print(out_x)
                print(out_y)
                print(out_z)
                print(out_r)
                yield from multipos_2D_xanes_scan2(
                    eng_list[jj],
                    x_list,
                    y_list,
                    z_list,
                    r_list,
                    out_x=out_x,
                    out_y=out_y,
                    out_z=out_z,
                    out_r=out_r,
                    exposure_time=exposure_time_2D[jj],
                    chunk_size=5,
                    simu=simu,
                    relative_move_flag=relative_move_flag,
                    note=note,
                    md=None,
                    sleep_time=0,
                    repeat_num=1,
                )

    if sam_in_pos_list_3D.size != 0:
        for ii in range(sam_in_pos_list_3D.shape[0]):
            for jj in range(len(elements)):
                x_list = sam_in_pos_list_3D[ii, :, 0]
                y_list = sam_in_pos_list_3D[ii, :, 1]
                z_list = sam_in_pos_list_3D[ii, :, 2]
                r_list = sam_in_pos_list_3D[ii, :, 3]
                out_x = sam_out_pos_list_3D[ii, :, 0]
                out_y = sam_out_pos_list_3D[ii, :, 1]
                out_z = sam_out_pos_list_3D[ii, :, 2]
                out_r = sam_out_pos_list_3D[ii, :, 3]
                yield from multi_pos_xanes_3D(
                    eng_list[jj, eng_3D],
                    x_list,
                    y_list,
                    z_list,
                    r_list,
                    exposure_time=exposure_time_3D[jj],
                    relative_rot_angle=relative_rot_angle,
                    rs=rs,
                    out_x=out_x,
                    out_y=out_y,
                    out_z=out_z,
                    out_r=out_r,
                    note=note,
                    simu=simu,
                    relative_move_flag=relative_move_flag,
                    traditional_sequence_flag=1,
                    sleep_time=0,
                    repeat=1,
                )


def zps_motor_scan_with_Andor(
    motors,
    starts,
    ends,
    num_steps,
    out_x=100,
    out_y=0,
    out_z=0,
    out_r=0,
    exposure_time=None,
    period=None,
    chunk_size=1,
    note="",
    relative_move_flag=1,
    simu=False,
    rot_first_flag=0,
    md=None,
):
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
        yield from mv(Andor.cam.acquire_period, exposure_time)

    if exposure_time is not None:
        yield from _set_andor_param()

    mot_ini = []
    mot_start = []
    mot_end = []
    for start, end, motor in zip(starts, ends, motors):
        mot_ini.append(getattr(motor, "position"))
        mot_start.append(getattr(motor, "position") + start)
        mot_end.append(getattr(motor, "position") + end)

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

    print("hello1")
    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motors],
        "num_bkg_images": 5,
        "num_dark_images": 5,
        "mot_start": starts,
        "motor_end": ends,
        "motor_num_step": mot_num_step,
        "out_x": out_x,
        "out_y": out_y,
        "out_z": out_z,
        "out_r": out_r,
        "exposure_time": exposure_time,
        "chunk_size": chunk_size,
        "XEng": XEng.position,
        "plan_args": {
            "mot_start": mot_start,
            "mot_end": mot_end,
            "mot_num_step": mot_num_step,
            "exposure_time": exposure_time,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "note": note if note else "None",
            "relative_move_flag": relative_move_flag,
            "rot_first_flag": rot_first_flag,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "zps_motor_scan_with_Andor",
        "hints": {},
        "operator": "FXI",
        "zone_plate": ZONE_PLATE,
        "note": note if note else "None",
        #'motor_pos':  wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    try:
        dimensions = [(motors.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + motors)
    @run_decorator(md=_md)
    def zps_motor_scan_inner():
        # take dark image
        print("take 5 dark image")
        yield from _take_dark_image(detectors, motors, num_dark=5)

        print("open shutter ...")
        yield from _open_shutter(simu)

        print("taking mosaic image ...")
        if len(motors) == 1:
            mot_pos = np.linspace(
                mot_start[0], mot_end[0], mot_num_step[0], endpoint=False
            )
        elif len(motors) == 2:
            mot_pos_coor1, mot_pos_coor2 = np.meshgrid(
                np.linspace(mot_start[0], mot_end[0], mot_num_step[0], endpoint=False),
                np.linspace(mot_start[1], mot_end[1], mot_num_step[1], endpoint=False),
            )
            mot_pos = np.array([mot_pos_coor1.flatten(), mot_pos_coor2.flatten()])
        elif len(motors) == 3:
            mot_pos_coor1, mot_pos_coor2, mot_pos_coor3 = np.meshgrid(
                np.linspace(mot_start[0], mot_end[0], mot_num_step[0], endpoint=False),
                np.linspace(mot_start[1], mot_end[1], mot_num_step[1], endpoint=False),
                np.linspace(mot_start[2], mot_end[2], mot_num_step[2], endpoint=False),
            )
            mot_pos = np.array(
                [
                    mot_pos_coor1.flatten(),
                    mot_pos_coor2.flatten(),
                    mot_pos_coor3.flatten(),
                ]
            )
        elif len(motors) == 4:
            mot_pos_coor1, mot_pos_coor2, mot_pos_coor3, mot_pos_coor4 = np.meshgrid(
                np.linspace(mot_start[0], mot_end[0], mot_num_step[0], endpoint=False),
                np.linspace(mot_start[1], mot_end[1], mot_num_step[1], endpoint=False),
                np.linspace(mot_start[2], mot_end[2], mot_num_step[2], endpoint=False),
                np.linspace(mot_start[3], mot_end[3], mot_num_step[3], endpoint=False),
            )
            mot_pos = np.array(
                [
                    mot_pos_coor1.flatten(),
                    mot_pos_coor2.flatten(),
                    mot_pos_coor3.flatten(),
                    mot_pos_coor4.flatten(),
                ]
            )

        for jj in range(mot_pos.shape[1]):
            #            yield from mv(motors, mot_pos[:, jj])
            for ii in range(len(motors)):
                yield from mv(motors[ii], mot_pos[ii, jj])
            yield from _take_image(detectors, motors, 1)

        print("moving sample out to take 5 background image")
        yield from _take_bkg_image(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            detectors,
            motors,
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

        print("closing shutter")
        yield from _close_shutter(simu)

    yield from zps_motor_scan_inner()
    yield from mv(Andor.cam.image_mode, 1)
    print("scan finished")
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)


def diff_tomo(
    sam_in_pos_list=[[0, 0, 0, 0],],
    sam_out_pos_list=[[0, 0, 0, 0],],
    exposure=[0.05],
    period=[0.05],
    relative_rot_angle=182,
    rs=1,
    eng=None,
    note="",
    filters=[],
    relative_move_flag=0,
    md=None,
):
    sam_in_pos_list = np.array(sam_in_pos_list)
    sam_out_pos_list = np.array(sam_out_pos_list)

    if eng is None:
        print("Please specify two energies as a list for differential tomo scans.")
        return

    if len(exposure) != sam_in_pos_list.shape[0]:
        exposure = np.ones(sam_in_pos_list.shape[0]) * exposure[0]

    if len(period) != sam_in_pos_list.shape[0]:
        period = np.ones(sam_in_pos_list.shape[0]) * period[0]

    for jj in range(sam_in_pos_list.shape[0]):
        for ii in range(len(eng)):
            yield from move_zp_ccd(
                eng[ii], move_flag=1, info_flag=1, move_clens_flag=0, move_det_flag=0
            )
            yield from mv(
                zps.sx,
                sam_in_pos_list[jj, 0],
                zps.sy,
                sam_in_pos_list[jj, 1],
                zps.sz,
                sam_in_pos_list[jj, 2],
            )
            yield from mv(zps.pi_r, sam_in_pos_list[jj, 3])
            yield from fly_scan(
                exposure_time=exposure[jj],
                relative_rot_angle=relative_rot_angle,
                period=period[jj],
                chunk_size=20,
                out_x=sam_out_pos_list[jj, 0],
                out_y=sam_out_pos_list[jj, 1],
                out_z=sam_out_pos_list[jj, 2],
                out_r=sam_out_pos_list[jj, 3],
                rs=rs,
                note=note,
                simu=False,
                relative_move_flag=relative_move_flag,
                traditional_sequence_flag=1,
                filters=filters,
                md=md,
            )


def damon_scan(
    eng_list1,
    eng_list2,
    x_list,
    y_list,
    z_list,
    r_list,
    exposure_time1=10.0,
    exposure_time2=10.0,
    chunk_size1=1,
    chunk_size2=1,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    iters=10,
    sleep_time=1,
    note="",
):

    export_pdf(1)
    insert_text('start "damon_scan"')
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    z_list = np.array(z_list)
    for n in range(iters):
        print(f"iteration # {n+1} / {iters}")
        """
        yield from move_zp_ccd(6.5)
        for i in range(4):
            yield from mv(filter2, 1)
            yield from mv(filter4, 1)
        yield from xanes_scan2() # for Mn


        yield from move_zp_ccd(9)
        for i in range(4):
            yield from mv(filter2, 1)
            yield from mv(filter4, 1)
        yield from xanes_scan2() # for Cu
        """

        yield from move_zp_ccd(6.5, move_flag=1, move_clens_flag=1, move_det_flag=0)
        # for i in range(4):
        #     yield from mv(filter1, 0)
        #     yield from mv(filter2, 0)
        #    yield from mv(filter3, 0)
        #    yield from mv(filter4, 0)
        #    yield from mv(ssa.v_gap, 1)

        # yield from multipos_2D_xanes_scan2(eng_list1, x_list, y_list, z_list, r_list, out_x, out_y, out_z, out_r, repeat_num=1, exposure_time=exposure_time1, sleep_time=0, chunk_size=chunk_size1, relative_move_flag=1, note=note)

        # once move energy above 8.86 keV, we have a sample shift of -40(x) and -20(y),
        # the sample at focus will not be at rotation center, but it is ok if doing 2D XANES

        yield from move_zp_ccd(
            eng_list2[0], move_flag=1, move_clens_flag=1, move_det_flag=0
        )
        for i in range(4):
            yield from mv(filter1, 0)
            yield from mv(filter2, 0)
            yield from mv(filter3, 1)
            yield from mv(filter4, 1)
            yield from mv(ssa.v_gap, 0.2)
        yield from multipos_2D_xanes_scan2(
            eng_list2,
            x_list,
            y_list,
            z_list,
            r_list,
            out_x,
            out_y,
            out_z,
            out_r,
            repeat_num=1,
            exposure_time=exposure_time2,
            sleep_time=0,
            chunk_size=chunk_size2,
            relative_move_flag=1,
            note=note,
        )

        print(f"sleep for {sleep_time} sec")
        yield from bps.sleep(sleep_time)

    print(f"finished scan, now moving back to {eng_list1[0]} keV")
    yield from mv(zps.sx, x_list[0], zps.sy, y_list[0], zps.sz, z_list[0])
    yield from move_zp_ccd(
        eng_list1[0], move_flag=1, move_clens_flag=1, move_det_flag=0
    )
    insert_text('finish "damon scan"')


def user_fly_scan(
    exposure_time=0.1,
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
    traditional_sequence_flag=1,
    filters=[],
    md=None,
):
    """
    Inputs:
    -------
    exposure_time: float, in unit of sec

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

    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    detectors = [Andor, ic3]
    offset_angle = -0.5 * rs
    current_rot_angle = zps.pi_r.position

    target_rot_angle = current_rot_angle + relative_rot_angle
    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motor],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exposure_time,
            "relative_rot_angle": relative_rot_angle,
            "period": period,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "rs": rs,
            "relative_move_flag": relative_move_flag,
            "traditional_sequence_flag": traditional_sequence_flag,
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

    #    yield from _set_andor_param(exposure_time=exposure_time, period=period, chunk_size=chunk_size)
    yield from _set_rotation_speed(rs=rs)
    print("set rotation speed: {} deg/sec".format(rs))

    @stage_decorator(list(detectors) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        # close shutter, dark images: numer=chunk_size (e.g.20)
        print("\nshutter closed, taking dark images...")
        yield from _set_andor_param(
            exposure_time=exposure_time, period=period, chunk_size=20
        )
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)
        yield from bps.sleep(1)
        yield from _set_andor_param(
            exposure_time=exposure_time, period=period, chunk_size=chunk_size
        )

        # open shutter, tomo_images
        yield from _open_shutter(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from mv(zps.pi_r, current_rot_angle + offset_angle)
        status = yield from abs_set(zps.pi_r, target_rot_angle, wait=False)
        yield from bps.sleep(1)
        while not status.done:
            yield from trigger_and_read(list(detectors) + motor)
        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=30)
        #        yield from abs_set(zps.pi_r.velocity, rs)
        for flt in filters:
            yield from mv(flt, 1)
            yield from mv(flt, 1)
        yield from bps.sleep(1)
        yield from _set_andor_param(
            exposure_time=exposure_time, period=period, chunk_size=20
        )
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
        yield from _close_shutter(simu=simu)
        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            trans_first_flag=traditional_sequence_flag,
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


def user_fly_only(
    exposure_time=0.1,
    end_rot_angle=180,
    period=0.15,
    chunk_size=20,
    rs=1,
    note="",
    simu=False,
    dark_scan_id=0,
    bkg_scan_id=0,
    md=None,
):

    global ZONE_PLATE
    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    detectors = [Andor, ic3]
    # offset_angle = 0 #-0.5 * rs * np.sign(relative_rot_angle)
    current_rot_angle = zps.pi_r.position

    target_rot_angle = end_rot_angle
    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motor],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exposure_time,
            "end_rot_angle": end_rot_angle,
            "period": period,
            "chunk_size": chunk_size,
            "rs": rs,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
            "dark_scan_id": dark_scan_id,
            "bkg_scan_id": bkg_scan_id,
        },
        "plan_name": "user_fly_only",
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

    yield from _set_andor_param(
        exposure_time=exposure_time, period=period, chunk_size=chunk_size
    )
    yield from _set_rotation_speed(rs=rs)
    print("set rotation speed: {} deg/sec".format(rs))

    @stage_decorator(list(detectors) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        yield from _open_shutter(simu=simu)
        status = yield from abs_set(zps.pi_r, target_rot_angle, wait=False)
        while not status.done:
            yield from trigger_and_read(list(detectors) + motor)

    uid = yield from fly_inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    print("scan finished")
    # yield from _set_rotation_speed(rs=30)
    txt = get_scan_parameter(print_flag=0)
    insert_text(txt)
    print(txt)
    return uid


def user_dark_only(exposure_time=0.1, chunk_size=20, note="", simu=False, md=None):
    """
    Take dark field images.

    Inputs:
    -------
    exposure_time: float, in unit of sec

    chunk_size: int, default setting is 20
        number of images taken for each trigger of Andor camera

    note: string
        adding note to the scan

    simu: Bool, default is False
        True: will simulate closing/open shutter without really closing/opening
        False: will really close/open shutter

    """
    global ZONE_PLATE
    period = exposure_time  # default to exposure time for backgrounds
    detectors = [Andor, ic3]
    motor = []

    _md = {
        "detectors": ["Andor"],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exposure_time,
            "chunk_size": chunk_size,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "user_dark_only",
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

    yield from _set_andor_param(
        exposure_time=exposure_time, period=period, chunk_size=chunk_size
    )

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def inner_scan():
        yield from _set_andor_param(
            exposure_time=exposure_time, period=period, chunk_size=chunk_size
        )
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

    uid = yield from inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    print("dark finished")
    txt = get_scan_parameter(print_flag=0)
    insert_text(txt)
    print(txt)
    return uid


def user_bkg_only(
    exposure_time=0.1,
    chunk_size=20,
    out_x=None,
    out_y=2000,
    out_z=None,
    out_r=None,
    note="",
    simu=False,
    relative_move_flag=1,
    traditional_sequence_flag=1,
    md=None,
):
    """
    Move sample out of the way and take background (aka flat) images.

    Inputs:
    -------
    exposure_time: float, in unit of sec

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

    note: string
        adding note to the scan

    simu: Bool, default is False
        True: will simulate closing/open shutter without really closing/opening
        False: will really close/open shutter

    """
    global ZONE_PLATE
    period = exposure_time  # default to exposure time for backgrounds
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

    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    detectors = [Andor, ic3]
    current_rot_angle = zps.pi_r.position

    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motor],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exposure_time,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "relative_move_flag": relative_move_flag,
            "traditional_sequence_flag": traditional_sequence_flag,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "user_bkg_only",
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

    # yield from _set_andor_param(exposure_time=exposure_time, period=period, chunk_size=chunk_size)

    @stage_decorator(list(detectors) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        yield from _open_shutter(simu=simu)
        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=30)
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
        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            trans_first_flag=traditional_sequence_flag,
        )

    uid = yield from fly_inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    print("bkg finished")
    txt = get_scan_parameter(print_flag=0)
    insert_text(txt)
    print(txt)
    return uid


def user_multiple_fly_scans(
    xyz_list,
    bkg_every_x_scans=10,
    exposure_time=0.1,
    angle=70,
    period=0.15,
    chunk_size=20,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    rs=1,
    note="",
    simu=False,
    relative_move_flag=0,
    traditional_sequence_flag=1,
    md=None,
):
    # first take dark field
    dark_scan_id = yield from user_dark_only(exposure_time, chunk_size, note, simu, md)
    # open shutter for rest of data taking
    yield from _open_shutter(simu=simu)
    print("\nshutter opened")

    bkg_index = 0
    bkg_scan_id = None
    for i, pos in enumerate(xyz_list):
        x, y, z = pos
        if i == 0 or bkg_index + bkg_every_x_scans <= i:
            # take background
            bkg_scan_id = yield from user_bkg_only(
                exposure_time,
                chunk_size,
                out_x,
                out_y,
                out_z,
                out_r,
                note,
                simu,
                relative_move_flag,
                traditional_sequence_flag,
                md,
            )
            bkg_index = i
        # mv x, y, z, r position
        yield from mv(zps.sx, x, zps.sy, y, zps.sz, z, zps.pi_r, angle)
        # take tomo
        angle *= -1  # rocker scan, switch angle back and forth
        while True:
            try:
                scan_id = yield from user_fly_only(
                    exposure_time,
                    angle,
                    period,
                    chunk_size,
                    rs,
                    note,
                    simu,
                    dark_scan_id,
                    bkg_scan_id,
                    md,
                )
                break
            except Exception as e:
                print(e)
    print("Finished scans %s - %s" % (dark_scan_id, scan_id))


def user_mosaic_gen(x_start, x_stop, x_step, y_start, y_stop, y_step, z_pos):
    xyz_list = []
    for y in range(y_start, y_stop + y_step, y_step):
        for x in range(x_start, x_stop + x_step, x_step):
            xyz_list.append((x, y, z_pos))
    return xyz_list


def user_hex_mosaic_xyz(
    x_start, x_stop, x_step, x_offset, y_start, y_stop, y_step, z_pos
):
    xyz_list = []
    # apply the x_offse every other row
    apply_offset = False
    for y in range(y_start, y_stop + y_step, y_step):
        if apply_offset:
            offset = x_offset
        else:
            offset = 0
        apply_offset = not apply_offset
        for x in range(x_start, x_stop + x_step, x_step):
            xyz_list.append((x + offset, y, z_pos))
    return xyz_list


def v4_z_offset(xyz_list):
    # offset is only dependent on y
    new_xyz_list = []
    for x, y, z in xyz_list:
        z = 50 + (56 - 50) * (-3873 - y) / (-1000)
        new_xyz_list.append((x, y, z))
    return new_xyz_list


def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def trim_points_to_polygon(xyz_list, poly):
    xyz_list_out = []
    for x, y, z in xyz_list:
        if point_inside_polygon(x, y, poly):
            xyz_list_out.append((x, y, z))
    return xyz_list_out


def ming_scan():
    x_list = [-892, -899, -865, -843, -782]
    y_list = [384, 437, 431, 427, 807]
    z_list = [-718, -726, -719, -715, -700]
    r_list = [0, 0, 0, 0, 0]
    out_x = -443
    out_y = 425
    out_z = -727
    out_r = 1
    # yield from multipos_2D_xanes_scan2(Ni_eng_list_short, x_list, y_list, z_list, r_list, out_x, out_y, out_z, out_r, repeat_num=1, exposure_time=0.04,  sleep_time=1, chunk_size=5, simu=False, relative_move_flag=0, note='P95_NMC_Ag_700_0.2C_20cy'))

    for i in range(4, 6):
        txt = f"start 3D xanes at pos {i}"
        insert_text(txt)
        print(txt)
        yield from mv(
            zps.sx, x_list[i], zps.sy, y_list[i], zps.sz, z_list[i], zps.pi_r, -80
        )
        yield from xanes_3D(
            eng_list=Ni_eng_list_in_situ,
            exposure_time=0.03,
            relative_rot_angle=160,
            period=0.03,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            rs=3,
            simu=False,
            relative_move_flag=0,
            traditional_sequence_flag=1,
            note=f"p95_700_Ag_0.2C_20cy_pos_{i+1}",
        )


def ming_scan2():
    yield from move_zp_ccd(6.5)
    yield from bps.sleep(5)
    yield from mv(zps.sx, -782, zps.sy, 807, zps.sz, -700, zps.pi_r, -80)
    yield from xanes_3D(
        eng_list=Mn_eng_list_in_situ,
        exposure_time=0.025,
        relative_rot_angle=160,
        period=0.025,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
        out_r=out_r,
        rs=6,
        simu=False,
        relative_move_flag=0,
        traditional_sequence_flag=1,
        note=f"p95_700_Ag_0.2C_20cy_pos_5_Mn",
    )
    print("start Co xanes")
    yield from move_zp_ccd(7.8)
    yield from mv(zps.sx, -782, zps.sy, 807, zps.sz, -700, zps.pi_r, -80)
    yield from bps.sleep(5)
    yield from xanes_3D(
        eng_list=Co_eng_list_in_situ,
        exposure_time=0.02,
        relative_rot_angle=160,
        period=0.025,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
        out_r=out_r,
        rs=6,
        simu=False,
        relative_move_flag=0,
        traditional_sequence_flag=1,
        note=f"p95_700_Ag_0.2C_20cy_pos_5_Co",
    )
    print("start Mn xanes")
    yield from move_zp_ccd(8.2)
    yield from mv(zps.sx, -782, zps.sy, 807, zps.sz, -700, zps.pi_r, -80)
    yield from bps.sleep(5)
    yield from xanes_3D(
        eng_list=Ni_eng_list_in_situ,
        exposure_time=0.015,
        relative_rot_angle=160,
        period=0.025,
        out_x=out_x,
        out_y=out_y,
        out_z=out_z,
        out_r=out_r,
        rs=6,
        simu=False,
        relative_move_flag=0,
        traditional_sequence_flag=1,
        note=f"p95_700_Ag_0.2C_20cy_pos_5_Ni",
    )


def ming_scan3():

    x_3D = [395, 513]
    y_3D = [1067, 756]
    z_3D = [-496, -508]
    r_3D = [-80, -80]

    yield from move_zp_ccd(8.2)
    yield from mv(zps.sx, x_3D[0], zps.sy, y_3D[0], zps.sz, z_3D[0], zps.pi_r, -80)
    yield from xanes_3D(
        eng_list=Ni_eng_list_in_situ,
        exposure_time=0.02,
        relative_rot_angle=160,
        period=0.02,
        out_x=2000,
        out_y=None,
        out_z=None,
        out_r=1,
        rs=6,
        simu=False,
        relative_move_flag=0,
        traditional_sequence_flag=1,
        note=f"p95_600_Ag_pristine_pos1",
    )

    yield from mv(zps.sx, x_3D[1], zps.sy, y_3D[1], zps.sz, z_3D[1], zps.pi_r, -80)
    yield from xanes_3D(
        eng_list=Ni_eng_list_in_situ,
        exposure_time=0.02,
        relative_rot_angle=160,
        period=0.02,
        out_x=2000,
        out_y=None,
        out_z=None,
        out_r=1,
        rs=6,
        simu=False,
        relative_move_flag=0,
        traditional_sequence_flag=1,
        note=f"p95_600_Ag_pristine_pos2",
    )


def qingchao_scan(
    eng_list,
    x_list1,
    y_list1,
    z_list1,
    r_list1,
    x_list2,
    y_list2,
    z_list2,
    r_list2,
    sleep_time=0,
    num=1,
):
    for i in range(num):
        print(f"repeat # {i}")
        for j in range(5):
            yield from mv(filter3, 1, filter4, 1)
        yield from multipos_2D_xanes_scan2(
            eng_list,
            x_list=x_list1,
            y_list=y_list1,
            z_list=z_list1,
            r_list=r_list1,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            repeat_num=1,
            exposure_time=0.1,
            sleep_time=sleep_time,
            chunk_size=5,
            relative_move_flag=True,
            note="622_filter3+4",
        )
        for j in range(5):
            yield from mv(filter3, 0, filter4, 1)
        yield from multipos_2D_xanes_scan2(
            eng_list,
            x_list=x_list2,
            y_list=y_list2,
            z_list=z_list2,
            r_list=r_list2,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            repeat_num=1,
            exposure_time=0.1,
            sleep_time=sleep_time,
            chunk_size=5,
            relative_move_flag=True,
            note="622_filter4",
        )
        print(f"slepp for {sleep_time} sec ...")
        yield from bps.sleep(sleep_time)



def ming():
    for i in range(2):
        yield from multipos_2D_xanes_scan2(Ni_list_2D,x_list,y_list,z_list,r_list,out_x=None,out_y=None,out_z=950,out_r=-90,exposure_time=0.1,repeat_num=3,sleep_time=600,relative_move_flag=0,chunk_size=5,simu=False,note='N83_insitu_pristine_filter_2+3+4')
        yield from movpos(2, x_list, y_list, z_list, r_list)
        yield from mv(zps.pi_r, -70)
        yield from xanes_3D(Ni_list_3D, exposure_time=0.1, relative_rot_angle=140, period=0.1, out_x=None, out_y=None, out_z=2500, out_r=-20, rs=3, simu=False, relative_move_flag=1, note='N83_pos2')
        yield from mv(zps.pi_r, 0)

        yield from multipos_2D_xanes_scan2(Ni_list_2D,x_list,y_list,z_list,r_list,out_x=None,out_y=None,out_z=950,out_r=-90,exposure_time=0.1,repeat_num=3,sleep_time=600,relative_move_flag=0,chunk_size=5,simu=False,note='N83_insitu_pristine_filter_2+3+4')
        yield from movpos(4, x_list, y_list, z_list, r_list)
        yield from mv(zps.pi_r, -70)
        yield from xanes_3D(Ni_list_3D, exposure_time=0.1, relative_rot_angle=145, period=0.1, out_x=None, out_y=None, out_z=2500, out_r=-20, rs=3, simu=False, relative_move_flag=1, note='N83_pos4')
        yield from mv(zps.pi_r, 0)

    insert_text('take xanes of full_eng_list')
    for i in range(1):
        yield from multipos_2D_xanes_scan2(Ni_eng_list_63pnt,x_list,y_list,z_list,r_list,out_x=None,out_y=None,out_z=950,out_r=-90,exposure_time=0.1,repeat_num=3,sleep_time=600,relative_move_flag=0,chunk_size=5,simu=False,note='N83_insitu_pristine_filter_2+3+4')

        for j in range(4):
            insert_text(f'taking 3D xanes at pos{j}\n')
            yield from movpos(j, x_list, y_list, z_list, r_list)
            yield from mv(zps.pi_r, -70)
            yield from xanes_3D(Ni_list_3D, exposure_time=0.1, relative_rot_angle=140, period=0.1, out_x=None, out_y=None, out_z=2500, out_r=-20, rs=3, simu=False, relative_move_flag=1, note=f'N83_pos{j}')
            yield from mv(zps.pi_r, 0)

    for i in range(4):
        yield from multipos_2D_xanes_scan2(Ni_list_2D,x_list,y_list,z_list,r_list,out_x=None,out_y=None,out_z=950,out_r=-90,exposure_time=0.1,repeat_num=3,sleep_time=600,relative_move_flag=0,chunk_size=5,simu=False,note='N83_insitu_pristine_filter_2+3+4')
        yield from movpos(2, x_list, y_list, z_list, r_list)
        yield from mv(zps.pi_r, -70)
        yield from xanes_3D(Ni_list_3D, exposure_time=0.1, relative_rot_angle=140, period=0.1, out_x=None, out_y=None, out_z=2500, out_r=-20, rs=3, simu=False, relative_move_flag=1, note='N83_pos2')
        yield from mv(zps.pi_r, 0)

        yield from multipos_2D_xanes_scan2(Ni_list_2D,x_list,y_list,z_list,r_list,out_x=None,out_y=None,out_z=950,out_r=-90,exposure_time=0.1,repeat_num=3,sleep_time=600,relative_move_flag=0,chunk_size=5,simu=False,note='N83_insitu_pristine_filter_2+3+4')
        yield from movpos(4, x_list, y_list, z_list, r_list)
        yield from mv(zps.pi_r, -70)
        yield from xanes_3D(Ni_list_3D, exposure_time=0.1, relative_rot_angle=145, period=0.1, out_x=None, out_y=None, out_z=2500, out_r=-20, rs=3, simu=False, relative_move_flag=1, note='N83_pos4')
        yield from mv(zps.pi_r, 0)



def scan_change_expo_time(x_range, y_range, t1, t2, out_x=None, out_y=None, out_z=None, out_r=None, img_sizeX=2560, img_sizeY=2160, pxl=20, relative_move_flag=1, note='', simu=False, sleep_time=0, md=None):
    '''
    take image
    '''
    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    detectors = [Andor, ic3]

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
    motor_eng = XEng
    motor = [motor_eng, zps.sx, zps.sy, zps.sz, zps.pi_r]

    _md = {
        "detectors": [det.name for det in detectors],
        "x_ray_energy": XEng.position,
        "plan_args": {
            "x_range": x_range,
            "y_range": y_range,
            "t1": t1,
            "t2": t2,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "img_sizeX": img_sizeX,
            "img_sizeY": img_sizeY,
            "pxl": pxl,
            "relative_move_flag": relative_move_flag,
            "note": note if note else "None",
            "sleep_time": sleep_time,
        },
        "plan_name": "scan_change_expo_time",
        "hints": {},
        "operator": "FXI",
        "zone_plate": ZONE_PLATE,
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
    def inner():
        # take dark image
        print(f"take 5 dark image with exposure = {t1}")
        yield from _set_andor_param(exposure_time=t1, period=t1, chunk_size=1)
        yield from _take_dark_image(detectors, motor, num_dark=5, simu=simu)
        print(f"take 5 dark image with exposure = {t2}")
        yield from _set_andor_param(exposure_time=t2, period=t2, chunk_size=1)
        yield from _take_dark_image(detectors, motor, num_dark=5, simu=simu)

        print("open shutter ...")
        yield from _open_shutter(simu)
        for ii in np.arange(x_range[0], x_range[1] + 1):
            for jj in np.arange(y_range[0], y_range[1] + 1):
                yield from mv(zps.sx, motor_x_ini + ii * img_sizeX * pxl * 1.0 / 1000)
                yield from mv(zps.sy, motor_y_ini + jj * img_sizeY * pxl * 1.0 / 1000)
                yield from bps.sleep(0.1)
                print(f'set exposure time = {t1}')
                yield from _set_andor_param(exposure_time=t1, period=t1, chunk_size=1)
                yield from bps.sleep(sleep_time)
                yield from _take_image(detectors, motor, 1)
                print(f'set exposure time = {t2}')
                yield from _set_andor_param(exposure_time=t2, period=t2, chunk_size=1)
                yield from bps.sleep(sleep_time)
                yield from _take_image(detectors, motor, 1)
                print(f'take bkg image with exposure time = {t1}')
                yield from _set_andor_param(exposure_time=t1, period=t1, chunk_size=1)
                yield from bps.sleep(sleep_time)
                yield from _take_bkg_image(motor_x_out, motor_y_out, motor_z_out, motor_r_out,
                                           detectors, motor, num_bkg=5, simu=simu)
                print(f'take bkg image with exposure time = {t2}')
                yield from _set_andor_param(exposure_time=t2, period=t2, chunk_size=1)
                yield from bps.sleep(sleep_time)
                yield from _take_bkg_image(motor_x_out, motor_y_out, motor_z_out, motor_r_out,
                                           detectors, motor, num_bkg=5, simu=simu)

        yield from _move_sample_in(motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini,
                                            repeat=1, trans_first_flag=0)

        print("closing shutter")
        yield from _close_shutter(simu)

    yield from inner()
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)
