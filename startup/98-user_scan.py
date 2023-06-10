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
from ophyd.sim import SynAxis


def select_filters(flts=[]):
    for key, item in FILTERS.items():
        yield from mv(item, 0)
    if flts:
        for ii in flts:
            yield from mv(FILTERS["filter" + str(ii)], 1)


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

    dets = [Andor, ic3]
    taxi_ang = -2.0 * rs
    cur_rot_ang = zps.pi_r.position

    #  tgt_rot_ang = cur_rot_ang + rel_rot_ang
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

    @stage_decorator(list(dets) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def inner_scan():
        # close shutter, dark images: numer=chunk_size (e.g.20)
        print("\nshutter closed, taking dark images...")
        yield from _take_dark_image(dets, motor, num_dark=1, simu=simu)

        yield from mv(zps.pi_x, 0)
        yield from mv(zps.pi_r, -50)
        yield from _set_rotation_speed(rs=rs)
        # open shutter, tomo_images
        yield from _open_shutter(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from mv(zps.pi_r, -50 + taxi_ang)
        status = yield from abs_set(zps.pi_r, 50, wait=False)
        yield from bps.sleep(2)
        while not status.done:
            yield from trigger_and_read(list(dets) + motor)
        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=30)
        yield from mv(zps.pi_r, 0)

        yield from mv(zps.pi_x, 12)
        yield from mv(zps.pi_r, 70)
        yield from trigger_and_read(list(dets) + motor)

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
    rel_rot_ang=150,
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
    x_ini = zps.sx.position
    y_ini = zps.sy.position
    z_ini = zps.sz.position
    r_ini = zps.pi_r.position
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
    _move_sample_in(x_ini, y_ini, z_ini, r_ini, repeat=1, trans_first_flag=1)
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
    rel_rot_ang=182,
    rs=2,
):
    """
    the sample_out position is in its absolute value:
    will move sample to out_x (um) out_y (um) out_z(um) and out_r (um) to take background image

    to run:

    RE(multi_pos_3D_xanes(Ni_eng_list, x_list=[a, b, c], y_list=[aa,bb,cc], z_list=[aaa,bbb, ccc], r_list=[0, 0, 0], exposure_time=0.05, rel_rot_ang=185, rs=3, out_x=1500, out_y=-1500, out_z=-770, out_r=0, note='NC')
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
            relative_rot_angle=rel_rot_ang,
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


def _open_shutter_xhx(simu=False):
    if simu:
        print("testing: open shutter")
    else:
        print("opening shutter ... ")
        i = 0
        reading = yield from bps.rd(shutter_status)
        while reading:  # if 1:  closed; if 0: open
            if i > 5:
                yield from abs_set(shutter_close, 1, wait=True)
                yield from bps.sleep(1)
            elif i > 10:
                print("fails to open shutter")
                raise Exception("fails to open shutter")
                break
            yield from abs_set(shutter_open, 1, wait=True)
            print(f"try opening {i} time(s) ...")
            yield from bps.sleep(1)
            i += 1
            reading = yield from bps.rd(shutter_status)


def _close_shutter_xhx(simu=False):
    if simu:
        print("testing: close shutter")
    else:
        print("closing shutter ... ")
        i = 0
        reading = yield from bps.rd(shutter_status)
        while not reading:  # if 1:  closed; if 0: open
            if i > 5:
                yield from abs_set(shutter_open, 1, wait=True)
                yield from bps.sleep(1)
            elif i > 10:
                print("fails to close shutter")
                raise Exception("fails to close shutter")
                break
            yield from abs_set(shutter_close, 1, wait=True)
            yield from bps.sleep(1)
            i += 1
            print(f"try closing {i} time(s) ...")
            reading = yield from bps.rd(shutter_status)


def _xanes_3D_xh(
    eng_list,
    exposure_time=0.05,
    start_angle=None,
    relative_rot_angle=180,
    period=0.06,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    rs=4,
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    note="",
    binning=1,
    flts=[],
    enable_z=True,
):
    for eng in eng_list:
        yield from move_zp_ccd(eng, move_flag=1)
        my_note = f"{note}@energy={eng}"
        yield from bps.sleep(1)
        print(f"current energy: {eng}")

        yield from fly_scan2(
            exposure_time,
            start_angle,
            relative_rot_angle,
            period=period,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            rs=rs,
            relative_move_flag=relative_move_flag,
            note=my_note,
            simu=simu,
            rot_first_flag=rot_first_flag,
            flts=flts,
            binning=binning,
            enable_z=enable_z,
        )
        yield from bps.sleep(1)
    yield from mv(Andor.cam.image_mode, 1)


def _multi_pos_xanes_3D_xh(
    eng_list,
    x_list,
    y_list,
    z_list,
    r_list,
    start_angle=None,
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
    rot_first_flag=1,
    note="",
    sleep_time=0,
    binning=1,
    flts=[],
    repeat=1,
    ref_flat_scan=False,
    enable_z=True,
):
    yield from select_filters(flts)
    # yield from _bin_cam(binning)
    n = len(x_list)
    for rep in range(repeat):
        for i in range(n):
            if x_list[i] is None:
                x_list[i] = zps.sx.position
            if y_list[i] is None:
                y_list[i] = zps.sy.position
            if z_list[i] is None:
                z_list[i] = zps.sz.position
            if r_list[i] is None:
                r_list[i] = zps.pi_r.position
            yield from _move_sample_in_xhx(
                x_list[i],
                y_list[i],
                z_list[i],
                r_list[i],
                repeat=2,
                trans_first_flag=1,
                enable_z=enable_z,
            )
            yield from _xanes_3D_xh(
                eng_list,
                exposure_time=exposure_time,
                start_angle=start_angle,
                relative_rot_angle=relative_rot_angle,
                period=period,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                out_r=out_r,
                rs=rs,
                simu=simu,
                relative_move_flag=relative_move_flag,
                rot_first_flag=rot_first_flag,
                note=note,
                binning=binning,
                enable_z=enable_z,
            )
            if ref_flat_scan:
                motor_x_ini = zps.sx.position
                motor_y_ini = zps.sy.position
                motor_z_ini = zps.sz.position
                motor_r_ini = zps.pi_r.position

                if relative_move_flag:
                    motor_x_out = (
                        motor_x_ini + out_x if not (out_x is None) else motor_x_ini
                    )
                    motor_y_out = (
                        motor_y_ini + out_y if not (out_y is None) else motor_y_ini
                    )
                    motor_z_out = (
                        motor_z_ini + out_z if not (out_z is None) else motor_z_ini
                    )
                    motor_r_out = (
                        motor_r_ini + out_r if not (out_r is None) else motor_r_ini
                    )
                else:
                    motor_x_out = out_x if not (out_x is None) else motor_x_ini
                    motor_y_out = out_y if not (out_y is None) else motor_y_ini
                    motor_z_out = out_z if not (out_z is None) else motor_z_ini
                    motor_r_out = out_r if not (out_r is None) else motor_r_ini

                yield from _move_sample_out_xhx(
                    motor_x_out,
                    motor_y_out,
                    motor_z_out,
                    motor_r_out,
                    repeat=2,
                    rot_first_flag=1,
                    enable_z=enable_z,
                )
                for ii in [
                    eng_list[-1],
                    eng_list[int(eng_list.shape[0] / 2)],
                    eng_list[0],
                ]:
                    yield from move_zp_ccd(ii, move_flag=1)
                    my_note = note + f"_ref_flat@energy={ii}_keV"
                    yield from bps.sleep(1)
                    print(f"current energy: {ii}")

                    if period:
                        yield from fly_scan2(
                            exposure_time,
                            start_angle=start_angle,
                            relative_rot_angle=relative_rot_angle,
                            period=period,
                            out_x=None,
                            out_y=None,
                            out_z=None,
                            out_r=None,
                            rs=rs,
                            relative_move_flag=relative_move_flag,
                            note=my_note,
                            simu=simu,
                            rot_first_flag=1,
                            binning=binning,
                            flts=flts,
                            enable_z=enable_z,
                        )
                    else:
                        print(f"invalid binning {binning}")
                yield from _move_sample_in_xhx(
                    motor_x_ini,
                    motor_y_ini,
                    motor_z_ini,
                    motor_r_ini,
                    repeat=2,
                    trans_first_flag=1,
                    enable_z=enable_z,
                )
        print(f"sleep for {sleep_time} sec\n\n\n\n")
        yield from bps.sleep(sleep_time)


def _xanes_3D_zebra_xh(
    eng_list,
    exp_t=0.05,
    acq_p=0.06,
    ang_s=0,
    ang_e=180,
    vel=4,
    acc_t=1,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    simu=False,
    rel_out_flag=1,
    note="",
    bin_fac=1,
    flts=[],
    cam=Andor,
    flyer=tomo_flyer,
):
    for eng in eng_list:
        yield from move_zp_ccd(eng, move_flag=1)
        print(121)
        my_note = f"{note}@energy={eng}"
        yield from bps.sleep(1)
        print(f"current energy: {eng}")

        yield from tomo_zfly(
            scn_mode=0,
            exp_t=exp_t,
            acq_p=acq_p,
            ang_s=ang_s,
            ang_e=ang_e,
            vel=vel,
            acc_t=acc_t,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            rel_out_flag=rel_out_flag,
            flts=flts,
            rot_back_velo=30,
            bin_fac=bin_fac,
            note=my_note,
            md=None,
            simu=simu,
            cam=cam,
            flyer=flyer,
            num_swing=1,
        )


def _multi_pos_xanes_3D_zebra_xh(
    eng_list,
    x_list,
    y_list,
    z_list,
    r_list,
    exp_t=0.05,
    acq_p=0.05,
    ang_s=0,
    ang_e=180,
    vel=2,
    acc_t=1,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    simu=False,
    rel_out_flag=1,
    note="",
    sleep_time=0,
    bin_fac=1,
    flts=[],
    repeat=1,
    ref_flat_scan=False,
    cam=Andor,
    flyer=tomo_flyer,
):
    yield from select_filters(flts)
    print(11)
    n = len(x_list)
    for rep in range(repeat):
        for i in range(n):
            if x_list[i] is None:
                x_list[i] = zps.sx.position
            if y_list[i] is None:
                y_list[i] = zps.sy.position
            if z_list[i] is None:
                z_list[i] = zps.sz.position
            if r_list[i] is None:
                r_list[i] = zps.pi_r.position
            yield from _move_sample_in_xhx(
                x_list[i],
                y_list[i],
                z_list[i],
                r_list[i],
                repeat=2,
                trans_first_flag=1,
                enable_z=True,
            )
            print(12)
            yield from _xanes_3D_zebra_xh(
                eng_list,
                exp_t=exp_t,
                acq_p=acq_p,
                ang_s=ang_s,
                ang_e=ang_e,
                vel=vel,
                acc_t=acc_t,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                out_r=out_r,
                simu=simu,
                rel_out_flag=rel_out_flag,
                note=note,
                bin_fac=bin_fac,
                flts=flts,
                cam=cam,
                flyer=flyer,
            )

            if ref_flat_scan:
                motor_x_ini = zps.sx.position
                motor_y_ini = zps.sy.position
                motor_z_ini = zps.sz.position
                motor_r_ini = zps.pi_r.position

                if rel_out_flag:
                    motor_x_out = (
                        motor_x_ini + out_x if not (out_x is None) else motor_x_ini
                    )
                    motor_y_out = (
                        motor_y_ini + out_y if not (out_y is None) else motor_y_ini
                    )
                    motor_z_out = (
                        motor_z_ini + out_z if not (out_z is None) else motor_z_ini
                    )
                    motor_r_out = (
                        motor_r_ini + out_r if not (out_r is None) else motor_r_ini
                    )
                else:
                    motor_x_out = out_x if not (out_x is None) else motor_x_ini
                    motor_y_out = out_y if not (out_y is None) else motor_y_ini
                    motor_z_out = out_z if not (out_z is None) else motor_z_ini
                    motor_r_out = out_r if not (out_r is None) else motor_r_ini

                yield from _move_sample_out_xhx(
                    motor_x_out,
                    motor_y_out,
                    motor_z_out,
                    motor_r_out,
                    repeat=2,
                    rot_first_flag=1,
                    enable_z=True,
                )
                for ii in [
                    eng_list[-1],
                    eng_list[int(eng_list.shape[0] / 2)],
                    eng_list[0],
                ]:
                    yield from move_zp_ccd(ii, move_flag=1)
                    my_note = f"{note}_ref_flat@energy={ii}keV"
                    yield from bps.sleep(1)
                    print(f"current energy: {ii}")

                    yield from tomo_zfly(
                        scn_mode=0,
                        exp_t=exp_t,
                        acq_p=acq_p,
                        ang_s=ang_s,
                        ang_e=ang_e,
                        vel=vel,
                        acc_t=acc_t,
                        out_x=0,
                        out_y=0,
                        out_z=0,
                        out_r=0,
                        rel_out_flag=True,
                        flts=flts,
                        rot_back_velo=30,
                        bin_fac=bin_fac,
                        note=my_note,
                        md=None,
                        simu=simu,
                        cam=cam,
                        flyer=flyer,
                        num_swing=1,
                    )

                yield from _move_sample_in_xhx(
                    motor_x_ini,
                    motor_y_ini,
                    motor_z_ini,
                    motor_r_ini,
                    repeat=2,
                    trans_first_flag=1,
                    enable_z=True,
                )
        print(f"sleep for {sleep_time} sec\n\n\n\n")
        yield from bps.sleep(sleep_time)


def _multi_pos_xanes_2D_xh(
    eng_list,
    x_list,
    y_list,
    z_list,
    r_list,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    repeat_num=1,
    exposure_time=0.2,
    sleep_time=1,
    chunk_size=5,
    simu=False,
    relative_move_flag=True,
    note="",
    md=None,
    binning=0,
    flts=[],
    enable_z=True,
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
    yield from select_filters(flts)
    # yield from _bin_cam(binning)

    detectors = [Andor, ic3, ic4]
    # print(f"{exposure_time=}")
    # period = yield from _exp_t_sanity_check(exposure_time, binning=binning)
    # print('444:', period)
    period = exposure_time
    yield from mv(Andor.cam.acquire, 0)
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

    if enable_z:
        motor = [XEng, zps.sx, zps.sy, zps.sz, zps.pi_r]
    else:
        motor = [XEng, zps.sx, zps.sy, zps.pi_r]

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
            "period": period,
            "binning": binning,
            "filters": ["filter{}".format(t) for t in flts] if flts else "None",
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
        num = len(x_list)  # num of position point
        print(f"\ntake {chunk_size} dark images...")
        yield from _take_dark_image_xhx(
            detectors,
            motor,
            num=1,
            chunk_size=chunk_size,
            stream_name="dark",
            simu=False,
        )
        yield from bps.sleep(1)

        # start repeating xanes scan
        print(
            f"\nopening shutter, and start xanes scan: {chunk_size} images per each energy... "
        )

        yield from _open_shutter_xhx(simu)
        for rep in range(repeat_num):
            print(f"repeat multi-pos xanes scan #{rep}")
            for eng in eng_list:
                yield from move_zp_ccd(eng, move_flag=1, info_flag=0)
                yield from _open_shutter_xhx(simu)
                # take image at multiple positions
                for i in range(num):
                    yield from _move_sample_in_xhx(
                        x_list[i],
                        y_list[i],
                        z_list[i],
                        r_list[i],
                        repeat=2,
                        trans_first_flag=1,
                        enable_z=enable_z,
                    )
                    yield from trigger_and_read(list(detectors) + motor)
                yield from _take_bkg_image_xhx(
                    motor_x_out,
                    motor_y_out,
                    motor_z_out,
                    motor_r_out,
                    detectors,
                    motor,
                    num=1,
                    chunk_size=chunk_size,
                    stream_name="flat",
                    simu=simu,
                    enable_z=enable_z,
                )
                yield from _move_sample_in_xhx(
                    motor_x_ini,
                    motor_y_ini,
                    motor_z_ini,
                    motor_r_ini,
                    repeat=2,
                    trans_first_flag=1,
                    enable_z=enable_z,
                )
            # end of eng_list
            # close shutter and sleep
            yield from _close_shutter_xhx(simu)
            # sleep
            if rep < repeat_num - 1:
                print(f"\nsleep for {sleep_time} seconds ...")
                yield from bps.sleep(sleep_time)

    yield from inner_scan()


def _mk_eng_list(elem, bulk=False):
    if bulk:
        eng_list = np.genfromtxt(
            "/nsls2/data/fxi-new/shared/config/xanes_ref/"
            + elem.split("_")[0]
            + "/eng_list_"
            + elem.split("_")[0]
            + "_xanes_standard_dense.txt"
        )
    else:
        if elem.split("_")[-1] == "wl":
            eng_list = np.genfromtxt(
                "/nsls2/data/fxi-new/shared/config/xanes_ref/"
                + elem.split("_")[0]
                + "/eng_list_"
                + elem.split("_")[0]
                + "_xanes_standard_21pnt.txt"
            )
        if elem.split("_")[-1] == "41":
            eng_list = np.genfromtxt(
                "/nsls2/data/fxi-new/shared/config/xanes_ref/"
                + elem.split("_")[0]
                + "/eng_list_"
                + elem.split("_")[0]
                + "_xanes_standard_41pnt.txt"
            )
        if elem.split("_")[-1] == "83":
            eng_list = np.genfromtxt(
                "/nsls2/data/fxi-new/shared/config/xanes_ref/"
                + elem.split("_")[0]
                + "/eng_list_"
                + elem.split("_")[0]
                + "_xanes_standard_83pnt.txt"
            )
        elif elem.split("_")[-1] == "101":
            eng_list = np.genfromtxt(
                "/nsls2/data/fxi-new/shared/config/xanes_ref/"
                + elem.split("_")[0]
                + "/eng_list_"
                + elem.split("_")[0]
                + "_xanes_standard_101pnt.txt"
            )
        elif elem.split("_")[-1] == "63":
            eng_list = np.genfromtxt(
                "/nsls2/data/fxi-new/shared/config/xanes_ref/"
                + elem.split("_")[0]
                + "/eng_list_"
                + elem.split("_")[0]
                + "_xanes_standard_63pnt.txt"
            )
        elif elem.split("_")[-1] == "diff":
            eng_list = np.genfromtxt(
                "/nsls2/data/fxi-new/shared/config/xanes_ref/"
                + elem.split("_")[0]
                + "/eng_list_"
                + elem.split("_")[0]
                + "_xanes_standard_diff.txt"
            )
    return eng_list


def _exp_t_sanity_check(exp_t, binning=None):
    if binning is None:
        binning = 0
    if binning == 0:  # 1x1
        # print('000')
        if exp_t < 0.05:
            period = 0.05
            # print('111')
        else:
            period = exp_t
            # print('222')
    elif binning == 1:  # 2x2
        if exp_t < 0.025:
            period = 0.025
        else:
            period = exp_t
    elif binning == 2:  # 3x3
        if exp_t < 0.017:
            period = 0.017
        else:
            period = exp_t
    elif binning == 3:  # 4x4
        if exp_t < 0.0125:
            period = 0.0125
        else:
            period = exp_t
    elif binning == 4:  # 8x8
        if exp_t < 0.00625:
            period = 0.00625
        else:
            period = exp_t
    else:
        period = None
    # print('333:', period)
    return period


def _bin_cam(binning, cam=Andor):
    yield from abs_set(cam.cam.acquire, 0, wait=True)
    if binning is None:
        binning = 0
    if int(binning) not in [0, 1, 2, 3, 4]:
        raise ValueError("binnng must be in [0, 1, 2, 3, 4]")
    yield from abs_set(cam.binning, binning, wait=True)
    yield from abs_set(cam.cam.image_mode, 0, wait=True)
    yield from abs_set(cam.cam.num_images, 5, wait=True)
    yield from abs_set(cam.cam.acquire, 1, wait=True)
    yield from abs_set(cam.cam.acquire, 0, wait=True)
    return int(binning)


def _sort_in_pos(in_pos_list):
    x_list = []
    y_list = []
    z_list = []
    r_list = []
    for ii in range(len(in_pos_list)):
        x_list.append(
            zps.sx.position if in_pos_list[ii][0] is None else in_pos_list[ii][0]
        )
        y_list.append(
            zps.sy.position if in_pos_list[ii][1] is None else in_pos_list[ii][1]
        )
        z_list.append(
            zps.sz.position if in_pos_list[ii][2] is None else in_pos_list[ii][2]
        )
        r_list.append(
            zps.pi_r.position if in_pos_list[ii][3] is None else in_pos_list[ii][3]
        )
    return (x_list, y_list, z_list, r_list)


def _move_sample_out_xhx(
    out_x, out_y, out_z, out_r, repeat=1, rot_first_flag=1, enable_z=False
):
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

    if enable_z:
        for i in range(repeat):
            if rot_first_flag:
                yield from mv(zps.pi_r, r_out)
                yield from mv(zps.sx, x_out, zps.sy, y_out, zps.sz, z_out)
            else:
                yield from mv(zps.sx, x_out, zps.sy, y_out, zps.sz, z_out)
                yield from mv(zps.pi_r, r_out)
    else:
        for i in range(repeat):
            if rot_first_flag:
                yield from mv(zps.pi_r, r_out)
                yield from mv(zps.sx, x_out, zps.sy, y_out)
            else:
                yield from mv(zps.sx, x_out, zps.sy, y_out)
                yield from mv(zps.pi_r, r_out)


def _move_sample_in_xhx(
    in_x, in_y, in_z, in_r, repeat=1, trans_first_flag=1, enable_z=False
):
    """
    move in at absolute position
    """
    if enable_z:
        for i in range(repeat):
            if trans_first_flag:
                yield from mv(zps.sx, in_x, zps.sy, in_y, zps.sz, in_z)
                yield from mv(zps.pi_r, in_r)
            else:
                yield from mv(zps.pi_r, in_r)
                yield from mv(zps.sx, in_x, zps.sy, in_y, zps.sz, in_z)
    else:
        for i in range(repeat):
            if trans_first_flag:
                yield from mv(zps.sx, in_x, zps.sy, in_y)
                yield from mv(zps.pi_r, in_r)
            else:
                yield from mv(zps.pi_r, in_r)
                yield from mv(zps.sx, in_x, zps.sy, in_y)


def _take_dark_image_xhx(
    detectors, motor, num=1, chunk_size=1, stream_name="dark", simu=False, cam=Andor
):
    yield from _close_shutter_xhx(simu)
    original_num_images = yield from rd(cam.cam.num_images)
    yield from _set_Andor_chunk_size_xhx(detectors, chunk_size, cam)
    yield from _take_image(detectors, [], num, stream_name=stream_name)
    yield from _set_Andor_chunk_size_xhx(detectors, original_num_images, cam)


def _take_bkg_image_xhx(
    out_x,
    out_y,
    out_z,
    out_r,
    detectors,
    motor,
    num=1,
    chunk_size=1,
    rot_first_flag=1,
    stream_name="flat",
    simu=False,
    enable_z=False,
    cam=Andor,
):
    yield from _move_sample_out_xhx(
        out_x,
        out_y,
        out_z,
        out_r,
        repeat=2,
        rot_first_flag=rot_first_flag,
        enable_z=enable_z,
    )
    original_num_images = yield from rd(cam.cam.num_images)
    yield from _set_Andor_chunk_size_xhx(detectors, chunk_size, cam)
    yield from _take_image(detectors, [], num, stream_name=stream_name)
    yield from _set_Andor_chunk_size_xhx(detectors, original_num_images, cam)


def _set_Andor_chunk_size_xhx(detectors, chunk_size, cam):
    for detector in detectors:
        yield from unstage(detector)
    yield from bps.configure(cam, {"cam.num_images": chunk_size})
    for detector in detectors:
        yield from stage(detector)


def _set_andor_param_xhx(
    exposure_time=0.1, period=0.1, chunk_size=1, binning=[1, 1], cam=Andor
):
    yield from abs_set(cam.cam.acquire, 0, wait=True)
    yield from abs_set(cam.cam.image_mode, 0, wait=True)
    yield from abs_set(cam.cam.num_images, chunk_size, wait=True)
    period_cor = period
    yield from abs_set(cam.cam.acquire_time, exposure_time, wait=True)
    yield from abs_set(cam.cam.acquire_period, period_cor, wait=True)


def multi_edge_xanes_zebra(
    elems=["Ni_wl"],
    scan_type="3D",
    flts={"Ni_filters": [1, 2, 3]},
    exp_t={"Ni_exp": 0.05},
    acq_p={"Ni_period": 0.05},
    ang_s=0,
    ang_e=180,
    vel=6,
    acc_t=1,
    in_pos_list=[[None, None, None, None]],
    out_pos=[None, None, None, None],
    note="",
    rel_out_flag=0,
    bin_fac=None,
    bulk=False,
    bulk_intgr=10,
    simu=False,
    sleep=0,
    repeat=None,
    ref_flat_scan=False,
    cam=Andor,
    flyer=tomo_flyer
):
    print(-2)
    yield from mv(cam.cam.acquire, 0)
    if repeat is None:
        repeat = 1
    repeat = int(repeat)
    print(-1)
    
    if scan_type == "2D":
        if bin_fac is None:
            bin_fac = 0
        # binning = yield from _bin_cam(binning)

        x_list, y_list, z_list, r_list = _sort_in_pos(in_pos_list)
        for elem in elems:
            for key in flts.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    flt = flts[key]
                    break
                else:
                    flt = []
            for key in exp_t.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    print(f"{exp_t[key]=}")
                    exposure = exp_t[key]
                    print(elem, exposure)
                    break
                else:
                    exposure = 0.05
                    print("use default exposure time 0.05s")
            for key in period_time.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    yield from mv(Andor.cam.acquire_time, exposure)
                    yield from bps.sleep(2)
                    # yield from mv(Andor.cam.acquire_period, period_time[key])
                    # period = yield from rd(Andor.cam.acquire_period)
                    # period = yield from _exp_t_sanity_check(period, binning=binning)
                    # print(elem, f"{period=}")
                    break
            eng_list = _mk_eng_list(elem, bulk=False)
            print(f"{out_pos=}")
            yield from _multi_pos_xanes_2D_xh(
                eng_list,
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
                relative_move_flag=rel_out_flag,
                note=note,
                md=None,
                sleep_time=sleep,
                repeat_num=repeat,
                binning=bin_fac,
                flts=flt,
                enable_z=True,
            )
    elif scan_type == "3D":
        print(-0.9)
        if bin_fac is None:
            bin_fac = 1
        bin_fac = yield from _bin_cam(bin_fac)
        print(-0.8)

        set_and_wait(Andor.cam.image_mode, 0)
        print(-0.89)
        set_and_wait(Andor.cam.num_images, 5)
        print(-0.88)
        yield from abs_set(Andor.cam.acquire, 1, wait=False)
        print(-0.7)

        x_list, y_list, z_list, r_list = _sort_in_pos(in_pos_list)
        print(0)
        for elem in elems:
            for key in flts.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    flt = flts[key]
                    break
                else:
                    flt = []
            print(0.1)
            for key in exp_t.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    exposure = exp_t[key]
                    print(elem, exposure)
                    break
                else:
                    exposure = 0.05
                    print("use default exposure time 0.05s")
            print(0.2)
            for key in acq_p.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    acquire_period = acq_p[key]
                    break
            print(0.3)
            eng_list = _mk_eng_list(elem, bulk=False)

            print(1)
            yield from _multi_pos_xanes_3D_zebra_xh(
                eng_list,
                x_list,
                y_list,
                z_list,
                r_list,
                exp_t=exposure,
                acq_p=acquire_period,
                ang_s=ang_s,
                ang_e=ang_e,
                vel=vel,
                acc_t=acc_t,
                out_x=out_pos[0],
                out_y=out_pos[1],
                out_z=out_pos[2],
                out_r=out_pos[3],
                simu=simu,
                rel_out_flag=rel_out_flag,
                note=note,
                sleep_time=sleep,
                bin_fac=bin_fac,
                flts=flt,
                repeat=1,
                ref_flat_scan=ref_flat_scan,
                cam=cam,
                flyer=flyer,
            )
            
            if bulk:
                eng_list = _mk_eng_list(elem, bulk=True)
                zpx = zp.x.position
                apx = aper.x.position
                cdx = clens.x.position
                yield from mvr(zp.x, -6500, clens.x, 6500, aper.x, -4000)
                xxanes_scan(eng_list, delay_time=0.2, intgr=bulk_intgr, note=note)
                yield from mv(clens.x, cdx, aper.x, apx, zp.x, zpx)

    else:
        print("wrong scan type")


def multi_edge_xanes(
    elements=["Ni_wl"],
    scan_type="3D",
    flts={"Ni_filters": [1, 2, 3]},
    exposure_time={"Ni_exp": 0.05},
    rel_rot_ang=185,
    rs=6,
    in_pos_list=[[None, None, None, None]],
    out_pos=[None, None, None, None],
    chunk_size=5,
    note="",
    relative_move_flag=0,
    binning=None,
    simu=False,
    ref_flat_scan=False,
    enable_z=True,
):
    yield from mv(Andor.cam.acquire, 0)
    x_list, y_list, z_list, r_list = _sort_in_pos(in_pos_list)
    for elem in elements:
        for key in flts.keys():
            if elem.split("_")[0] == key.split("_")[0]:
                yield from select_filters(flts[key])
                break
            else:
                yield from select_filters([])
        for key in exposure_time.keys():
            if elem.split("_")[0] == key.split("_")[0]:
                exposure = exposure_time[key]
                print(elem, exposure)
                break
            else:
                exposure = 0.05
                print("use default exposure time 0.05s")
        eng_list = _mk_eng_list(elem, bulk=False)
        if scan_type == "2D":
            if binning is None:
                binning = 0
            if int(binning) not in [0, 1, 2, 3, 4]:
                raise ValueError("binnng must be in [0, 1, 2, 3, 4]")
            yield from mv(Andor.binning, binning)

            yield from _multi_pos_xanes_2D_xh(
                eng_list,
                x_list,
                y_list,
                z_list,
                r_list,
                out_x=out_pos[0],
                out_y=out_pos[1],
                out_z=out_pos[2],
                out_r=out_pos[3],
                exposure_time=exposure,
                chunk_size=chunk_size,
                simu=simu,
                relative_move_flag=relative_move_flag,
                note=note,
                md=None,
                sleep_time=0,
                repeat_num=1,
                enable_z=enable_z,
            )
        elif scan_type == "3D":
            if binning is None:
                binning = 1
            if int(binning) not in [0, 1, 2, 3, 4]:
                raise ValueError("binnng must be in [0, 1, 2, 3, 4]")
            yield from mv(Andor.binning, binning)

            yield from _multi_pos_xanes_3D_xh(
                eng_list,
                x_list,
                y_list,
                z_list,
                r_list,
                exposure_time=exposure,
                relative_rot_angle=rel_rot_ang,
                rs=rs,
                out_x=out_pos[0],
                out_y=out_pos[1],
                out_z=out_pos[2],
                out_r=out_pos[3],
                note=note,
                simu=simu,
                relative_move_flag=relative_move_flag,
                rot_first_flag=1,
                sleep_time=0,
                repeat=1,
                enable_z=enable_z,
            )
            if ref_flat_scan:
                motor_x_ini = zps.sx.position
                motor_y_ini = zps.sy.position
                motor_z_ini = zps.sz.position
                motor_r_ini = zps.pi_r.position

                if relative_move_flag:
                    motor_x_out = (
                        motor_x_ini + out_pos[0]
                        if not (out_pos[0] is None)
                        else motor_x_ini
                    )
                    motor_y_out = (
                        motor_y_ini + out_pos[1]
                        if not (out_pos[1] is None)
                        else motor_y_ini
                    )
                    motor_z_out = (
                        motor_z_ini + out_pos[2]
                        if not (out_pos[2] is None)
                        else motor_z_ini
                    )
                    motor_r_out = (
                        motor_r_ini + out_pos[3]
                        if not (out_pos[3] is None)
                        else motor_r_ini
                    )
                else:
                    motor_x_out = (
                        out_pos[0] if not (out_pos[0] is None) else motor_x_ini
                    )
                    motor_y_out = (
                        out_pos[1] if not (out_pos[1] is None) else motor_y_ini
                    )
                    motor_z_out = (
                        out_pos[2] if not (out_pos[2] is None) else motor_z_ini
                    )
                    motor_r_out = (
                        out_pos[0] if not (out_pos[3] is None) else motor_r_ini
                    )

                _move_sample_out_xhx(
                    motor_x_out,
                    motor_y_out,
                    motor_z_out,
                    motor_r_out,
                    repeat=2,
                    rot_first_flag=1,
                    enable_z=enable_z,
                )
                for ii in [
                    eng_list[0],
                    eng_list[int(eng_list.shape[0] / 2)],
                    eng_list[-1],
                ]:
                    yield from move_zp_ccd(ii, move_flag=1)
                    my_note = note + f"_ref_flat@energy={ii}_keV"
                    yield from bps.sleep(1)
                    print(f"current energy: {ii}")

                    # period = _exp_t_sanity_check(exposure, binning)
                    period = exposure
                    if period:
                        yield from fly_scan2(
                            exposure,
                            start_angle=None,
                            rel_rot_ang=rel_rot_ang,
                            period=period,
                            out_x=out_pos[0],
                            out_y=out_pos[1],
                            out_z=out_pos[2],
                            out_r=out_pos[3],
                            rs=rs,
                            relative_move_flag=relative_move_flag,
                            note=my_note,
                            simu=simu,
                            rot_first_flag=1,
                            enable_z=enable_z,
                        )
                    else:
                        print(f"invalid binning {binning}")
                _move_sample_in_xhx(
                    motor_x_ini,
                    motor_y_ini,
                    motor_z_ini,
                    motor_r_ini,
                    repeat=2,
                    trans_first_flag=1,
                    enable_z=enable_z,
                )
        else:
            print("wrong scan type")
            return


def multi_edge_xanes2(
    elements=["Ni_wl"],
    scan_type="3D",
    flts={"Ni_filters": [1, 2, 3]},
    exposure_time={"Ni_exp": 0.05},
    period_time={"Ni_period": 0.05},
    rel_rot_ang=185,
    start_angle=None,
    rs=6,
    in_pos_list=[[None, None, None, None]],
    out_pos=[None, None, None, None],
    note="",
    relative_move_flag=0,
    binning=None,
    bulk=False,
    bulk_intgr=10,
    simu=False,
    sleep=0,
    repeat=None,
    ref_flat_scan=False,
    enable_z=True,
):
    yield from mv(Andor.cam.acquire, 0)
    if repeat is None:
        repeat = 1
    repeat = int(repeat)
    if start_angle is None:
        start_angle = zps.pi_r.position
    if scan_type == "2D":
        if binning is None:
            binning = 0
        # binning = yield from _bin_cam(binning)
        print(1)

        x_list, y_list, z_list, r_list = _sort_in_pos(in_pos_list)
        print(2)
        for elem in elements:
            for key in flts.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    flt = flts[key]
                    break
                else:
                    flt = []
            for key in exposure_time.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    print(f"{exposure_time[key]=}")
                    exposure = exposure_time[key]
                    print(elem, exposure)
                    break
                else:
                    exposure = 0.05
                    print("use default exposure time 0.05s")
            for key in period_time.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    yield from mv(Andor.cam.acquire_time, exposure)
                    yield from bps.sleep(2)
                    # yield from mv(Andor.cam.acquire_period, period_time[key])
                    # period = yield from rd(Andor.cam.acquire_period)
                    # period = yield from _exp_t_sanity_check(period, binning=binning)
                    # print(elem, f"{period=}")
                    break
            eng_list = _mk_eng_list(elem, bulk=False)
            print(f"{out_pos=}")
            yield from _multi_pos_xanes_2D_xh(
                eng_list,
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
                sleep_time=sleep,
                repeat_num=repeat,
                binning=binning,
                flts=flt,
                enable_z=enable_z,
            )
    elif scan_type == "3D":
        if binning is None:
            binning = 1
        binning = yield from _bin_cam(binning)

        yield from mv(Andor.cam.image_mode, 0)
        yield from mv(Andor.cam.num_images, 5)
        yield from mv(Andor.cam.acquire, 1)

        x_list, y_list, z_list, r_list = _sort_in_pos(in_pos_list)
        for elem in elements:
            for key in flts.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    flt = flts[key]
                    break
                else:
                    flt = []
            for key in exposure_time.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    exposure = exposure_time[key]
                    print(elem, exposure)
                    break
                else:
                    exposure = 0.05
                    print("use default exposure time 0.05s")
            for key in period_time.keys():
                print(f"{period_time=}\n{key=}")
                if elem.split("_")[0] == key.split("_")[0]:
                    print(f"{exposure=}")
                    yield from mv(Andor.cam.acquire_time, exposure)
                    yield from bps.sleep(2)
                    print(f"{period_time[key]=}")
                    # yield from mv(Andor.cam.acquire_period, period_time[key])
                    # period = yield from rd(Andor.cam.acquire_period)
                    # period = _exp_t_sanity_check(period, binning)
                    # print(elem, f"{period=}")
                    break
            eng_list = _mk_eng_list(elem, bulk=False)

            yield from _multi_pos_xanes_3D_xh(
                eng_list,
                x_list,
                y_list,
                z_list,
                r_list,
                exposure_time=exposure,
                start_angle=start_angle,
                relative_rot_angle=rel_rot_ang,
                # period=period,
                period=exposure,
                rs=rs,
                out_x=out_pos[0],
                out_y=out_pos[1],
                out_z=out_pos[2],
                out_r=out_pos[3],
                note=note,
                simu=simu,
                relative_move_flag=relative_move_flag,
                rot_first_flag=1,
                sleep_time=sleep,
                binning=binning,
                flts=flt,
                repeat=repeat,
            )

            if bulk:
                eng_list = _mk_eng_list(elem, bulk=True)
                zpx = zp.x.position
                apx = aper.x.position
                cdx = clens.x.position
                yield from mv(zp.x, -6500 + zpx)
                yield from mv(clens.x, 6500 + cds)
                yield from mv(aper.x, -4000 + apx)
                xxanes_scan(eng_list, delay_time=0.2, intgr=bulk_intgr, note=note)
                yield from mv(clens.x, cds)
                yield from mv(aper.x, apx)
                yield from mv(zp.x, zpx)

    else:
        print("wrong scan type")

        # if itr != repeat - 1:
        #    yield from bps.sleep(sleep)
        # print(f"repeat # {itr} finished")


def multi_edge_xanes3(
    elements=["Ni_wl"],
    scan_type="3D",
    flts={"Ni_filters": [1, 2, 3]},
    exposure_time={"Ni_exp": 0.05},
    period_time={"Ni_period": 0.05},
    rel_rot_ang=185,
    start_angle=None,
    rs=6,
    in_pos_list=[[None, None, None, None]],
    out_pos=[None, None, None, None],
    note="",
    relative_move_flag=0,
    binning=None,
    bulk=False,
    bulk_intgr=10,
    simu=False,
    sleep=0,
    repeat=None,
    ref_flat_scan=False,
    enable_z=True,
):
    yield from abs_set(Andor.cam.acquire, 0)
    if repeat is None:
        repeat = 1
    repeat = int(repeat)
    if start_angle is None:
        start_angle = zps.pi_r.position
    if scan_type == "2D":
        if binning is None:
            binning = 0
        # binning = yield from _bin_cam(binning)
        print(1)

        x_list, y_list, z_list, r_list = _sort_in_pos(in_pos_list)
        print(2)
        for elem in elements:
            for key in flts.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    flt = flts[key]
                    break
                else:
                    flt = []
            for key in exposure_time.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    print(f"{exposure_time[key]=}")
                    exposure = exposure_time[key]
                    print(elem, exposure)
                    break
                else:
                    exposure = 0.05
                    print("use default exposure time 0.05s")
            for key in period_time.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    yield from abs_set(Andor.cam.acquire_time, exposure)
                    yield from bps.sleep(2)
                    # yield from mv(Andor.cam.acquire_period, period_time[key])
                    # period = yield from rd(Andor.cam.acquire_period)
                    # period = yield from _exp_t_sanity_check(period, binning=binning)
                    # print(elem, f"{period=}")
                    break
            eng_list = _mk_eng_list(elem, bulk=False)
            print(f"{out_pos=}")
            yield from _multi_pos_xanes_2D_xh(
                eng_list,
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
                sleep_time=sleep,
                repeat_num=repeat,
                binning=binning,
                flts=flt,
                enable_z=enable_z,
            )
    elif scan_type == "3D":
        if binning is None:
            binning = 1
        binning = yield from _bin_cam(binning)

        yield from abs_set(Andor.cam.image_mode, 0)
        yield from abs_set(Andor.cam.trigger_mode, 0)
        yield from abs_set(Andor.cam.num_images, 5)
        yield from abs_set(Andor.cam.acquire, 1)

        x_list, y_list, z_list, r_list = _sort_in_pos(in_pos_list)
        for elem in elements:
            for key in flts.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    flt = flts[key]
                    break
                else:
                    flt = []
            for key in exposure_time.keys():
                if elem.split("_")[0] == key.split("_")[0]:
                    exposure = exposure_time[key]
                    print(elem, exposure)
                    break
                else:
                    exposure = 0.05
                    print("use default exposure time 0.05s")
            for key in period_time.keys():
                print(f"{period_time=}\n{key=}")
                if elem.split("_")[0] == key.split("_")[0]:
                    print(f"{exposure=}")
                    yield from abs_set(Andor.cam.acquire_time, exposure)
                    yield from bps.sleep(2)
                    print(f"{period_time[key]=}")
                    break
            eng_list = _mk_eng_list(elem, bulk=False)

            yield from _multi_pos_xanes_3D_xh(
                eng_list,
                x_list,
                y_list,
                z_list,
                r_list,
                exposure_time=exposure,
                start_angle=start_angle,
                relative_rot_angle=rel_rot_ang,
                # period=period,
                period=exposure,
                rs=rs,
                out_x=out_pos[0],
                out_y=out_pos[1],
                out_z=out_pos[2],
                out_r=out_pos[3],
                note=note,
                simu=simu,
                relative_move_flag=relative_move_flag,
                rot_first_flag=1,
                sleep_time=sleep,
                binning=binning,
                flts=flt,
                repeat=repeat,
            )

            if bulk:
                eng_list = _mk_eng_list(elem, bulk=True)
                zpx = zp.x.position
                apx = aper.x.position
                cdx = clens.x.position
                yield from mv(zp.x, -6500 + zpx)
                yield from mv(clens.x, 6500 + cds)
                yield from mv(aper.x, -4000 + apx)
                xxanes_scan(eng_list, delay_time=0.2, intgr=bulk_intgr, note=note)
                yield from mv(clens.x, cds)
                yield from mv(aper.x, apx)
                yield from mv(zp.x, zpx)

    else:
        print("wrong scan type")


def fly_scan2(
    exposure_time=0.05,
    start_angle=None,
    rel_rot_ang=180,
    period=0.05,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    rs=3,
    relative_move_flag=1,
    rot_first_flag=1,
    flts=[],
    rot_back_velo=30,
    binning=None,
    note="",
    md=None,
    move_to_ini_pos=True,
    simu=False,
    enable_z=True,
    cam=Andor,
):
    """
    Inputs:
    -------
    exposure_time: float, in unit of sec

    start_angle: float
        starting angle

    rel_rot_ang: float,
        total rotation angles start from current rotary stage (zps.pi_r) position

    period: float, in unit of sec
        period of taking images, "period" should >= "exposure_time"

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
    yield from mv(cam.cam.acquire, 0)
    if binning is None:
        binning = 0
    if int(binning) not in [0, 1, 2, 3, 4]:
        raise ValueError("binnng must be in [0, 1, 2, 3, 4]")
    yield from mv(cam.binning, binning)
    yield from mv(cam.cam.image_mode, 0)
    yield from mv(cam.cam.num_images, 5)
    yield from mv(cam.cam.acquire, 1)

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

    if enable_z:
        motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]
    else:
        motor = [zps.sx, zps.sy, zps.pi_r]

    dets = [cam, ic3]
    taxi_ang = -1 * rs
    cur_rot_ang = zps.pi_r.position
    tgt_rot_ang = cur_rot_ang + rel_rot_ang
    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motor],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exposure_time,
            "start_angle": start_angle if start_angle else "None",
            "relative_rot_angle": rel_rot_ang,
            "period": period,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "rs": rs,
            "relative_move_flag": relative_move_flag,
            "rot_first_flag": rot_first_flag,
            "enable_z": "True" if enable_z else "False",
            "filters": ["filter{}".format(t) for t in flts] if flts else "None",
            "binning": "None" if binning is None else binning,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "fly_scan2",
        "num_bkg_images": 20,
        "num_dark_images": 20,
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        "zone_plate": ZONE_PLATE,
    }
    _md.update(md or {})
    try:
        dimensions = [(zps.pi_r.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    yield from _set_andor_param_xhx(
        exposure_time=exposure_time,
        period=period,
        chunk_size=20,
        binning=binning,
        cam=cam,
    )
    yield from _set_rotation_speed(rs=np.abs(rs))
    print("set rotation speed: {} deg/sec".format(rs))

    @stage_decorator(list(dets) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        # yield from select_filters(flts)
        yield from bps.sleep(1)

        # close shutter, dark images: numer=chunk_size (e.g.20)
        print("\nshutter closed, taking dark images...")
        yield from _take_dark_image_xhx(
            dets, motor, num=1, chunk_size=20, stream_name="dark", simu=simu, cam=cam
        )

        # open shutter, tomo_images
        true_period = yield from rd(cam.cam.acquire_period)
        rot_time = np.abs(rel_rot_ang) / np.abs(rs)
        num_img = int(rot_time / true_period) + int(10 * rs)

        yield from _open_shutter_xhx(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from _set_Andor_chunk_size_xhx(dets, num_img, cam)
        # yield from mv(zps.pi_r, cur_rot_ang + taxi_ang)
        status = yield from abs_set(zps.pi_r, tgt_rot_ang, wait=False)
        # yield from bps.sleep(1)
        yield from _take_image(dets, motor, num=1, stream_name="primary")
        while not status.done:
            yield from bps.sleep(0.01)
            # yield from trigger_and_read(list(dets) + motor)

        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=rot_back_velo)
        #        yield from abs_set(zps.pi_r.velocity, rs)

        yield from _take_bkg_image_xhx(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            dets,
            [],
            num=1,
            chunk_size=20,
            rot_first_flag=rot_first_flag,
            stream_name="flat",
            simu=simu,
            enable_z=enable_z,
            cam=cam,
        )
        yield from _close_shutter_xhx(simu=simu)
        if move_to_ini_pos:
            yield from _move_sample_in_xhx(
                motor_x_ini,
                motor_y_ini,
                motor_z_ini,
                motor_r_ini,
                trans_first_flag=rot_first_flag,
                repeat=2,
                enable_z=enable_z,
            )

    uid = yield from fly_inner_scan()
    yield from mv(cam.cam.image_mode, 1)
    # yield from select_filters([])
    print("scan finished")
    return uid


def fly_scan3(
    exposure_time=0.05,
    start_angle=None,
    rel_rot_ang=180,
    period=0.05,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    rs=3,
    relative_move_flag=1,
    rot_first_flag=1,
    flts=[],
    rot_back_velo=30,
    binning=None,
    note="",
    md=None,
    move_to_ini_pos=True,
    simu=False,
    noDark=False,
    noFlat=False,
    enable_z=True,
    cam=Andor,
):
    """
    Inputs:
    -------
    exposure_time: float, in unit of sec

    start_angle: float
        starting angle

    rel_rot_ang: float,
        total rotation angles start from current rotary stage (zps.pi_r) position

    period: float, in unit of sec
        period of taking images, "period" should >= "exposure_time"

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
    yield from mv(cam.cam.acquire, 0)
    global ZONE_PLATE
    yield from select_filters(flts)

    if binning is None:
        binning = 0
    if int(binning) not in [0, 1, 2, 3, 4]:
        raise ValueError("binnng must be in [0, 1, 2, 3, 4]")
    yield from mv(cam.binning, binning)

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

    if enable_z:
        motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]
    else:
        motor = [zps.sx, zps.sy, zps.pi_r]

    dets = [cam, ic3]
    taxi_ang = -1 * rs
    cur_rot_ang = zps.pi_r.position
    tgt_rot_ang = cur_rot_ang + rel_rot_ang
    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motor],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exposure_time,
            "start_angle": start_angle,
            "relative_rot_angle": rel_rot_ang,
            "period": period,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "rs": rs,
            "relative_move_flag": relative_move_flag,
            "rot_first_flag": rot_first_flag,
            "filters": ["filter{}".format(t) for t in flts] if flts else "None",
            "binning": binning,
            "enable_z": "True" if enable_z else "False",
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
            "noDark": "True" if noDark else "False",
            "noFlat": "True" if noFlat else "False",
        },
        "plan_name": "fly_scan3",
        "num_bkg_images": 20,
        "num_dark_images": 20,
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

    yield from _set_andor_param_xhx(
        exposure_time=exposure_time,
        period=period,
        chunk_size=20,
        binning=binning,
        cam=cam,
    )
    yield from _set_rotation_speed(rs=np.abs(rs))
    print("set rotation speed: {} deg/sec".format(rs))

    @stage_decorator(list(dets) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        # close shutter, dark images: numer=chunk_size (e.g.20)
        if not noDark:
            print("\nshutter closed, taking dark images...")
            yield from _take_dark_image_xhx(
                dets,
                motor,
                num=1,
                chunk_size=20,
                stream_name="dark",
                simu=simu,
                cam=cam,
            )

        # open shutter, tomo_images
        true_period = yield from rd(cam.cam.acquire_period)
        rot_time = np.abs(rel_rot_ang) / np.abs(rs)
        num_img = int(rot_time / true_period) + 2

        yield from _open_shutter_xhx(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from _set_Andor_chunk_size_xhx(dets, num_img, cam)
        # yield from mv(zps.pi_r, cur_rot_ang + taxi_ang)
        status = yield from abs_set(zps.pi_r, tgt_rot_ang, wait=False)
        # yield from bps.sleep(1)
        yield from _take_image(dets, motor, num=1, stream_name="primary")
        while not status.done:
            yield from bps.sleep(0.01)
            # yield from trigger_and_read(list(dets) + motor)

        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=rot_back_velo)
        #        yield from abs_set(zps.pi_r.velocity, rs)

        if not noFlat:
            yield from _take_bkg_image_xhx(
                motor_x_out,
                motor_y_out,
                motor_z_out,
                motor_r_out,
                dets,
                [],
                num=1,
                chunk_size=20,
                rot_first_flag=rot_first_flag,
                stream_name="flat",
                simu=simu,
                enable_z=enable_z,
                cam=cam,
            )

        if not noDark:
            yield from _close_shutter_xhx(simu=simu)

        if move_to_ini_pos:
            yield from _move_sample_in_xhx(
                motor_x_ini,
                motor_y_ini,
                motor_z_ini,
                motor_r_ini,
                trans_first_flag=rot_first_flag,
                repeat=2,
                enable_z=enable_z,
            )

    uid = yield from fly_inner_scan()
    yield from mv(cam.cam.image_mode, 1)
    yield from select_filters([])
    print("scan finished")
    return uid


def rock_scan(
    exp_t=0.05,
    period=0.05,
    t_span=10,
    start_angle=None,
    rel_rot_ang=30,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    rs=30,
    relative_move_flag=1,
    rot_first_flag=1,
    rot_back_velo=30,
    flts=[],
    binning=None,
    note="",
    md=None,
    move_to_ini_pos=True,
    simu=False,
    noDark=False,
    noFlat=False,
    enable_z=True,
):
    """
    Inputs:
    -------
    exp_t: float, in unit of sec

    start_angle: float
        starting angle

    rel_rot_ang: float,
        total rotation angles start from current rotary stage (zps.pi_r) position

    period: float, in unit of sec
        period of taking images, "period" should >= "exposure_time"

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
    yield from mv(Andor.cam.acquire, 0)
    global ZONE_PLATE

    if binning is None:
        binning = 0
    if int(binning) not in [0, 1, 2, 3, 4]:
        raise ValueError("binnng must be in [0, 1, 2, 3, 4]")
    yield from mv(Andor.binning, binning)

    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if not (start_angle is None):
        yield from mv(zps.pi_r, start_angle)
    else:
        start_angle = zps.pi_r.position

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

    rev = int(np.ceil(t_span / (2 * rel_rot_ang / rs))) + 1
    mots = [zps.pi_r]

    dets = [Andor]
    tgt_ang = start_angle + rel_rot_ang
    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in mots],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exp_t,
            "start_angle": start_angle,
            "relative_rot_angle": rel_rot_ang,
            "period": period,
            "time_span": t_span,
            "rock_velocity": rs,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "relative_move_flag": relative_move_flag,
            "rot_first_flag": rot_first_flag,
            "filters": ["filter{}".format(t) for t in flts] if flts else "None",
            "binning": binning,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
            "noDark": "True" if noDark else "False",
            "noFlat": "True" if noFlat else "False",
        },
        "plan_name": "rock_scan",
        "num_bkg_images": 20,
        "num_dark_images": 20,
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

    yield from _set_andor_param(exposure_time=exp_t, period=period, chunk_size=20)
    true_period = yield from rd(Andor.cam.acquire_period)
    num_img = int(t_span / true_period) + 2

    yield from _set_rotation_speed(rs=np.abs(rs))
    print("set rotation speed: {} deg/sec".format(rs))

    @stage_decorator(dets + mots)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def rock_inner_scan():
        yield from select_filters(flts)
        yield from bps.sleep(1)

        # close shutter, dark images: numer=chunk_size (e.g.20)
        if not noDark:
            print("\nshutter closed, taking dark images...")
            yield from _take_dark_image_xhx(
                dets, mots, num=1, chunk_size=20, stream_name="dark", simu=simu
            )

        # open shutter, tomo_images
        yield from _set_Andor_chunk_size(dets, chunk_size=num_img)
        yield from _open_shutter_xhx(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from abs_set(zps.pi_r, start_angle, wait=True)

        # modified based on trigger_and_read
        tgt, old_tgt = tgt_ang, start_angle
        yield from trigger(Andor, group="Andor", wait=False)
        for ii in range(rev):
            yield from mv(zps.pi_r, tgt)
            old_tgt, tgt = tgt, old_tgt
        yield from bps.wait(group="Andor")
        yield from bps.create("primary")
        yield from bps.read(Andor)
        yield from bps.save()

        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=rot_back_velo)

        if not noFlat:
            yield from _take_bkg_image_xhx(
                motor_x_out,
                motor_y_out,
                motor_z_out,
                motor_r_out,
                dets,
                [],
                num=1,
                chunk_size=20,
                rot_first_flag=rot_first_flag,
                stream_name="flat",
                simu=simu,
                enable_z=enable_z,
            )

        if move_to_ini_pos:
            yield from _move_sample_in_xhx(
                motor_x_ini,
                motor_y_ini,
                motor_z_ini,
                motor_r_ini,
                trans_first_flag=rot_first_flag,
                repeat=2,
                enable_z=enable_z,
            )
        yield from select_filters([])

    uid = yield from rock_inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    print("scan finished")
    return uid


def mosaic_fly_scan_xh(
    x_ini=None,
    y_ini=None,
    z_ini=None,
    x_num_steps=1,
    y_num_steps=1,
    z_num_steps=1,
    x_step_size=0,
    y_step_size=0,
    z_step_size=0,
    exposure_time=0.1,
    period=0.1,
    rs=4,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    start_angle=None,
    rel_rot_ang=180,
    binning=None,
    flts=[],
    relative_move_flag=True,
    simu=False,
    note="",
    enable_z=True,
    repeat=1,
    sleep=0,
):
    yield from select_filters(flts)
    binning = yield from _bin_cam(binning)

    if x_ini is None:
        x_ini = zps.sx.position
    if y_ini is None:
        y_ini = zps.sy.position
    if z_ini is None:
        z_ini = zps.sz.position
    r_ini = zps.pi_r.position

    y_list = y_ini + np.arange(y_num_steps) * y_step_size
    x_list = x_ini + np.arange(x_num_steps) * x_step_size
    z_list = z_ini + np.arange(z_num_steps) * z_step_size
    txt1 = "\n###############################################"
    txt2 = "\n#######    start mosaic tomography scan  ######"
    txt3 = "\n###############################################"
    txt = txt1 + txt2 + txt3
    print(txt)

    yield from bps.sleep(1)

    for ii in range(int(repeat)):
        for y in y_list:
            for z in z_list:
                for x in x_list:
                    yield from mv(zps.sx, x, zps.sy, y, zps.sz, z)
                    yield from fly_scan2(
                        exposure_time=exposure_time,
                        start_angle=start_angle,
                        rel_rot_ang=rel_rot_ang,
                        period=period,
                        out_x=out_x,
                        out_y=out_y,
                        out_z=out_z,
                        out_r=out_r,
                        rs=rs,
                        relative_move_flag=relative_move_flag,
                        note=f"{note}; pos_y: {y}, pos_z: {z}, pos_x: {x}; iteration: {ii}",
                        simu=simu,
                        rot_first_flag=True,
                        enable_z=enable_z,
                    )
        if ii < int(repeat) - 1:
            print(f"sleeping for {sleep} seconds before iteration #{ii+1}")
            yield from bps.sleep(sleep)
    yield from mv(zps.sx, x_ini, zps.sy, y_ini, zps.sz, z_ini, zps.pi_r, r_ini)
    yield from select_filters([])


def grid_z_scan(
    zstart=-0.03,
    zstop=0.03,
    zsteps=5,
    gmesh=[[-5, 0, 5], [-5, 0, 5]],
    out_x=-100,
    out_y=-100,
    chunk_size=10,
    exposure_time=0.1,
    note="",
    md=None,
    simu=False,
):
    """
    scan the zone-plate to find best focus
    use as:
    z_scan(zstart=-0.03, zstop=0.03, zsteps=5, gmesh=[[-5, 0, 5], [-5, 0, 5]], out_x=-100, out_y=-100, chunk_size=10, exposure_time=0.1, fn='/home/xf18id/Documents/tmp/z_scan.h5', note='', md=None)

    Input:
    ---------
    zstart: float, relative starting position of zp_z

    zstop: float, relative zstop position of zp_z

    zsteps: int, number of zstep between [zstart, zstop]

    out_x: float, relative amount to move sample out for zps.sx

    out_y: float, relative amount to move sample out for zps.sy

    chunk_size: int, number of images per each subscan (for Andor camera)

    exposure_time: float, exposure time for each image

    note: str, experiment notes

    """
    yield from mv(Andor.cam.acquire, 0)
    dets = [Andor]
    motor = zp.z
    z_ini = motor.position  # zp.z intial position
    z_start = z_ini + zstart
    z_stop = z_ini + zstop
    zp_x_ini = zp.x.position
    zp_y_ini = zp.y.position
    #    dets = [Andor]
    y_ini = zps.sy.position  # sample y position (initial)
    y_out = (
        y_ini + out_y if not (out_y is None) else y_ini
    )  # sample y position (out-position)
    x_ini = zps.sx.position
    x_out = x_ini + out_x if not (out_x is None) else x_ini
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.image_mode, 0)
    yield from mv(Andor.cam.num_images, chunk_size)
    yield from mv(Andor.cam.acquire_time, exposure_time)
    period_cor = max(exposure_time + 0.01, 0.05)
    # yield from mv(Andor.cam.acquire_period, period_cor)

    _md = {
        "detectors": [det.name for det in dets],
        "motors": [motor.name],
        "XEng": XEng.position,
        "plan_args": {
            "zstart": zstart,
            "zstop": zstop,
            "zsteps": zsteps,
            "gmesh": gmesh,
            "out_x": out_x,
            "out_y": out_y,
            "chunk_size": chunk_size,
            "exposure_time": exposure_time,
            "note": note if note else "None",
        },
        "plan_name": "grid_z_scan",
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "motor_pos": wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    my_var = np.linspace(z_start, z_stop, zsteps)
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(dets) + [motor])
    @run_decorator(md=_md)
    def inner_scan():
        yield from _open_shutter_xhx(simu=simu)
        for xx in gmesh[0]:
            for yy in gmesh[1]:
                yield from mv(zp.x, zp_x_ini + xx, zp.y, zp_y_ini + yy, wait=True)
                # yield from mv(zp.y, zp_y_ini+yy, wait=True)
                yield from bps.sleep(1)
                for x in my_var:
                    yield from mv(motor, x)
                    yield from trigger_and_read(list(dets) + [motor], name="primary")
                # backgroud images
                yield from mv(zps.sx, x_out, zps.sy, y_out, wait=True)
                yield from bps.sleep(1)
                yield from trigger_and_read(list(dets) + [motor], name="flat")
                yield from mv(zps.sx, x_ini, zps.sy, y_ini, wait=True)
                yield from bps.sleep(1)
                # yield from mv(zps.sy, y_ini, wait=True)
        yield from _close_shutter_xhx(simu=simu)
        yield from bps.sleep(1)
        yield from trigger_and_read(list(dets) + [motor], name="dark")

        yield from mv(zps.sx, x_ini)
        yield from mv(zps.sy, y_ini)
        yield from mv(zp.z, z_ini)
        yield from mv(zp.x, zp_x_ini, zp.y, zp_y_ini, wait=True)
        yield from mv(Andor.cam.image_mode, 1)

    uid = yield from inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    yield from _close_shutter_xhx(simu=simu)
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)


def xxanes_scan(
    eng_list,
    delay_time=0.5,
    intgr=1,
    dets=[ic1, ic2, ic3],
    note="",
    md=None,
    repeat=None,
    sleep=1200,
):
    """
    eng_list: energy list in keV
    delay_time: delay_time between each energy step, in unit of sec
    note: string; optional, description of the scan
    """
    if repeat is None:
        repeat = 1
    repeat = int(repeat)

    check_eng_range([eng_list[0], eng_list[-1]])
    print(0)
    yield from _open_shutter_xhx(simu=False)
    # dets=[ic1, ic2, ic3]
    motor_x = XEng
    motor_x_ini = motor_x.position  # initial position of motor_x

    # added by XH -- start
    motor_y = dcm
    # added by XH -- end

    _md = {
        "detectors": "".join(ii.name + " " for ii in dets),
        "motors": [motor_x.name, motor_y.name],
        "XEng": XEng.position,
        "plan_name": "xxanes",
        "plan_args": {
            "eng": eng_list,
            "detectors": "".join(ii.name + " " for ii in dets),
            "delay_time": delay_time,
            "repeat": intgr,
            "note": note if note else "None",
        },
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
    }
    _md.update(md or {})
    try:
        dimensions = [
            (motor_x.hints["fields"], "primary"),
            (motor_y.hints["fields"], "primary"),
        ]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(dets) + [motor_x, motor_y])
    @run_decorator(md=_md)
    def eng_inner_scan():
        for eng in eng_list:
            yield from mv(motor_x, eng)
            yield from bps.sleep(delay_time)
            yield from bps.repeat(
                partial(bps.trigger_and_read, list(dets) + [motor_x, motor_y]),
                num=intgr,
                delay=0.01,
            )
            # yield from trigger_and_read(list(dets) + [motor_x, motor_y])
        yield from mv(motor_x, motor_x_ini)

    for itr in range(repeat):
        yield from _open_shutter_xhx(simu=False)
        yield from eng_inner_scan()
        yield from _close_shutter_xhx(simu=False)
        if itr != (repeat - 1):
            yield from bps.sleep(sleep)
        print(f"repeat # {itr} finished")


def xxanes_scan2(
    eng_list, dets=[ic1, ic2, ic3], note="", md=None, repeat=1, sleep=100, simu=False
):
    """
    eng_list: energy list in keV
    note: string; optional, description of the scan
    """
    repeat = int(repeat)

    check_eng_range([eng_list[0], eng_list[-1]])
    print(0)
    XEng_target = eng_list[-1]
    # motor_x = dcm
    # motor_y = XEng
    XEng_ini = XEng.position  # initial position of motor_x
    dcm_vel_ini = dcm.th1.velocity.value
    # yield from mv(dcm.th1.velocity, 0.1)
    # yield from mv(XEng, eng_list[0])

    ang0 = np.arcsin(12.398 / eng_list[0] / 2 / (5.43 / np.sqrt(3)))
    ang1 = np.arcsin(12.398 / eng_list[-1] / 2 / (5.43 / np.sqrt(3)))
    dcm_vel = (
        10
        * 180
        * np.abs(ang1 - ang0)
        / np.pi
        / (4 * np.abs(eng_list[0] - eng_list[-1]))
        / 1000
    )
    if dcm_vel < 0.00279:
        dcm_vel = 0.00279
    intgr = int(np.ceil(10 * 180 * np.abs(ang1 - ang0) / np.pi / dcm_vel))

    _md = {
        "detectors": "".join(ii.name + " " for ii in dets),
        "motors": [XEng.name, dcm.name],
        "XEng": XEng.position,
        "plan_name": "xxanes2",
        "plan_args": {
            "eng": eng_list,
            "detectors": "".join(ii.name + " " for ii in dets),
            "repeat": repeat,
            "IC_rate": 10,
            "dcm velocity": dcm_vel,
            "note": note if note else "None",
        },
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
    }
    _md.update(md or {})
    try:
        dimensions = [
            (XEng.hints["fields"], "primary"),
            (dcm.hints["fields"], "primary"),
        ]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(dets) + [XEng, dcm])
    @bpp.monitor_during_decorator(
        [XEng.user_readback, dcm.th1.user_readback] + list(dets)
    )
    @run_decorator(md=_md)
    def eng_inner_scan():
        while XEng.moving:
            yield from bps.sleep(1)
        status = yield from abs_set(XEng, XEng_target, wait=False)
        # yield from trigger_and_read(dets + [dcm, XEng], name=stream_name)
        yield from bps.repeat(
            partial(bps.trigger_and_read, list(dets) + [XEng, dcm]),
            num=intgr,
            delay=0.1,
        )
        while not status.done:
            yield from bps.sleep(0.01)

    for itr in range(repeat):
        yield from _open_shutter_xhx(simu=simu)
        yield from mv(dcm.th1.velocity, 0.1)
        yield from mv(XEng, eng_list[0])
        yield from mv(dcm.th1.velocity, dcm_vel)
        yield from bps.sleep(1)
        yield from eng_inner_scan()
        yield from _close_shutter_xhx(simu=simu)
        if itr != (repeat - 1):
            yield from bps.sleep(sleep)
        print(f"repeat # {itr} finished")

    yield from mv(dcm.th1.velocity, 0.1)
    yield from mv(XEng, XEng_ini)
    yield from mv(dcm.th1.velocity, dcm_vel_ini)


def mosaic_2D_rel_grid_xh(
    mot1=zps.sx,
    mot1_start=-100,
    mot1_end=100,
    mot1_points=6,
    mot2=zps.sy,
    mot2_start=-50,
    mot2_end=50,
    mot2_points=6,
    mot2_snake=False,
    out_x=100,
    out_y=100,
    exp_time=0.1,
    chunk_size=1,
    note="",
    md=None,
    simu=False,
):
    yield from mv(Andor.cam.acquire, 0)
    dets = [Andor]
    y_ini = zps.sy.position  # sample y position (initial)
    y_out = (
        y_ini + out_y if not (out_y is None) else y_ini
    )  # sample y position (out-position)
    x_ini = zps.sx.position
    x_out = x_ini + out_x if not (out_x is None) else x_ini
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.image_mode, 0)
    yield from mv(Andor.cam.num_images, chunk_size)
    yield from mv(Andor.cam.acquire_time, exp_time)
    period_cor = max(exp_time + 0.01, 0.05)
    # yield from mv(Andor.cam.acquire_period, period_cor)

    _md = {
        "detectors": [det.name for det in dets],
        "motors": [mot1.name, mot2.name],
        "XEng": XEng.position,
        "plan_args": {
            "mot1_start": mot1_start,
            "mot1_stop": mot1_end,
            "mot1_pnts": mot1_points,
            "mot2_start": mot2_start,
            "mot2_stop": mot2_end,
            "mot2_pnts": mot2_points,
            "mot2_snake": mot2_snake,
            "out_x": out_x,
            "out_y": out_y,
            "exposure_time": exp_time,
            "note": note if note else "",
        },
        "plan_name": "raster_scan_xh",
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "motor_pos": wh_pos(print_on_screen=0),
    }
    _md.update(md or {})

    # @stage_decorator(list(dets) + [mot1, mot2])
    # @run_decorator(md=_md)
    # def inner_scan():
    yield from _open_shutter_xhx(simu=simu)
    yield from rel_grid_scan(
        dets,
        mot1,
        mot1_start,
        mot1_end,
        mot1_points,
        mot2,
        mot2_start,
        mot2_end,
        mot2_points,
        mot2_snake,
    )
    yield from mv(zps.sx, x_out, zps.sy, y_out, wait=True)
    yield from stage(Andor)
    yield from bps.sleep(1)
    yield from trigger_and_read(list(dets) + [mot1, mot2], name="flat")
    yield from mv(zps.sx, x_ini, zps.sy, y_ini, wait=True)
    yield from bps.sleep(1)
    yield from _close_shutter_xhx(simu=simu)
    yield from stage(Andor)
    yield from bps.sleep(1)
    yield from trigger_and_read(list(dets) + [mot1, mot2], name="dark")

    yield from mv(zps.sx, x_ini)
    yield from mv(zps.sy, y_ini)
    yield from unstage(Andor)
    yield from mv(Andor.cam.image_mode, 1)
    yield from _close_shutter_xhx(simu=simu)


def mosaic_2D_xh(
    x_range=[-1, 1],
    y_range=[-1, 1],
    exposure_time=0.1,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    img_sizeX=2560,
    img_sizeY=2160,
    binning=2,
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    note="",
    scan_x_flag=1,
    flts=[],
    md=None,
    enable_z=True,
):
    yield from mv(Andor.cam.acquire, 0)
    # zp_z_pos = zp.z.position
    # DetU_z_pos = DetU.z.position
    # M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    M = GLOBAL_MAG
    pxl = 6.5 / M * (2560.0 / img_sizeX)

    global ZONE_PLATE
    if enable_z:
        motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]
    else:
        motor = [zps.sx, zps.sy, zps.pi_r]
    dets = [Andor, ic3]
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
        "detectors": [det.name for det in dets],
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
            "enable_z": "True" if enable_z else "False",
            "note": note if note else "None",
            "relative_move_flag": relative_move_flag,
            "rot_first_flag": rot_first_flag,
            "note": note if note else "None",
            "scan_x_flag": scan_x_flag,
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "mosaic_2D_xh",
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

    @stage_decorator(list(dets) + motor)
    @run_decorator(md=_md)
    def mosaic_2D_inner():
        if len(flts):
            yield from select_filters(flts)
        # take dark image
        print("take 5 dark image")
        yield from _take_dark_image_xhx(
            dets, motor, num=5, stream_name="dark", simu=simu
        )

        print("open shutter ...")
        yield from _open_shutter_xhx(simu)

        print("taking mosaic image ...")
        for ii in np.arange(x_range[0], x_range[1] + 1):
            if scan_x_flag == 1:
                yield from abs_set(
                    zps.sx, motor_x_ini + ii * img_sizeX * pxl, wait=True
                )
            else:
                yield from abs_set(
                    zps.sz, motor_z_ini + ii * img_sizeX * pxl, wait=True
                )
            for jj in np.arange(y_range[0], y_range[1] + 1):
                yield from abs_set(
                    zps.sy, motor_y_ini + jj * img_sizeY * pxl, wait=True
                )
                yield from _take_image(dets, motor, 1)

        print("moving sample out to take 5 background image")

        yield from _take_bkg_image_xhx(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            dets,
            motor,
            num=1,
            stream_name="flat",
            simu=simu,
            rot_first_flag=rot_first_flag,
            enable_z=enable_z,
        )

        # move sample in
        yield from _move_sample_in_xhx(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            repeat=1,
            trans_first_flag=1 - rot_first_flag,
            enable_z=enable_z,
        )
        if len(flts):
            yield from select_filters(flts)
        print("closing shutter")
        yield from _close_shutter_xhx(simu)

    yield from mosaic_2D_inner()


def dummy_scan(
    exposure_time=0.1,
    start_angle=None,
    rel_rot_ang=180,
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
    flts=[],
    rot_back_velo=30,
    repeat=1,
):
    yield from mv(Andor.cam.acquire, 0)
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

    # motors = [zps.sx, zps.sy, zps.sz, zps.pi_r]
    motors = [zps.sx, zps.sy, zps.pi_r]

    dets = [Andor, ic3]
    taxi_ang = -2 * rs
    cur_rot_ang = zps.pi_r.position

    tgt_rot_ang = cur_rot_ang + rel_rot_ang
    _md = {"dummy scan": "dummy scan"}

    yield from mv(Andor.cam.acquire, 0)
    yield from _set_andor_param(exposure_time=exposure_time, period=period)
    yield from mv(Andor.cam.image_mode, 1)
    yield from mv(Andor.cam.acquire, 1)

    @stage_decorator(motors)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        # open shutter, tomo_images
        yield from _open_shutter_xhx(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from _set_rotation_speed(rs=rs)
        yield from mv(zps.pi_r, cur_rot_ang + taxi_ang)
        status = yield from abs_set(zps.pi_r, tgt_rot_ang, wait=False)
        while not status.done:
            yield from bps.sleep(1)
        yield from _set_rotation_speed(rs=30)
        print("set rotation speed: {} deg/sec".format(rs))
        status = yield from abs_set(zps.pi_r, cur_rot_ang + taxi_ang, wait=False)
        while not status.done:
            yield from bps.sleep(1)
        yield from abs_set(zps.sx, motor_x_out, wait=True)
        yield from abs_set(zps.sy, motor_y_out, wait=True)
        # yield from abs_set(zps.sz, motor_z_out, wait=True)
        yield from abs_set(zps.pi_r, motor_r_out, wait=True)

        yield from abs_set(zps.sx, motor_x_ini, wait=True)
        yield from abs_set(zps.sy, motor_y_ini, wait=True)
        # yield from abs_set(zps.sz, motor_z_ini, wait=True)
        yield from abs_set(zps.pi_r, motor_r_ini, wait=True)

    for ii in range(repeat):
        yield from fly_inner_scan()
        print("{}th scan finished".format(ii))
    yield from _set_rotation_speed(rs=rot_back_velo)
    print("dummy scan finished")


def radiographic_record(
    exp_t=0.1,
    period=0.1,
    t_span=10,
    stop=True,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    flts=[],
    binning=None,
    md={},
    note="",
    simu=False,
    rot_first_flag=1,
    relative_move_flag=1,
):
    if binning is None:
        binning = 0
    if int(binning) not in [0, 1, 2, 3, 4]:
        raise ValueError("binnng must be in [0, 1, 2, 3, 4]")
    yield from mv(Andor.binning, binning)

    yield from mv(Andor.cam.acquire, 0)
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

    dets = [Andor, ic3]
    _md = {
        "detectors": ["Andor"],
        #        "motors": [mot.name for mot in motors],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exp_t,
            "period": period,
            "time_span": t_span,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "filters": [filt.name for filt in flts] if flts else "None",
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
        "zone_plate": ZONE_PLATE,
    }
    _md.update(md or {})

    yield from mv(Andor.cam.acquire, 0)
    yield from _set_andor_param(exposure_time=exp_t, period=period)
    yield from mv(Andor.cam.image_mode, 0)

    @stage_decorator(list(dets))
    @run_decorator(md=_md)
    def rad_record_inner():
        yield from _open_shutter_xhx(simu=simu)
        yield from select_filters(flts)
        yield from bps.sleep(1)

        yield from mv(Andor.cam.num_images, int(t_span / period))
        yield from trigger_and_read([Andor], name="primary")

        yield from mv(
            zps.sx,
            motor_x_out,
            zps.sy,
            motor_y_out,
            zps.sz,
            motor_z_out,
            zps.pi_r,
            motor_r_out,
        )
        yield from mv(Andor.cam.num_images, 20)
        yield from trigger_and_read([Andor], name="flat")
        yield from _close_shutter_xhx(simu=simu)
        yield from mv(
            zps.sx,
            motor_x_ini,
            zps.sy,
            motor_y_ini,
            zps.sz,
            motor_z_ini,
            zps.pi_r,
            motor_r_ini,
        )
        yield from trigger_and_read([Andor], name="dark")
        yield from mv(Andor.cam.image_mode, 1)
        yield from select_filters([])

    yield from rad_record_inner()


def multi_pos_2D_and_3D_xanes(
    elements=["Ni_wl"],
    flts={"Ni_filters": [1, 2, 3]},
    sam_in_pos_list_2D={"Ni_2D_in_pos_list": [[None, None, None, None]]},
    sam_out_pos_list_2D={"Ni_2D_out_pos_list": [None, None, None, None]},
    sam_in_pos_list_3D={"Ni_3D_in_pos_list": [[None, None, None, None]]},
    sam_out_pos_list_3D={"Ni_3D_out_pos_list": [None, None, None, None]},
    exposure_time_2D={"Ni_2D_exp": 0.05},
    exposure_time_3D={"Ni_3D_exp": 0.05},
    rel_rot_ang=185,
    rs=6,
    sleep_time=0,
    repeat_num=1,
    chunk_size=5,
    note="",
    relative_move_flag=0,
    simu=False,
    enable_z=True,
):
    for ii in range(repeat_num):
        for elem in elements:
            en = elem.split("_")[0]
            yield from multi_edge_xanes(
                elements=[en + "_101"],
                scan_type="2D",
                flts={en + "_filters": []},
                exposure_time=exposure_time_2D,
                rel_rot_ang=rel_rot_ang,
                rs=rs,
                in_pos_list=sam_in_pos_list_2D[en + "_2D_in_pos_list"],
                out_pos=sam_out_pos_list_2D[en + "_2D_out_pos_list"],
                chunk_size=chunk_size,
                note=note + "2D_xanes" + elem,
                relative_move_flag=relative_move_flag,
                binning=None,
                simu=False,
                ref_flat_scan=False,
                enable_z=enable_z,
            )
            yield from multi_edge_xanes(
                elements=[elem],
                scan_type="3D",
                flts=flts,
                exposure_time=exposure_time_3D,
                rel_rot_ang=rel_rot_ang,
                rs=rs,
                in_pos_list=sam_in_pos_list_3D[en + "_3D_in_pos_list"],
                out_pos=sam_out_pos_list_3D[en + "_3D_out_pos_list"],
                chunk_size=chunk_size,
                note=note + "3D_xanes" + elem,
                relative_move_flag=relative_move_flag,
                binning=None,
                simu=False,
                ref_flat_scan=False,
                enable_z=enable_z,
            )
            if ii < repeat_num - 1:
                yield from bps.sleep(sleep_time)


def multi_pos_2D_xanes_and_3D_tomo(
    elements=["Ni"],
    sam_in_pos_list_2D=[[[0, 0, 0, 0]]],
    sam_out_pos_list_2D=[[[0, 0, 0, 0]]],
    sam_in_pos_list_3D=[[[0, 0, 0, 0]]],
    sam_out_pos_list_3D=[[[0, 0, 0, 0]]],
    exposure_time_2D=[0.05],
    exposure_time_3D=[0.05],
    rel_rot_ang=0,
    rs=6,
    eng_3D=[10, 60],
    note="",
    relative_move_flag=0,
    simu=False,
):
    yield from mv(Andor.cam.acquire, 0)
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
                    "/nsls2/data/fxi-new/shared/config/xanes_ref/"
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
                    relative_rot_angle=rel_rot_ang,
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


def cal_ccd_zp_xh(eng, mag, zp_cfg=None):
    """
    INPUT:
        eng: X-ray energy in keV
        mag: zone plate magnification
    OUTPUTS:
        p: sample-zp distance in mm
        det_pos: detector-sample distance in mm
        wl: X-ray wavelength
        na: numerical aperture
        f: zp focal length in mm
    """
    if zp_cfg is None:
        zp_cfg = {
            "D": 244,
            "dr": 30,
        }  # 'D': zp diameter in um; 'dr': outmost zone width in nm
    # wl = 12.39847/eng/10  # in nm
    wl = 6.6261e-34 * 299792458 / (1.602176565e-19 * eng) * 1e6  # in nm
    na = wl / 2 / zp_cfg["dr"]  # numberical apeture in rad
    dof = wl / (na**2) / 1000  # depth of focus in um
    f = zp_cfg["dr"] * zp_cfg["D"] / wl / 1000  # focal length in mm
    p = f * (mag + 1) / mag  # object distance from zone plate in mm
    q = p * mag  # detector distance from zone plate in mm
    det_pos = p + q  # ccd detector distance from the sample in mm
    return p, det_pos, wl, na, f


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
    dets = [Andor, ic3]

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
        # yield from mv(Andor.cam.acquire_period, exposure_time)

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
        "detectors": [det.name for det in dets],
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

    @stage_decorator(list(dets) + motors)
    @run_decorator(md=_md)
    def zps_motor_scan_inner():
        # take dark image
        print("take 5 dark image")
        yield from _take_dark_image(dets, motors, num_dark=5)

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
            yield from _take_image(dets, motors, 1)

        print("moving sample out to take 5 background image")
        yield from _take_bkg_image(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            dets,
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
    sam_in_pos_list=[
        [0, 0, 0, 0],
    ],
    sam_out_pos_list=[
        [0, 0, 0, 0],
    ],
    exposure=[0.05],
    period=[0.05],
    rel_rot_ang=182,
    rs=1,
    eng=None,
    note="",
    flts=[],
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
                relative_rot_angle=rel_rot_ang,
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
                filters=flts,
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
    rel_rot_ang=180,
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
    flts=[],
    md=None,
):
    """
    Inputs:
    -------
    exposure_time: float, in unit of sec

    rel_rot_ang: float,
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

    dets = [Andor, ic3]
    taxi_ang = -0.5 * rs
    cur_rot_ang = zps.pi_r.position

    tgt_rot_ang = cur_rot_ang + rel_rot_ang
    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motor],
        "XEng": XEng.position,
        "ion_chamber": ic3.name,
        "plan_args": {
            "exposure_time": exposure_time,
            "relative_rot_angle": rel_rot_ang,
            "period": period,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "rs": rs,
            "relative_move_flag": relative_move_flag,
            "traditional_sequence_flag": traditional_sequence_flag,
            "filters": [filt.name for filt in flts] if flts else "None",
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

    @stage_decorator(list(dets) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        # close shutter, dark images: numer=chunk_size (e.g.20)
        print("\nshutter closed, taking dark images...")
        yield from _set_andor_param(
            exposure_time=exposure_time, period=period, chunk_size=20
        )
        yield from _take_dark_image(dets, motor, num_dark=1, simu=simu)
        yield from bps.sleep(1)
        yield from _set_andor_param(
            exposure_time=exposure_time, period=period, chunk_size=chunk_size
        )

        # open shutter, tomo_images
        yield from _open_shutter(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from mv(zps.pi_r, cur_rot_ang + taxi_ang)
        status = yield from abs_set(zps.pi_r, tgt_rot_ang, wait=False)
        yield from bps.sleep(1)
        while not status.done:
            yield from trigger_and_read(list(dets) + motor)
        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=30)
        #        yield from abs_set(zps.pi_r.velocity, rs)
        yield from select_filters(flts)
        yield from bps.sleep(1)
        yield from _set_andor_param(
            exposure_time=exposure_time, period=period, chunk_size=20
        )
        yield from _take_bkg_image(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            dets,
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
        yield from select_filters([])

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

    dets = [Andor, ic3]
    # taxi_ang = 0 #-0.5 * rs * np.sign(rel_rot_ang)
    cur_rot_ang = zps.pi_r.position

    tgt_rot_ang = end_rot_angle
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

    @stage_decorator(list(dets) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        yield from _open_shutter(simu=simu)
        status = yield from abs_set(zps.pi_r, tgt_rot_ang, wait=False)
        while not status.done:
            yield from trigger_and_read(list(dets) + motor)

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
    dets = [Andor, ic3]
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

    @stage_decorator(list(dets) + motor)
    @run_decorator(md=_md)
    def inner_scan():
        yield from _set_andor_param(
            exposure_time=exposure_time, period=period, chunk_size=chunk_size
        )
        yield from _take_dark_image(dets, motor, num_dark=1, simu=simu)

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

    dets = [Andor, ic3]
    cur_rot_ang = zps.pi_r.position

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

    @stage_decorator(list(dets) + motor)
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
            dets,
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
        print(f"sleep for {sleep_time} sec ...")
        yield from bps.sleep(sleep_time)


def scan_change_expo_time(
    x_range,
    y_range,
    t1,
    t2,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    img_sizeX=2560,
    img_sizeY=2160,
    pxl=20,
    relative_move_flag=1,
    note="",
    simu=False,
    sleep_time=0,
    md=None,
):
    """
    take image
    """
    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    dets = [Andor, ic3]

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
        "detectors": [det.name for det in dets],
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

    @stage_decorator(list(dets) + motor)
    @run_decorator(md=_md)
    def inner():
        # take dark image
        print(f"take 5 dark image with exposure = {t1}")
        yield from _set_andor_param(exposure_time=t1, period=t1, chunk_size=1)
        yield from _take_dark_image(dets, motor, num_dark=5, simu=simu)
        print(f"take 5 dark image with exposure = {t2}")
        yield from _set_andor_param(exposure_time=t2, period=t2, chunk_size=1)
        yield from _take_dark_image(dets, motor, num_dark=5, simu=simu)

        print("open shutter ...")
        yield from _open_shutter(simu)
        for ii in np.arange(x_range[0], x_range[1] + 1):
            for jj in np.arange(y_range[0], y_range[1] + 1):
                yield from mv(zps.sx, motor_x_ini + ii * img_sizeX * pxl * 1.0 / 1000)
                yield from mv(zps.sy, motor_y_ini + jj * img_sizeY * pxl * 1.0 / 1000)
                yield from bps.sleep(0.1)
                print(f"set exposure time = {t1}")
                yield from _set_andor_param(exposure_time=t1, period=t1, chunk_size=1)
                yield from bps.sleep(sleep_time)
                yield from _take_image(dets, motor, 1)
                print(f"set exposure time = {t2}")
                yield from _set_andor_param(exposure_time=t2, period=t2, chunk_size=1)
                yield from bps.sleep(sleep_time)
                yield from _take_image(dets, motor, 1)
                print(f"take bkg image with exposure time = {t1}")
                yield from _set_andor_param(exposure_time=t1, period=t1, chunk_size=1)
                yield from bps.sleep(sleep_time)
                yield from _take_bkg_image(
                    motor_x_out,
                    motor_y_out,
                    motor_z_out,
                    motor_r_out,
                    dets,
                    motor,
                    num_bkg=5,
                    simu=simu,
                )
                print(f"take bkg image with exposure time = {t2}")
                yield from _set_andor_param(exposure_time=t2, period=t2, chunk_size=1)
                yield from bps.sleep(sleep_time)
                yield from _take_bkg_image(
                    motor_x_out,
                    motor_y_out,
                    motor_z_out,
                    motor_r_out,
                    dets,
                    motor,
                    num_bkg=5,
                    simu=simu,
                )

        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            repeat=1,
            trans_first_flag=0,
        )

        print("closing shutter")
        yield from _close_shutter(simu)

    yield from inner()
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)
