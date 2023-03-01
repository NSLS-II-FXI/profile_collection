def _move_sample_out(out_x, out_y, out_z, out_r, repeat=1, rot_first_flag=1):
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

    for i in range(repeat):
        if rot_first_flag:
            yield from mv(zps.pi_r, r_out)
            yield from mv(zps.sx, x_out, zps.sy, y_out, zps.sz, z_out)
        else:
            yield from mv(zps.sx, x_out, zps.sy, y_out, zps.sz, z_out)
            yield from mv(zps.pi_r, r_out)


def _move_sample_in(in_x, in_y, in_z, in_r, repeat=1, trans_first_flag=1):
    """
    move in at absolute position
    """
    for i in range(repeat):
        if trans_first_flag:
            yield from mv(zps.sx, in_x, zps.sy, in_y, zps.sz, in_z)
            yield from mv(zps.pi_r, in_r)
        else:
            yield from mv(zps.pi_r, in_r)
            yield from mv(zps.sx, in_x, zps.sy, in_y, zps.sz, in_z)


def _take_image(detectors, motor, num, stream_name="primary"):
    if not (type(detectors) == list):
        detectors = list(detectors)
    if not (type(motor) == list):
        motor = list(motor)
    for i in range(num):
        yield from trigger_and_read(detectors + motor, name=stream_name)


def _set_Andor_chunk_size(detectors, chunk_size):
    for detector in detectors:
        yield from unstage(detector)
    yield from bps.configure(Andor, {"cam.num_images": chunk_size})
    for detector in detectors:
        yield from stage(detector)


def _take_dark_image(
    detectors, motor, num=1, chunk_size=1, stream_name="dark", simu=False
):
    yield from _close_shutter(simu)
    original_num_images = yield from rd(Andor.cam.num_images)
    yield from _set_Andor_chunk_size(detectors, chunk_size)
    yield from _take_image(detectors, motor, num, stream_name=stream_name)
    yield from _set_Andor_chunk_size(detectors, original_num_images)


def _take_bkg_image(
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
):
    yield from _move_sample_out(
        out_x, out_y, out_z, out_r, repeat=2, rot_first_flag=rot_first_flag
    )
    original_num_images = yield from rd(Andor.cam.num_images)
    yield from _set_Andor_chunk_size(detectors, chunk_size)
    yield from _take_image(detectors, motor, num, stream_name=stream_name)
    yield from _set_Andor_chunk_size(detectors, original_num_images)


def _set_andor_param(exposure_time=0.1, period=0.1, chunk_size=1, binning=[1, 1]):
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.image_mode, 0)
    yield from mv(Andor.cam.num_images, chunk_size)
    period_cor = period
    yield from mv(Andor.cam.acquire_time, exposure_time)
    # yield from abs_set(Andor.cam.acquire_period, period_cor)
    Andor.cam.acquire_period.put(period_cor)


def _xanes_per_step(
    eng,
    detectors,
    motor,
    move_flag=1,
    move_clens_flag=1,
    info_flag=0,
    stream_name="primary",
):
    yield from move_zp_ccd(
        eng, move_flag=move_flag, move_clens_flag=move_clens_flag, info_flag=info_flag
    )
    yield from bps.sleep(0.1)
    if not (type(detectors) == list):
        detectors = list(detectors)
    if not (type(motor) == list):
        motor = list(motor)
    yield from trigger_and_read(detectors + motor, name=stream_name)


"""        
def _close_shutter(simu=False):
    if simu:
        print("testing: close shutter")
    else:
        yield from mv(shutter, 'Close')


def _open_shutter(simu=False):
    if simu:
        print("testing: open shutter")
    else:
        yield from mv(shutter, 'Open')      
"""


def _close_shutter(simu=False):
    if simu:
        print("testing: close shutter")
    else:
        print("closing shutter ... ")
        # yield from mv(shutter, 'Close')
        i = 0
        reading = yield from bps.rd(shutter_status)
        while not reading:  # if 1:  closed; if 0: open
            yield from abs_set(shutter_close, 1, wait=True)
            yield from bps.sleep(3)
            i += 1
            print(f"try closing {i} time(s) ...")
            if i > 20:
                print("fails to close shutter")
                raise Exception("fails to close shutter")
                break
            reading = yield from bps.rd(shutter_status)


def _open_shutter(simu=False):
    if simu:
        print("testing: open shutter")
    else:
        print("opening shutter ... ")
        i = 0
        reading = yield from bps.rd(shutter_status)
        while reading:  # if 1:  closed; if 0: open
            yield from abs_set(shutter_open, 1, wait=True)
            print(f"try opening {i} time(s) ...")
            yield from bps.sleep(1)
            i += 1
            if i > 5:
                print("fails to open shutter")
                raise Exception("fails to open shutter")
                break
            reading = yield from bps.rd(shutter_status)


def _set_rotation_speed(rs=30):
    yield from abs_set(zps.pi_r.velocity, rs)
    yield from bps.sleep(0.5)
    yield from abs_set(zps.pi_r.velocity, rs)


#
