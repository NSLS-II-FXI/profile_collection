print(f"Loading {__file__}...")

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

    r_ini = zps.pi_r.position

    for i in range(repeat):
        if rot_first_flag:
            if np.abs(r_ini - r_out) > 0.5:
                yield from mv(zps.pi_r, r_out)
            yield from mv(zps.sx, x_out, zps.sy, y_out, zps.sz, z_out)
        else:
            yield from mv(zps.sx, x_out, zps.sy, y_out, zps.sz, z_out)
            if np.abs(r_ini - r_out) > 0.5:
                yield from mv(zps.pi_r, r_out)


def _move_sample_in(in_x, in_y, in_z, in_r, repeat=1, trans_first_flag=1):
    """
    move in at absolute position
    """
    r_ini = zps.pi_r.position
    for i in range(repeat):
        if trans_first_flag:
            yield from mv(zps.sx, in_x, zps.sy, in_y, zps.sz, in_z)
            if np.abs(r_ini - in_r) > 0.5:
                yield from mv(zps.pi_r, in_r)
        else:
            if np.abs(r_ini - in_r) > 0.5:
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
        yield from detector.unstage()
    yield from mv(KinetixU.cam.acquire, 0)
    yield from bps.sleep(0.2)
    yield from mv(KinetixU.cam.image_mode, 0)
    yield from bps.sleep(0.2)
    yield from mv(KinetixU.cam.num_images, chunk_size)
    for detector in detectors:
        yield from detector.stage()


def _set_cam_chunk_size(detectors, chunk_size, scan_type='fly'):
    if detectors[0].cam.num_images.value == chunk_size:
        return
    print(detectors[0])
    cam_name = _get_cam_model(detectors[0])
    image_mode_id, trigger_mode_id = _get_image_and_trigger_mode_ids(
            cam_name, scan_type=scan_type
            )
    print('change chunk size')
    print(image_mode_id, trigger_mode_id)


    for detector in detectors:
        yield from bps.unstage(detector)
        yield from bps.sleep(0.2)

    yield from mv(detectors[0].cam.acquire, 0)
    yield from bps.sleep(0.2)
    yield from mv(detectors[0].cam.image_mode, image_mode_id)
    yield from bps.sleep(0.2)
    yield from mv(detectors[0].cam.trigger_mode, trigger_mode_id)
    yield from bps.sleep(0.2)
    yield from mv(detectors[0].cam.num_images, chunk_size)
    # bps.configure re-prepares all stream descriptors that include this detector using
    # _current_stream_cache. That cache only holds objects from the most recently active
    # stream, so any stream set up WITHOUT motors (e.g. "flat") must not be the last
    # active stream before this call — otherwise descriptor re-preparation raises a
    # KeyError for motors. See test_scan2 for the enforced dark → primary → background order.
    yield from bps.configure(detectors[0], {})
    yield from bps.sleep(0.2)

    for detector in detectors:
        yield from bps.sleep(0.2)
        yield from bps.stage(detector)


def _take_dark_image(
    detectors, motor, num=1, chunk_size=1, stream_name="dark", simu=False
):
    yield from _close_shutter(simu)
    yield from _set_cam_chunk_size(detectors, chunk_size)
    yield from _take_image(detectors, motor, num, stream_name=stream_name)


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
    yield from _set_cam_chunk_size(detectors, chunk_size)
    yield from _take_image(detectors, motor, num, stream_name=stream_name)


def _set_cam_param(exposure_time=0.1, period=0.1, chunk_size=1, binning=[1, 1], cam=None):
    cam = _sel_cam(cam)
    image_mode_id, trigger_mode_id = _get_image_and_trigger_mode_ids(
        _get_cam_model(cam), scan_type='fly'
        )
    print(image_mode_id, trigger_mode_id)
    yield from mv(cam.cam.trigger_mode, trigger_mode_id)

    for i in range(2):
        yield from mv(cam.cam.acquire, 0)
        yield from bps.sleep(0.2)
    for i in range(2):
        yield from mv(cam.cam.image_mode, image_mode_id)
        yield from bps.sleep(0.5)
    yield from mv(cam.cam.num_images, chunk_size)
    period_cor = max(period, exposure_time+0.024)

    yield from mv(cam.cam.acquire_time, exposure_time)
    yield from mv(cam.cam.acquire_period, period_cor)


def _xanes_per_step(
    eng,
    detectors,
    motor,
    move_flag=1,
    move_clens_flag=1,
    info_flag=0,
    stream_name="primary",
    mag=None,
):
    yield from move_zp_ccd_TEST(
        eng, move_flag=move_flag, move_clens_flag=move_clens_flag, info_flag=info_flag, mag=mag,
    )
    yield from bps.sleep(0.1)
    if not (type(detectors) == list):
        detectors = list(detectors)
    if not (type(motor) == list):
        motor = list(motor)
    yield from trigger_and_read(detectors + motor, name=stream_name)


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
            yield from bps.sleep(1)
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
    if np.abs(zps.pi_r.velocity.value - rs) > 0.1:
        yield from abs_set(zps.pi_r.velocity, rs, wait=True)


def _move_sample(tgt_pos, repeat=1):
    """_summary_

    Args:
        x_pos (float): absolute position x-stage moving to
        y_pos (float): absolute position y-stage moving to
        z_pos (float): absolute position z-stage moving to
        r_pos (float): absolute position r-stage moving to
        repeat (int, optional): number of trials. Defaults to 1.
    """
    for i in range(repeat):
        yield from mv(zps.pi_r, tgt_pos['r'])
        yield from mv(zps.sx, tgt_pos['x'], zps.sy, tgt_pos['y'], zps.sz, tgt_pos['z'])


def _set_cam_chunk_size_xhx(cam, chunk_size, scan_type='fly'):
    cam_name = _get_cam_model(cam)
    image_mode_id, trigger_mode_id = _get_image_and_trigger_mode_ids(
        cam_name, scan_type=scan_type
        )

    yield from mv(cam.cam.acquire, 0)
    if cam.cam.image_mode.value != image_mode_id:
        yield from mv(cam.cam.image_mode, image_mode_id)
    if cam.cam.trigger_mode.value != trigger_mode_id:
        yield from mv(cam.cam.trigger_mode, trigger_mode_id)
    if cam.cam.num_images.value != chunk_size:
        yield from mv(cam.cam.num_images, chunk_size)


def _take_image_xh(detectors, motor, stream_name="primary"):
    if not (type(detectors) == list):
        detectors = list(detectors)
    if not (type(motor) == list):
        motor = list(motor)
    readings = yield from trigger_and_read(detectors + motor, name=stream_name)
    return readings


def _take_ref_image(
    dets,
    mots_pos = {},
    chunk_size=1,
    stream_name="flat",
    simu=False,
    stage=True,
):
    print(f"ref set cam starts at {ttime.asctime()}")
    yield from _set_cam_chunk_size_xhx(dets[0], chunk_size, scan_type='fly')
    print(f"ref set cam finishes at {ttime.asctime()}")

    if stage:
        for d in dets:
            try:
                d.stage()
            except Exception as e:
                print(f"error: {e}")
                d.unstage()
                d.stage()
    
    if stream_name == "flat":
        print(f"ref move sam starts at {ttime.asctime()}")
        yield from _move_sample(
            mots_pos, repeat=2
        )
        print(f"ref open shutter starts at {ttime.asctime()}")
        yield from _open_shutter_xhx(simu)
        print(f"ref open shutter finishes at {ttime.asctime()}")
    elif stream_name == "dark":
        print(f"ref close shutter starts at {ttime.asctime()}")
        yield from _move_sample(
            mots_pos, repeat=2
        )
        yield from _close_shutter_xhx(simu)
        print(f"ref close shutter finishes at {ttime.asctime()}")

    print(f"ref take image starts at {ttime.asctime()}")
    yield from _take_image_xh(dets, [], stream_name=stream_name)

    if stage:
        for d in dets:
            try:
                d.unstage()
            except Exception as e:
                print(f"error: {e}")
                d.unstage(d)
    print(f"ref finishes at {ttime.asctime()}")


def move_and_wait(motor, target, attr="user_setpoint", atol=0.1, timeout=10.0):
    yield from mv(motor, target)

    import time
    t0 = time.time()
    while True:
        rbv = motor.user_readback.value
        if abs(rbv - target) <= atol:
            break
        if (time.time() - t0) > timeout:
            raise TimeoutError(
                f"Motor did not reach target within {timeout}s, last rbv={rbv}"
            )
        yield from bps.sleep(0.1)


