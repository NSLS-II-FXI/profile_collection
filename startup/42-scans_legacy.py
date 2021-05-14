def _close_shutter_legacy(simu=False):
    if simu:
        print("testing: close shutter")
    else:
        print("closing shutter ... ")
        # yield from mv(shutter, 'Close')
        i = 0
        reading = (yield from bps.rd(shutter_status))
        while not reading:  # if 1:  closed; if 0: open
            yield from abs_set(shutter_close, 1, wait=True)
            yield from bps.sleep(3)
            i += 1
            print(f"try closing again ...")
            if i > 20:
                print("fails to close shutter")
                raise Exception("fails to close shutter")
                break


def _open_shutter_legacy(simu=False):
    if simu:
        print("testing: open shutter")
    else:
        print("opening shutter ... ")
        i = 0
        reading = (yield from bps.rd(shutter_status))
        while reading:  # if 1:  closed; if 0: open
            yield from abs_set(shutter_open, 1, wait=True)
            yield from bps.sleep(1)
            i += 1
            if i > 5:
                print("fails to open shutter")
                raise Exception("fails to open shutter")
                break
                
def tomo_scan_legacy(
    start,
    stop,
    num,
    exposure_time=1,
    bkg_num=10,
    dark_num=10,
    chunk_size=1,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    relative_move_flag=1,
    traditional_sequence_flag=1,
    note="",
    simu=False,
    md=None,
):
    """
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
    """
    global ZONE_PLATE

    detectors = [Andor, ic3]
    yield from _set_andor_param(
        exposure_time=exposure_time, period=exposure_time, chunk_size=chunk_size
    )

    motor_eng = XEng
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

    motor = [motor_eng, zps.sx, zps.sy, zps.sz, zps.pi_r]
    _md = {
        "detectors": [det.name for det in detectors],
        "x_ray_energy": XEng.position,
        "num_angles": num,
        "num_bkg_images": bkg_num,
        "num_dark_images": dark_num,
        "plan_args": {
            "start": start,
            "stop": stop,
            "num": num,
            "exposure_time": exposure_time,
            "bkg_num": bkg_num,
            "dark_num": dark_num,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "relative_move_flag": relative_move_flag,
            "note": note if note else "None",
        },
        "zone_plate": ZONE_PLATE,
        "plan_name": "tomo_scan",
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "zone_plate": ZONE_PLATE,
        "note": note if note else "None",
        # 'motor_pos':  wh_pos(print_on_screen=0),
    }

    _md.update(md or {})
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)
    steps = np.linspace(start, stop, num)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def tomo_inner_scan():
        # close shutter, dark images
        print("\nshutter closed, taking dark images...")
        yield from _close_shutter(simu)
        yield from _take_dark_image(detectors, motor, num_dark=dark_num, simu=simu)
        # Open shutter, tomo images
        yield from _open_shutter(simu)
        print("shutter opened, starting tomo_scan...")
        for step in steps:  # take tomography images
            # yield from one_1d_step(detectors, zps.pi_r, step)
            yield from mv(zps.pi_r, step)
            yield from _take_image(detectors, motor, 1)

        print("\n\nTaking background images...")
        yield from _take_bkg_image(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            detectors,
            motor,
            num_bkg=bkg_num,
            simu=False,
            traditional_sequence_flag=traditional_sequence_flag,
        )
        # close shutter, move sample back
        yield from _close_shutter(simu)
        yield from _move_sample_in(
            motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini, repeat=2
        )

    yield from tomo_inner_scan()
    print("tomo-scan is disabled, try to use fly_scan")
    
                    
def fly_scan_legacy(
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

    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    detectors = [Andor, ic3]
    offset_angle = -1 * rs
    current_rot_angle = zps.pi_r.position

    target_rot_angle = current_rot_angle + relative_rot_angle
    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motor],
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

    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.bin_y, binning[0],
                  Andor.cam.bin_x, binning[1])
    yield from _set_andor_param(
        exposure_time=exposure_time, period=period, chunk_size=chunk_size
    )
    yield from _set_rotation_speed(rs=np.abs(rs))
    print("set rotation speed: {} deg/sec".format(rs))

    @stage_decorator(list(detectors) + motor)
    @bpp.monitor_during_decorator([zps.pi_r])
    @run_decorator(md=_md)
    def fly_inner_scan():
        for flt in filters:
            yield from mv(flt, 1)
            yield from mv(flt, 1)
        yield from bps.sleep(1) 
        
        # close shutter, dark images: numer=chunk_size (e.g.20)
        print("\nshutter closed, taking dark images...")
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

        # open shutter, tomo_images
        yield from _open_shutter(simu=simu)
        print("\nshutter opened, taking tomo images...")
        yield from mv(zps.pi_r, current_rot_angle + offset_angle)
        status = yield from abs_set(zps.pi_r, target_rot_angle, wait=False)
        yield from bps.sleep(0.6)
        while not status.done:
            yield from trigger_and_read(list(detectors) + motor)
        # bkg images
        print("\nTaking background images...")
        yield from _set_rotation_speed(rs=rot_back_velo)
        #        yield from abs_set(zps.pi_r.velocity, rs)
        
        yield from _take_bkg_image(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            detectors,
            motor,
            num_bkg=1,
            simu=False,
            traditional_sequence_flag=rot_first_flag,
        )
        yield from _close_shutter(simu=simu)
        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            trans_first_flag=rot_first_flag,
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
    
    
def xanes_scan_legacy(
    eng_list,
    exposure_time=0.1,
    chunk_size=5,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    simu=False,
    relative_move_flag=1,
    note="",
    rot_first_flag=1,
    md=None,
):
    """
    Scan the energy and take 2D image, will take background after take all images for all energy points
    Example: RE(xanes_scan([8.9, 9.0, 9.1], exposure_time=0.1, bkg_num=10, dark_num=10, out_x=1, out_y=0, note='xanes scan test'))

    Inputs:
    -------
    eng_list: list or numpy array,
        energy in unit of keV

    exposure_time: float
        in unit of seconds

    chunk_size: int
        number of background images == num of dark images

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
    detectors = [Andor, ic3]
    period = exposure_time if exposure_time >= 0.05 else 0.05
    yield from _set_andor_param(exposure_time, period, chunk_size)
    motor_eng = XEng
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

    rs_ini = (yield from bps.rd(zps.pi_r.velocity))
    motor = [motor_eng, zps.sx, zps.sy, zps.sz, zps.pi_r]

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motor],
        "num_eng": len(eng_list),
        "num_bkg_images": chunk_size,
        "num_dark_images": chunk_size,
        "chunk_size": chunk_size,
        "out_x": out_x,
        "out_y": out_y,
        "out_r": out_z,
        "out_z": out_r,
        "relative_move_flag": relative_move_flag,
        "exposure_time": exposure_time,
        "eng_list": eng_list,
        "XEng": XEng.position,
        "plan_args": {
            "eng_list": "eng_list",
            "exposure_time": exposure_time,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "relative_move_flag": relative_move_flag,
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "xanes_scan",
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
    def xanes_inner_scan():
        print("\ntake {} dark images...".format(chunk_size))
        yield from _set_rotation_speed(rs=30)
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)

        print(
            "\nopening shutter, and start xanes scan: {} images per each energy... ".format(
                chunk_size
            )
        )
        yield from _open_shutter(simu)
        for eng in eng_list:
            yield from _xanes_per_step(eng, detectors, motor, move_flag=1, info_flag=0)
        yield from _move_sample_out(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            repeat=2,
            rot_first_flag=rot_first_flag,
        )
        print(
            "\ntake bkg image after xanes scan, {} per each energy...".format(
                chunk_size
            )
        )
        for eng in eng_list:
            yield from _xanes_per_step(eng, detectors, motor, move_flag=1, info_flag=0)
        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            repeat=2,
            trans_first_flag=rot_first_flag,
        )
        yield from move_zp_ccd(eng_ini, info_flag=0)

        print("closing shutter")
        yield from _close_shutter(simu)

    yield from xanes_inner_scan()
    txt1 = get_scan_parameter()
    eng_list = np.round(eng_list, 5)
    if len(eng_list) > 10:
        txt2 = f"eng_list: {eng_list[0:10]}, ... {eng_list[-5:]}\n"
    else:
        txt2 = f"eng_list: {eng_list}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)
    print(txt)


def xanes_scan2_legacy(
    eng_list,
    exposure_time=0.1,
    chunk_size=5,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    simu=False,
    relative_move_flag=1,
    note="",
    flt=[],
    rot_first_flag=1,
    md=None,
):
    """
    Different from xanes_scan:  In xanes_scan2, it moves out sample and take background image at each energy

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
    detectors = [Andor, ic3, ic4]
    period = exposure_time if exposure_time >= 0.05 else 0.05
    yield from _set_andor_param(exposure_time, period, chunk_size)
    motor_eng = XEng
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

    rs_ini = (yield from bps.rd(zps.pi_r.velocity))
    #rs_ini = zps.pi_r.velocity.get()

    motor = [motor_eng, zps.sx, zps.sy, zps.sz, zps.pi_r]

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
        "XEng": XEng.position,
        "plan_args": {
            "eng_list": "eng_list",
            "exposure_time": exposure_time,
            "chunk_size": chunk_size,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "our_r": out_r,
            "relative_move_flag": relative_move_flag,
            "note": note if note else "None",
            "filters": [t.name for t in flt if flt],
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "xanes_scan2",
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
    def xanes_inner_scan():
        yield from _set_rotation_speed(rs=30)
        # yield from abs_set(motor_r.velocity, 30)
        # take dark image
        print("\ntake {} dark images...".format(chunk_size))
        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)
        
        print(
            "\nopening shutter, and start xanes scan: {} images per each energy... ".format(
                chunk_size
            )
        )
        yield from _open_shutter(simu)

        for eng in eng_list:
            yield from _xanes_per_step(
                eng, detectors, motor, move_flag=1, move_clens_flag=0, info_flag=0
            )
            if len(flt):
                for filt in flt:
                    yield from mv(filt, 1)
                    yield from bps.sleep(0.5)
            yield from _take_bkg_image(
                motor_x_out,
                motor_y_out,
                motor_z_out,
                motor_r_out,
                detectors,
                motor,
                num_bkg=1,
                simu=simu,
                traditional_sequence_flag=rot_first_flag,
            )
            yield from _move_sample_in(
                motor_x_ini,
                motor_y_ini,
                motor_z_ini,
                motor_r_ini,
                repeat=2,
                trans_first_flag=rot_first_flag,
            )
            if len(flt):
                for filt in flt:
                    yield from mv(filt, 0)
                    yield from bps.sleep(0.5)
        yield from move_zp_ccd(eng_ini, move_flag=1, info_flag=0)
        print("closing shutter")
        yield from _close_shutter(simu=simu)

    yield from xanes_inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    txt1 = get_scan_parameter()
    eng_list = np.round(eng_list, 5)
    if len(eng_list) > 10:
        txt2 = f"eng_list: {eng_list[0:10]}, ... {eng_list[-5:]}\n"
    else:
        txt2 = f"eng_list: {eng_list}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)
    print(txt)
    
    
def fly_scan_repeat_legacy(
    exposure_time=0.03,
    start_angle = None,
    relative_rot_angle=185,
    period=0.05,
    chunk_size=20,
    x_list=[],
    y_list=[],
    z_list=[],
    out_x=0,
    out_y=-100,
    out_z=0,
    out_r=0,
    rs=6,
    note="",
    repeat=1,
    sleep_time=0,
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    rot_back_velo=30,
    md=None,
):
    nx = len(x_list)
    ny = len(y_list)
    nz = len(z_list)
    if nx == 0 & ny == 0 & nz == 0:
        for i in range(repeat):
            yield from fly_scan(
                exposure_time=exposure_time,
                start_angle = start_angle,
                relative_rot_angle=relative_rot_angle,
                period=period,
                chunk_size=chunk_size,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                out_r=out_r,
                rs=rs,
                note=note,
                simu=simu,
                relative_move_flag=relative_move_flag,
                rot_first_flag=rot_first_flag,
                rot_back_velo=rot_back_velo,
                md=md,
            )
            print(
                f"Scan at time point {i:3d} is finished; sleep for {sleep_time:3.1f} seconds now."
            )
            insert_text(
                f"Scan at time point {i:3d} is finished; sleep for {sleep_time:3.1f} seconds now."
            )
            if i != repeat - 1:
                yield from bps.sleep(sleep_time)
        export_pdf(1)
    else:
        if nx != ny or nx != nz or ny != nz:
            print(
                "!!!!! Position lists are not equal in length. Please check your position list definition !!!!!"
            )
        else:
            for i in range(repeat):
                for j in range(nx):
                    yield from mv(
                        zps.sx, x_list[j], zps.sy, y_list[j], zps.sz, z_list[j]
                    )
                    yield from fly_scan_legacy(
                        exposure_time=exposure_time,
                        start_angle=start_angle,
                        relative_rot_angle=relative_rot_angle,
                        period=period,
                        chunk_size=chunk_size,
                        out_x=out_x,
                        out_y=out_y,
                        out_z=out_z,
                        out_r=out_r,
                        rs=rs,
                        note=note,
                        simu=simu,
                        relative_move_flag=relative_move_flag,
                        rot_first_flag=rot_first_flag,
                        rot_back_velo=rot_back_velo,
                        md=md,
                    )
                insert_text(
                    f"Scan at time point {i:3d} is finished; sleep for {sleep_time:3.1f} seconds now."
                )
                print(
                    f"Scan at time point {i:3d} is finished; sleep for {sleep_time:3.1f} seconds now."
                )
                if i != repeat - 1:
                    yield from bps.sleep(sleep_time)
            export_pdf(1)
   
   
def xanes_3D_legacy(
    eng_list,
    exposure_time=0.05,
    start_angle = None,
    relative_rot_angle=185,
    period=0.05,
    chunk_size=20,
    out_x=0,
    out_y=0,
    out_z=0,
    out_r=0,
    rs=2,
    simu=False,
    relative_move_flag=1,
    rot_first_flag=1,
    note="",
    binning = [2, 2]
):
    txt = "start 3D xanes scan, containing following fly_scan:\n"
    insert_text(txt)
    yield from mv(Andor.cam.acquire, 0)
    yield from mv(Andor.cam.bin_y, binning[0],
                  Andor.cam.bin_x, binning[1])
    for eng in eng_list:
        yield from move_zp_ccd(eng, move_flag=1)
        my_note = note + f"_energy={eng}"
        yield from bps.sleep(1)
        print(f"current energy: {eng}")
        # yield from fly_scan(exposure_time, relative_rot_angle=relative_rot_angle, period=period, out_x=out_x, out_y=out_y, out_z=out_z, out_r= out_r, rs=rs, note=my_note, simu=simu, relative_move_flag=relative_move_flag, traditional_sequence_flag=traditional_sequence_flag)
        yield from fly_scan_legacy(
            exposure_time,
            start_angle=start_angle,
            relative_rot_angle=relative_rot_angle,
            period=period,
            chunk_size = chunk_size,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            rs=rs,
            relative_move_flag=relative_move_flag,
            note=my_note,
            simu=simu,
            rot_first_flag=rot_first_flag,
        )
        yield from bps.sleep(1)
    yield from mv(Andor.cam.image_mode, 1)
    export_pdf(1)
    
             
    
def multi_pos_xanes_3D_legacy(
    eng_list,
    x_list,
    y_list,
    z_list,
    r_list,
    start_angle = None,
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
    traditional_sequence_flag=1,
    note="",
    sleep_time=0,
    binning = [2, 2],
    repeat=1,
):
    n = len(x_list)
    for rep in range(repeat):
        for i in range(n):
            yield from mv(
                zps.sx,
                x_list[i],
                zps.sy,
                y_list[i],
                zps.sz,
                z_list[i],
                zps.pi_r,
                r_list[i],
            )
            txt = f"start xanes_3D at pos1: x={x_list[i]}, y={y_list[i]}, z={z_list[i]}\nrepeat:{rep}"
            insert_text(txt)
            print(f"{txt}\n##########################\n\n\n\n")
            yield from xanes_3D_legacy(
                eng_list,
                exposure_time=exposure_time,
                start_angle =start_angle,
                relative_rot_angle=relative_rot_angle,
                period=period,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                out_r=out_r,
                rs=rs,
                simu=simu,
                relative_move_flag=relative_move_flag,
                rot_first_flag=traditional_sequence_flag,
                note=note,
                binning = [2, 2],
            )
        print(f"sleep for {sleep_time} sec\n\n\n\n")
        yield from bps.sleep(sleep_time)
        
# ### Backup before modification 09/23/2019
#
# def fly_scan(exposure_time=0.1, relative_rot_angle = 180, period=0.15, chunk_size=20, out_x=None, out_y=2000, out_z=None,  out_r=None, rs=1, note='', simu=False, relative_move_flag=1, traditional_sequence_flag=1, filters=[], md=None):
#    '''
#    Inputs:
#    -------
#    exposure_time: float, in unit of sec
#
#    relative_rot_angle: float,
#        total rotation angles start from current rotary stage (zps.pi_r) position
#
#    period: float, in unit of sec
#        period of taking images, "period" should >= "exposure_time"
#
#    chunk_size: int, default setting is 20
#        number of images taken for each trigger of Andor camera
#
#    out_x: float, default is 0
#        relative movement of sample in "x" direction using zps.sx to move out sample (in unit of um)
#        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
#
#    out_y: float, default is 0
#        relative movement of sample in "y" direction using zps.sy to move out sample (in unit of um)
#        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
#
#    out_z: float, default is 0
#        relative movement of sample in "z" direction using zps.sz to move out sample (in unit of um)
#        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
#
#    out_r: float, default is 0
#        relative movement of sample by rotating "out_r" degrees, using zps.pi_r to move out sample
#        NOTE:  BE CAUSION THAT IT WILL ROTATE SAMPLE BY "out_r" FIRST, AND THEN MOVE X, Y, Z
#
#    rs: float, default is 1
#        rotation speed in unit of deg/sec
#
#    note: string
#        adding note to the scan
#
#    simu: Bool, default is False
#        True: will simulate closing/open shutter without really closing/opening
#        False: will really close/open shutter
#
#    '''
#    global ZONE_PLATE
#    motor_x_ini = zps.sx.position
#    motor_y_ini = zps.sy.position
#    motor_z_ini = zps.sz.position
#    motor_r_ini = zps.pi_r.position
#
#    if relative_move_flag:
#        motor_x_out = motor_x_ini + out_x if out_x else motor_x_ini
#        motor_y_out = motor_y_ini + out_y if out_y else motor_y_ini
#        motor_z_out = motor_z_ini + out_z if out_z else motor_z_ini
#        motor_r_out = motor_r_ini + out_r if out_r else motor_r_ini
#    else:
#        motor_x_out = out_x if out_x else motor_x_ini
#        motor_y_out = out_y if out_y else motor_y_ini
#        motor_z_out = out_z if out_z else motor_z_ini
#        motor_r_out = out_r if out_r else motor_r_ini
#
#    motor = [zps.sx, zps.sy, zps.sz, zps.pi_r]
#
#    detectors = [Andor, ic3]
#    offset_angle = -0.5 * rs
#    current_rot_angle = zps.pi_r.position
#
#    target_rot_angle = current_rot_angle + relative_rot_angle
#    _md = {'detectors': ['Andor'],
#           'motors': [mot.name for mot in motor],
#           'XEng': XEng.position,
#           'ion_chamber': ic3.name,
#           'plan_args': {'exposure_time': exposure_time,
#                         'relative_rot_angle': relative_rot_angle,
#                         'period': period,
#                         'chunk_size': chunk_size,
#                         'out_x': out_x,
#                         'out_y': out_y,
#                         'out_z': out_z,
#                         'out_r': out_r,
#                         'rs': rs,
#                         'relative_move_flag': relative_move_flag,
#                         'traditional_sequence_flag': traditional_sequence_flag,
#                         'filters': [filt.name for filt in filters] if filters else 'None',
#                         'note': note if note else 'None',
#                         'zone_plate': ZONE_PLATE,
#                        },
#           'plan_name': 'fly_scan',
#           'num_bkg_images': chunk_size,
#           'num_dark_images': chunk_size,
#           'chunk_size': chunk_size,
#           'plan_pattern': 'linspace',
#           'plan_pattern_module': 'numpy',
#           'hints': {},
#           'operator': 'FXI',
#           'note': note if note else 'None',
#           'zone_plate': ZONE_PLATE,
#           #'motor_pos': wh_pos(print_on_screen=0),
#            }
#    _md.update(md or {})
#    try:  dimensions = [(zps.pi_r.hints['fields'], 'primary')]
#    except (AttributeError, KeyError):    pass
#    else: _md['hints'].setdefault('dimensions', dimensions)
#
#    yield from _set_andor_param(exposure_time=exposure_time, period=period, chunk_size=chunk_size)
#    yield from _set_rotation_speed(rs=rs)
#    print('set rotation speed: {} deg/sec'.format(rs))
#
#
#    @stage_decorator(list(detectors) + motor)
#    @bpp.monitor_during_decorator([zps.pi_r])
#    @run_decorator(md=_md)
#    def fly_inner_scan():
#        #close shutter, dark images: numer=chunk_size (e.g.20)
#        print('\nshutter closed, taking dark images...')
#        yield from _take_dark_image(detectors, motor, num_dark=1, simu=simu)
#
#        #open shutter, tomo_images
#        yield from _open_shutter(simu=simu)
#        print ('\nshutter opened, taking tomo images...')
#        yield from mv(zps.pi_r, current_rot_angle + offset_angle)
#        status = yield from abs_set(zps.pi_r, target_rot_angle, wait=False)
#        yield from bps.sleep(1)
#        while not status.done:
#            yield from trigger_and_read(list(detectors) + motor)
#        # bkg images
#        print ('\nTaking background images...')
#        yield from _set_rotation_speed(rs=30)
#        for flt in filters:
#            yield from mv(flt, 1)
#            yield from mv(flt, 1)
#        yield from bps.sleep(1)
#        yield from _take_bkg_image(motor_x_out, motor_y_out, motor_z_out, motor_r_out, detectors, motor, num_bkg=1, simu=False, traditional_sequence_flag=traditional_sequence_flag)
#        yield from _close_shutter(simu=simu)
#        yield from _move_sample_in(motor_x_ini, motor_y_ini, motor_z_ini, motor_r_ini, trans_first_flag=traditional_sequence_flag)
#        for flt in filters:
#            yield from mv(flt, 0)
#    uid = yield from fly_inner_scan()
#    yield from mv(Andor.cam.image_mode, 1)
#    print('scan finished')
#    txt = get_scan_parameter(print_flag=0)
#    insert_text(txt)
#    print(txt)
#    return uid


