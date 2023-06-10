from bluesky.plan_stubs import kickoff, collect, complete, wait
from bluesky.utils import short_uid


def tomo_zfly(
    scn_mode=0,
    exp_t=0.05,
    acq_p=0.05,
    ang_s=0,
    ang_e=180,
    vel=3,
    acc_t=1,
    num_swing=1,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    rel_out_flag=True,
    flts=[],
    rot_back_velo=30,
    bin_fac=None,
    note="",
    md=None,
    simu=False,
    sleep=0,
    cam=Andor,
    flyer=tomo_flyer,
):
    """_summary_

    Args:
        scn_mode (int, optional): _description_. Defaults to 0.
        exp_t (float, optional): _description_. Defaults to 0.05.
        acq_p (float, optional): _description_. Defaults to 0.05.
        ang_s (float or None, optional): _description_. Defaults to None.
        ang_e (float, optional): _description_. Defaults to 180.
        vel (int, optional): _description_. Defaults to 3.
        acc_t (float, optional): _description_. Defaults to 1.
        num_swing (int, optional): _description_. Defaults to 1.
        out_x (float, optional): _description_. Defaults to None.
        out_y (float, optional): _description_. Defaults to None.
        out_z (float, optional): _description_. Defaults to None.
        out_r (float, optional): _description_. Defaults to None.
        flts (list, optional): _description_. Defaults to [].
        rot_back_velo (int, optional): _description_. Defaults to 30.
        binning (int, optional): _description_. Defaults to None.
        note (str, optional): _description_. Defaults to "".
        md (dict, optional): _description_. Defaults to None.
        simu (bool, optional): _description_. Defaults to False.
        cam (ophyd.Device, optional): detector; choose between Andor, Marana, and Oryx.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """
    global ZONE_PLATE
    sleep_plan = _schedule_sleep(sleep, num_swing)
    if not sleep_plan:
        print(f"A wrong sleep pattern {sleep=} and {num_swing=} breaks the scan. Quit")
        return
    
    mots = [zps.sx, zps.sy, zps.sz]
    flyer.detectors = [
        cam,
    ]
    flyer.scn_mode = flyer.scn_modes[scn_mode]
    scn_cfg = FXITomoFlyer.compose_scn_cfg(
        scn_mode,
        exp_t,
        acq_p,
        bin_fac,
        ang_s,
        ang_e,
        vel,
        acc_t,
        rot_back_velo,
        num_swing,
    )
    scn_cfg, pc_cfg = yield from flyer.preset_flyer(scn_cfg)
    (x_ini, y_ini, z_ini, r_ini) = FXITomoFlyer.get_txm_cur_pos()
    (mot_x_out, mot_y_out, mot_z_out, mot_r_out) = FXITomoFlyer.def_abs_out_pos(
        out_x, out_y, out_z, out_r, rel_out_flag
    )

    _md = {
        "detectors": [flyer.detectors[0].name],
        "motors": [mot.name for mot in mots],
        "XEng": XEng.position,
        "storage_ring_current (mA)": round(sr_current.get(), 1),
        "plan_args": {
            "exposure_time": scn_cfg["exp_t"],
            "start_angle": scn_cfg["ang_s"],
            "end_angle": scn_cfg["ang_e"],
            "acquisition_period": scn_cfg["acq_p"],
            "slew_speed": scn_cfg["vel"],
            "mv_back_vel": scn_cfg["mb_vel"],
            "acceleration": scn_cfg["tacc"],
            "number_of_swings": scn_cfg["num_swing"],
            "out_x": mot_x_out,
            "out_y": mot_y_out,
            "out_z": mot_z_out,
            "out_r": mot_r_out,
            "filters": ["filter{}".format(t) for t in flts] if flts else "None",
            "binning": 0 if scn_cfg["bin_fac"] is None else scn_cfg["bin_fac"],
            "note": note if note else "None",
            "sleep": sleep,
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "tomo_zfly",
        "num_bkg_images": 10,
        "num_dark_images": 10,
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        "zone_plate": ZONE_PLATE,
    }
    _md.update(md or {})
    print("preset done")

    @stage_decorator(list(mots))
    @run_decorator(md=_md)
    def inner_fly_plan():
        yield from select_filters(flts)
        if flyer.scn_mode == "snaked: single file":
            yield from FXITomoFlyer.set_cam_step_for_scan(cam, scn_cfg)
            yield from FXITomoFlyer.set_mot_r_step_for_scan(scn_cfg)
            yield from _open_shutter_xhx(simu)
            for d in flyer.detectors:
                try:
                    d.stage()
                except:
                    d.unstage()
                    d.stage()
            st = yield from kickoff(flyer, wait=True, scn_cfg=scn_cfg)
            st.wait(timeout=10)

            det_stream = short_uid("dets")
            for d in flyer.detectors:
                yield from bps.trigger(d, group=det_stream)
            wait(det_stream)

            yield from abs_set(flyer.encoder.pc.arm, 1, wait=True)
            t0 = ttime.monotonic()
            for ii in range(scn_cfg["num_swing"]):
                yield from abs_set(
                    zps.pi_r,
                    scn_cfg["ang_e"] + scn_cfg["rot_dir"] * scn_cfg["taxi_dist"],
                    wait=True,
                )
                (scn_cfg["ang_s"], scn_cfg["ang_e"]) = (
                    scn_cfg["ang_e"],
                    scn_cfg["ang_s"],
                )
                scn_cfg["rot_dir"] *= -1
                if ii == scn_cfg["num_swing"] - 1:
                    set_and_wait(flyer.encoder.pc.disarm, 1)

            t1 = ttime.monotonic()
            while int(flyer.encoder.pc.gated.get()):
                if ttime.monotonic() - t1 > 60:
                    print("Scan finished abnormally. Quit!")
                    return
                yield from bps.sleep(flyer._staging_delay)
                print(ttime.time())
            print(f"Scan # {ii} takes {ttime.monotonic() - t0} seconds.")
            st = yield from complete(flyer, wait=True)
            st.wait(timeout=10)
            yield from collect(flyer)
            for d in flyer.detectors:
                try:
                    d.unstage()
                except:
                    print(f"Cannot unstage detector {d.name}")
                    return
        else:
            for ii in range(scn_cfg["num_swing"]):
                yield from FXITomoFlyer.set_cam_step_for_scan(cam, scn_cfg)
                yield from FXITomoFlyer.set_mot_r_step_for_scan(scn_cfg)
                yield from _open_shutter_xhx(simu)
                for d in flyer.detectors:
                    try:
                        d.stage()
                    except:
                        d.unstage()
                        d.stage()
                st = yield from kickoff(flyer, wait=True, scn_cfg=scn_cfg)
                st.wait(timeout=10)
                yield from abs_set(
                    flyer.encoder.pc.gate_start, scn_cfg["ang_s"], wait=True
                )

                det_stream = short_uid("dets")
                for d in flyer.detectors:
                    yield from bps.trigger(d, group=det_stream)
                wait(det_stream)

                yield from abs_set(flyer.encoder.pc.arm, 1, wait=True)
                t0 = ttime.monotonic()
                yield from abs_set(
                    zps.pi_r,
                    scn_cfg["ang_e"] + scn_cfg["rot_dir"] * scn_cfg["taxi_dist"],
                    wait=True,
                )

                t1 = ttime.monotonic()
                while int(flyer.encoder.pc.gated.get()):
                    if ttime.monotonic() - t1 > 60:
                        print("Scan finished abnormally. Quit!")
                        return
                    yield from bps.sleep(flyer._staging_delay)
                    print(ttime.time())
                print(f"Scan # {ii} takes {ttime.monotonic() - t0} seconds.")
                st = yield from complete(flyer, wait=True)
                st.wait(timeout=10)
                yield from collect(flyer)
                for d in flyer.detectors:
                    try:
                        d.unstage()
                    except:
                        print(f"Cannot unstage detector {d.name}")
                        return None

                if scn_cfg["num_swing"] > 1:
                    (scn_cfg["ang_s"], scn_cfg["ang_e"]) = (
                        scn_cfg["ang_e"],
                        scn_cfg["ang_s"],
                    )
                    scn_cfg["rot_dir"] *= -1
                    pc_cfg[flyer.scn_mode]["gate_start"] = scn_cfg["ang_s"]
                    pc_cfg[flyer.scn_mode]["dir"] = flyer.pc_trig_dir[
                        int(scn_cfg["rot_dir"])
                    ]
                    yield from flyer.set_pc_step_for_scan(flyer.scn_mode, pc_cfg)
                
                if ii < scn_cfg["num_swing"] - 1:
                    print(f"Sleeping {sleep_plan[ii]} seconds before {ii}th scan ...")
                    bps.sleep(sleep_plan[ii])

        scn_cfg["ang_s"] = r_ini
        yield from FXITomoFlyer.init_mot_r(scn_cfg)

        yield from FXITomoFlyer.set_cam_mode(cam, stage="ref-scan")
        yield from _take_ref_image(
            [cam],
            mots_pos={
                "x": mot_x_out,
                "y": mot_y_out,
                "z": mot_z_out,
                "r": mot_r_out,
            },
            num=1,
            chunk_size=10,
            stream_name="flat",
            simu=simu,
        )
        yield from _take_ref_image(
            [cam],
            mots_pos={},
            num=1,
            chunk_size=10,
            stream_name="dark",
            simu=simu,
        )
        for d in flyer.detectors:
            try:
                d.unstage()
            except:
                print(f"Cannot unstage detector {d.name}")
                return None
        yield from _move_sample(
            x_ini,
            y_ini,
            z_ini,
            r_ini,
            repeat=2,
        )
        yield from FXITomoFlyer.set_cam_mode(cam, stage="post-scan")
        yield from select_filters([])

    uid = yield from inner_fly_plan()
    print("scan finished")
    return uid


def tomo_grid_zfly(
    scn_mode=0,
    exp_t=0.05,
    acq_p=0.05,
    ang_s=0,
    ang_e=180,
    vel=3,
    acc_t=1,
    grid_nodes={},
    num_swing=1,
    out_x=None,
    out_y=None,
    out_z=None,
    out_r=None,
    rel_out_flag=True,
    flts=[],
    rot_back_velo=30,
    bin_fac=None,
    note="",
    md=None,
    sleep=0,
    simu=False,
    cam=Andor,
    flyer=tomo_flyer,
):
    """_summary_

    Args:
        scn_mode (int, optional): _description_. Defaults to 0.
        exp_t (float, optional): _description_. Defaults to 0.05.
        acq_p (float, optional): _description_. Defaults to 0.05.
        ang_s (float or None, optional): _description_. Defaults to None.
        ang_e (float, optional): _description_. Defaults to 180.
        vel (int, optional): _description_. Defaults to 3.
        acc_t (float, optional): _description_. Defaults to 1.
        grid_nodes (dic, optional): a dictionary with two items, one is
            a list of motors that loop with rotary stage; another is a table
            that lists all the grid nodes.
        num_swing (int, optional): _description_. Defaults to 1.
        out_x (float, optional): _description_. Defaults to None.
        out_y (float, optional): _description_. Defaults to None.
        out_z (float, optional): _description_. Defaults to None.
        out_r (float, optional): _description_. Defaults to None.
        flts (list, optional): _description_. Defaults to [].
        rot_back_velo (int, optional): _description_. Defaults to 30.
        binning (int, optional): _description_. Defaults to None.
        note (str, optional): _description_. Defaults to "".
        md (dict, optional): _description_. Defaults to None.
        simu (bool, optional): _description_. Defaults to False.
        cam (ophyd.Device, optional): detector; choose between Andor, Marana, and Oryx.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """
    global ZONE_PLATE
    sleep_plan = _schedule_sleep(sleep, num_swing)
    if not sleep_plan:
        print(f"A wrong sleep pattern {sleep=} and {num_swing=} breaks the scan. Quit")
        return
    mots = [zps.sx, zps.sy, zps.sz]
    flyer.detectors = [
        cam,
    ]
    flyer.scn_mode = flyer.scn_modes[scn_mode]
    scn_cfg = FXITomoFlyer.compose_scn_cfg(
        scn_mode,
        exp_t,
        acq_p,
        bin_fac,
        ang_s,
        ang_e,
        vel,
        acc_t,
        rot_back_velo,
        num_swing,
    )
    scn_cfg, pc_cfg = yield from flyer.preset_flyer(scn_cfg)
    (x_ini, y_ini, z_ini, r_ini) = FXITomoFlyer.get_txm_cur_pos()
    (mot_x_out, mot_y_out, mot_z_out, mot_r_out) = FXITomoFlyer.def_abs_out_pos(
        out_x, out_y, out_z, out_r, rel_out_flag
    )

    _md = {
        "detectors": [flyer.detectors[0].name],
        "motors": [mot.name for mot in mots],
        "XEng": XEng.position,
        "storage_ring_current (mA)": round(sr_current.get(), 1),
        "plan_args": {
            "exposure_time": scn_cfg["exp_t"],
            "start_angle": scn_cfg["ang_s"],
            "end_angle": scn_cfg["ang_e"],
            "acquisition_period": scn_cfg["acq_p"],
            "slew_speed": scn_cfg["vel"],
            "mv_back_vel": scn_cfg["mb_vel"],
            "acceleration": scn_cfg["tacc"],
            "number_of_swings": scn_cfg["num_swing"],
            "grid_mots": "none"
            if not grid_nodes
            else [mot.name for mot in grid_nodes["mots"]],
            "grid_nodes": "none" if not grid_nodes else grid_nodes["pos"],
            "out_x": mot_x_out,
            "out_y": mot_y_out,
            "out_z": mot_z_out,
            "out_r": mot_r_out,
            "rel_out_flag": rel_out_flag,
            "filters": ["filter{}".format(t) for t in flts] if flts else "None",
            "binning": 0 if scn_cfg["bin_fac"] is None else scn_cfg["bin_fac"],
            "note": note if note else "None",
            "zone_plate": ZONE_PLATE,
        },
        "plan_name": "tomo_grid_zfly",
        "num_bkg_images": 10,
        "num_dark_images": 10,
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        "zone_plate": ZONE_PLATE,
    }
    _md.update(md or {})

    if grid_nodes:
        all_mots = list(set(list(mots) + list(grid_nodes["mots"])))
    else:
        all_mots = list(list(mots))
    print("preset done")

    @run_decorator(md=_md)
    def inner_fly_plan():
        yield from select_filters(flts)
        for jj in grid_nodes["pos"] if grid_nodes else range(1):
            if grid_nodes:
                yield from mv(*zip(grid_nodes["mots"], jj))
            for mot in all_mots:
                mot.stage()

            if flyer.scn_mode == "snaked: single file":
                print(
                    "Scan mode 'snaked: single file' is not currently supported. Quit!"
                )
                yield from select_filters([])
                yield from _move_sample(
                    x_ini,
                    y_ini,
                    z_ini,
                    r_ini,
                    repeat=2,
                )
                return
            else:
                for ii in range(scn_cfg["num_swing"]):
                    yield from FXITomoFlyer.set_cam_step_for_scan(cam, scn_cfg)
                    yield from FXITomoFlyer.set_mot_r_step_for_scan(scn_cfg)
                    yield from _open_shutter_xhx(simu)
                    for d in flyer.detectors:
                        try:
                            d.stage()
                        except:
                            d.unstage()
                            d.stage()
                    st = yield from kickoff(flyer, wait=True, scn_cfg=scn_cfg)
                    st.wait(timeout=10)
                    yield from abs_set(
                        flyer.encoder.pc.gate_start, scn_cfg["ang_s"], wait=True
                    )

                    det_stream = short_uid("dets")
                    for d in flyer.detectors:
                        yield from bps.trigger(d, group=det_stream)
                    wait(det_stream)

                    yield from abs_set(flyer.encoder.pc.arm, 1, wait=True)
                    t0 = ttime.monotonic()
                    yield from abs_set(
                        zps.pi_r,
                        scn_cfg["ang_e"] + scn_cfg["rot_dir"] * scn_cfg["taxi_dist"],
                        wait=True,
                    )

                    t1 = ttime.monotonic()
                    while int(flyer.encoder.pc.gated.get()):
                        if ttime.monotonic() - t1 > 60:
                            print("Scan finished abnormally. Quit!")
                            return
                        yield from bps.sleep(flyer._staging_delay)
                        print(ttime.time())
                    print(f"Scan # {ii} takes {ttime.monotonic() - t0} seconds.")
                    st = yield from complete(flyer, wait=True)
                    st.wait(timeout=10)
                    yield from collect(flyer)
                    for d in flyer.detectors:
                        try:
                            d.unstage()
                        except:
                            print(f"Cannot unstage detector {d.name}")
                            return None

                    if scn_cfg["num_swing"] > 1:
                        (scn_cfg["ang_s"], scn_cfg["ang_e"]) = (
                            scn_cfg["ang_e"],
                            scn_cfg["ang_s"],
                        )
                        scn_cfg["rot_dir"] *= -1
                        pc_cfg[flyer.scn_mode]["gate_start"] = scn_cfg["ang_s"]
                        pc_cfg[flyer.scn_mode]["dir"] = flyer.pc_trig_dir[
                            int(scn_cfg["rot_dir"])
                        ]
                        yield from flyer.set_pc_step_for_scan(flyer.scn_mode, pc_cfg)

                    if ii < scn_cfg["num_swing"] - 1:
                        print(f"Sleeping {sleep_plan[ii]} seconds before {ii}th scan ...")
                        bps.sleep(sleep_plan[ii])

            for mot in all_mots:
                mot.unstage()

            yield from FXITomoFlyer.set_cam_mode(cam, stage="ref-scan")
            yield from _take_ref_image(
                [cam],
                mots_pos={
                    "x": mot_x_out,
                    "y": mot_y_out,
                    "z": mot_z_out,
                    "r": mot_r_out,
                },
                num=1,
                chunk_size=10,
                stream_name="flat",
                simu=simu,
            )
            yield from _take_ref_image(
                [cam],
                mots_pos={},
                num=1,
                chunk_size=10,
                stream_name="dark",
                simu=simu,
            )
            for d in flyer.detectors:
                try:
                    d.unstage()
                except:
                    print(f"Cannot unstage detector {d.name}")
                    return None
            yield from _move_sample(
                x_ini,
                y_ini,
                z_ini,
                r_ini,
                repeat=2,
            )
            yield from FXITomoFlyer.set_cam_mode(cam, stage="post-scan")
            yield from select_filters([])

    uid = yield from inner_fly_plan()
    print("scan finished")
    return uid


def _schedule_sleep(sleep, num_scan):
    sleep_plan = {}
    for ii in range(1, num_scan - 1):
        if isinstance(sleep, list):
            if len(sleep) != num_scan - 2:
                print(
                    f"The list of sleep time has length {len(sleep)} that is inconsistent \
                        with the number of scans {num_scan}. \
                            The length of sleep time should be {num_scan - 2}"
                )
                return False
            else:
                sleep_plan[ii] = sleep[ii]  
                return sleep_plan     
        elif isinstance(sleep, int) or isinstance(sleep, float):
            sleep_plan[ii] = sleep
            return sleep_plan
        else:
            print(f"Unrecognized sleep pattern {sleep}. Quit.")
            return False

