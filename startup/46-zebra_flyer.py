print(f"Loading {__file__}...")

from bluesky.plan_stubs import kickoff, collect, complete, wait
from bluesky.utils import short_uid


# ts_flat_file = EpicsSignal("NSLS2:TomoStream:FlatFileName", name="ts_flat_file")
# ts_dark_file = EpicsSignal("NSLS2:TomoStream:DarkFileName", name="ts_dark_file")
# ts_load_refs = EpicsSignal("NSLS2:TomoStream:LoadRefImages", name="ts_load_refs")
# ts_refs_loaded = EpicsSignal("NSLS2:TomoStream:RefImagesLoaded", name="ts_refs_loaded")


def tomo_zfly(
    scn_mode=0,
    exp_t=0.05,
    acq_p=0.05,
    ang_s=0,
    ang_e=180,
    vel=3,
    acc_t=1,
    num_swing=1,
    out_pos=[None, None, None, None],
    rel_out_flag=True,
    flts=[],
    rot_back_velo=30,
    bin_fac=None,
    roi={"min_x": 176, "size_x": 2048, "min_y": 176, "size_y": 2048},
    note="",
    md=None,
    simu=False,
    sleep=0,
    cam=None,
    flyer=None,
):
    """_summary_

    Args:
        scn_mode (int, optional):
            0: "standard",  # a single scan in a given angle range
            1: "snaked: multiple files",  # back-forth rocking scan with each swing being saved into a file
            2: "snaked: single file",  # back-forth rocking scan being saved into a single file
            Defaults to 0.
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
        cam (ophyd.Device, optional): detector; choose between Andor, MaranaU, and Oryx.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """
    cam = _sel_cam(cam)
    flyer = _sel_flyer(flyer)

    global ZONE_PLATE
    yield from FXITomoFlyer.stop_det(cam)
    yield from FXITomoFlyer.set_roi_det(cam, roi)

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
    out_x, out_y, out_z, out_r = out_pos
    (mot_x_out, mot_y_out, mot_z_out, mot_r_out) = FXITomoFlyer.def_abs_out_pos(
        out_x, out_y, out_z, out_r, rel_out_flag
    )
    _md = {
        "detectors": [flyer.detectors[0].name],
        "motors": [mot.name for mot in mots],
        "XEng": XEng.position,
        "storage_ring_current (mA)": round(sr_current.get(), 1),
        "plan_args": {
            "scan_mode": scn_cfg["scn_mode"],
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
    print("preset scan is done")

    # @stage_decorator(list(mots))
    @run_decorator(md=_md)
    def inner_fly_plan():
        yield from select_filters(flts)

        if flyer.scn_mode == "standard":  # scn_mode = 0
            print(f"{sleep_plan=}")
            for ii in range(scn_cfg["num_swing"]):
                yield from FXITomoFlyer.set_cam_mode(cam, stage="ref-scan")
                print(f"Scan # {ii} set ref-scan finishes at {ttime.asctime()}")

                print(f"taking dark images at {ttime.asctime()}")
                yield from _take_ref_image(
                    flyer.detectors,
                    mots_pos={
                        "x": mot_x_out,
                        "y": mot_y_out,
                        "z": mot_z_out,
                        "r": mot_r_out,
                    },
                    chunk_size=10,
                    stream_name="dark",
                    simu=simu,
                )
                print(f"{resolver.latest_dark()['file_reference']}")

                print(f"taking flat images at {ttime.asctime()}")
                yield from _take_ref_image(
                    flyer.detectors,
                    mots_pos={
                        "x": mot_x_out,
                        "y": mot_y_out,
                        "z": mot_z_out,
                        "r": mot_r_out,
                    },
                    chunk_size=10,
                    stream_name="flat",
                    simu=simu,
                )
                print(f"{resolver.latest_flat()['file_reference']}")

                print(f"Scan # {ii} set sam in position starts at {ttime.asctime()}")
                yield from _move_sample(
                    {"x": x_ini, "y": y_ini, "z": z_ini, "r": r_ini},
                    repeat=2,
                )
                print(f"Scan # {ii} set sam in position finishes at {ttime.asctime()}")

                yield from FXITomoFlyer.set_cam_mode(
                    flyer.detectors[0], stage="pre-scan"
                )
                yield from FXITomoFlyer.set_cam_step_for_scan(cam, scn_cfg)
                yield from FXITomoFlyer.set_mot_r_step_for_scan(scn_cfg)
                yield from _open_shutter_xhx(simu)
                for d in flyer.detectors:
                    try:
                        d.stage()
                    except:
                        d.unstage()
                        d.stage()
                for mot in mots:
                    mot.stage()

                print(f"{scn_cfg=}")
                st = yield from kickoff(flyer, wait=True, scn_cfg=scn_cfg)
                st.wait(timeout=10)

                det_stream = short_uid("dets")
                for d in flyer.detectors:
                    yield from bps.trigger(d, group=det_stream)
                wait(det_stream)

                set_and_wait(flyer.encoder.pc.gate_start, scn_cfg["ang_s"], rtol=0.1)
                yield from abs_set(flyer.encoder.pc.arm, 1, wait=True)

                t0 = ttime.monotonic()
                yield from move_and_wait(
                    zps.pi_r,
                    scn_cfg["ang_e"] + scn_cfg["rot_dir"] * scn_cfg["taxi_dist"],
                    atol=0.1,
                )

                t1 = ttime.monotonic()
                while int(flyer.encoder.pc.gated.get()):
                    if ttime.monotonic() - t1 > 60:
                        print("Scan finished abnormally. Quit!")
                        return
                    yield from bps.sleep(flyer._staging_delay)
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
                for mot in mots:
                    mot.unstage()

                yield from _close_shutter_xhx(simu=simu)
                print(f"Scan # {ii} cleaning at {ttime.asctime()}")
                scn_cfg["ang_s"] = r_ini
                yield from FXITomoFlyer.init_mot_r(scn_cfg)

                print(f"Scan # {ii} post init_mot_r finishes at {ttime.asctime()}")

                yield from FXITomoFlyer.set_cam_mode(cam, stage="post-scan")
                print(f"set post-scan finishes at {ttime.asctime()}")

                if ii < (scn_cfg["num_swing"] - 1):
                    print(
                        f" Sleeping {sleep_plan[ii]} seconds before {ii + 1}th scan ... ".center(
                            100, "#"
                        )
                    )
                    print("\n")
                    yield from bps.sleep(sleep_plan[ii])
            yield from select_filters([])
        elif (
            flyer.scn_mode == "snaked: multiple files"
        ):  # scn_mode = 1; not working due to buggy Zebra IOC
            yield from FXITomoFlyer.set_cam_mode(flyer.detectors[0], stage="pre-scan")
            yield from bps.sleep(1)

            for ii in range(scn_cfg["num_swing"]):
                print(1)
                yield from FXITomoFlyer.set_cam_step_for_scan(cam, scn_cfg)
                print(2)
                yield from FXITomoFlyer.set_mot_r_step_for_scan(scn_cfg)
                print(3)
                yield from _open_shutter_xhx(simu)
                print(4)
                for d in flyer.detectors:
                    try:
                        d.stage()
                    except:
                        d.unstage()
                        d.stage()
                for mot in mots:
                    mot.stage()

                print(5)
                print(f"{scn_cfg=}")
                st = yield from kickoff(flyer, wait=True, scn_cfg=scn_cfg)
                st.wait(timeout=10)

                print(5.5)
                det_stream = short_uid("dets")
                for d in flyer.detectors:
                    yield from bps.trigger(d, group=det_stream)
                wait(det_stream)

                print(6)
                yield from abs_set(flyer.encoder.pc.arm, 1, wait=True)

                t0 = ttime.monotonic()
                print(7)
                yield from move_and_wait(
                    zps.pi_r,
                    scn_cfg["ang_e"] + scn_cfg["rot_dir"] * scn_cfg["taxi_dist"],
                    atol=0.1,
                )

                t1 = ttime.monotonic()
                while int(flyer.encoder.pc.gated.get()):
                    if ttime.monotonic() - t1 > 60:
                        print("Scan finished abnormally. Quit!")
                        return
                    yield from bps.sleep(flyer._staging_delay)
                print(f"Scan # {ii} takes {ttime.monotonic() - t0} seconds.")
                print(8)
                st = yield from complete(flyer, wait=True)
                st.wait(timeout=10)
                print(9)
                yield from collect(flyer)

                for d in flyer.detectors:
                    try:
                        d.unstage()
                    except:
                        print(f"Cannot unstage detector {d.name}")
                        return None
                for mot in mots:
                    mot.unstage()

                print(10)
                if ii < (scn_cfg["num_swing"] - 1):
                    (scn_cfg["ang_s"], scn_cfg["ang_e"]) = (
                        scn_cfg["ang_e"],
                        scn_cfg["ang_s"],
                    )
                    scn_cfg["rot_dir"] *= -1
                    print(f"{scn_cfg=}")
                    # pc_cfg[flyer.scn_mode]["gate_start"] = scn_cfg["ang_s"]
                    # pc_cfg[flyer.scn_mode]["dir"] = flyer.pc_trig_dir[
                    #     int(scn_cfg["rot_dir"])
                    # ]
                    print(11)
                    print(f"{pc_cfg=}")
                    # yield from flyer.preset_zebra(pc_cfg)
                    # print("preset_flyer is done")

                    yield from flyer.set_pc_step_for_scan(scn_cfg, pc_cfg)
                    print(12)

            yield from FXITomoFlyer.set_cam_mode(cam, stage="ref-scan")
            yield from bps.sleep(1)
            yield from _take_ref_image(
                [cam],
                mots_pos={
                    "x": mot_x_out,
                    "y": mot_y_out,
                    "z": mot_z_out,
                    "r": mot_r_out,
                },
                chunk_size=10,
                stream_name="flat",
                simu=simu,
            )
            yield from _take_ref_image(
                [cam],
                mots_pos={
                    "x": mot_x_out,
                    "y": mot_y_out,
                    "z": mot_z_out,
                    "r": mot_r_out,
                },
                chunk_size=10,
                stream_name="dark",
                simu=simu,
            )
            print(13)
            yield from _move_sample(
                {"x": x_ini, "y": y_ini, "z": z_ini, "r": r_ini},
                repeat=2,
            )
            print(14)
            scn_cfg["ang_s"] = r_ini
            yield from FXITomoFlyer.init_mot_r(scn_cfg)
            yield from FXITomoFlyer.set_cam_mode(cam, stage="post-scan")
        elif (
            flyer.scn_mode == "snaked: single file"
        ):  # scn_mode = 2; external trigger not allowing precise angle alignment in different repeats
            yield from FXITomoFlyer.set_cam_mode(flyer.detectors[0], stage="pre-scan")
            yield from FXITomoFlyer.set_cam_step_for_scan(cam, scn_cfg)
            yield from FXITomoFlyer.set_mot_r_step_for_scan(scn_cfg)
            yield from _open_shutter_xhx(simu)
            for d in flyer.detectors:
                try:
                    d.stage()
                except:
                    d.unstage()
                    d.stage()
            print(1)
            for mot in mots:
                mot.stage()

            print(2)
            st = yield from kickoff(flyer, wait=True, scn_cfg=scn_cfg)
            st.wait(timeout=10)

            print(3)
            det_stream = short_uid("dets")
            for d in flyer.detectors:
                yield from bps.trigger(d, group=det_stream)
            wait(det_stream)

            print(4)
            yield from abs_set(flyer.encoder.pc.arm, 1, wait=True)

            t0 = ttime.monotonic()
            for ii in range(scn_cfg["num_swing"]):
                yield from move_and_wait(
                    zps.pi_r,
                    scn_cfg["ang_e"] + scn_cfg["rot_dir"] * scn_cfg["taxi_dist"],
                    atol=0.1,
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
            for mot in mots:
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
                chunk_size=10,
                stream_name="flat",
                simu=simu,
            )
            yield from _take_ref_image(
                [cam],
                mots_pos={
                    "x": mot_x_out,
                    "y": mot_y_out,
                    "z": mot_z_out,
                    "r": mot_r_out,
                },
                chunk_size=10,
                stream_name="dark",
                simu=simu,
            )
            yield from _move_sample(
                {"x": x_ini, "y": y_ini, "z": z_ini, "r": r_ini},
                repeat=2,
            )
            yield from FXITomoFlyer.set_cam_mode(cam, stage="post-scan")
        yield from select_filters([])

    yield from inner_fly_plan()
    print("scan finished")


def tomo_zfly_repeat(
    scn_mode=0,
    exp_t=0.05,
    acq_p=0.05,
    ang_s=0,
    ang_e=180,
    vel=3,
    acc_t=1,
    out_pos=[None, None, None, None],
    rel_out_flag=True,
    flts=[],
    rot_back_velo=30,
    bin_fac=None,
    roi={"min_x": 176, "size_x": 2048, "min_y": 176, "size_y": 2048},
    note="",
    md=None,
    simu=False,
    sleep=0,
    repeat=1,
    cam=None,
    flyer=None,
    open_sh=False,
):

    cam = _sel_cam(cam)
    flyer = _sel_flyer(flyer)

    for ii in range(repeat):
        yield from tomo_zfly(
            scn_mode=scn_mode,
            exp_t=exp_t,
            acq_p=acq_p,
            ang_s=ang_s,
            ang_e=ang_e,
            vel=vel,
            acc_t=acc_t,
            num_swing=1,
            out_pos=out_pos,
            rel_out_flag=rel_out_flag,
            flts=flts,
            rot_back_velo=rot_back_velo,
            bin_fac=bin_fac,
            roi=roi,
            note=note,
            md=md,
            simu=simu,
            sleep=0,
            cam=cam,
            flyer=flyer,
        )
        if ii != repeat - 1:
            print(
                f" Sleeping {sleep} seconds before {ii + 2}th scan ... ".center(
                    100, "#"
                )
            )
            print("\n")
            if open_sh:
                yield from _open_shutter_xhx(simu)
            yield from bps.sleep(sleep)


def tomo_grid_zfly(
    scn_mode=0,
    exp_t=0.05,
    acq_p=0.05,
    ang_s=0,
    ang_e=180,
    vel=3,
    acc_t=1,
    pos_dict={
        "zps_x": [None, None, None],
        "zps_y": [None, None, None],
        "zps_z": [None, None, None],
    },
    num_swing=1,
    out_pos=[None, None, None, None],
    rel_out_flag=True,
    flts=[],
    rot_back_velo=30,
    bin_fac=None,
    roi={"min_x": 176, "size_x": 2048, "min_y": 176, "size_y": 2048},
    note="",
    md=None,
    sleep=0,
    simu=False,
    cam=None,
    flyer=None,
):
    """_summary_

    Args:
        scn_mode (int, optional):
            0: "standard",  # a single scan in a given angle range
            1: "snaked: multiple files",  # back-forth rocking scan with each swing being saved into a file
            2: "snaked: single file",  # back-forth rocking scan being saved into a single file
            Defaults to 0.
        exp_t (float, optional): _description_. Defaults to 0.05.
        acq_p (float, optional): _description_. Defaults to 0.05.
        ang_s (float or None, optional): _description_. Defaults to None.
        ang_e (float, optional): _description_. Defaults to 180.
        vel (int, optional): _description_. Defaults to 3.
        acc_t (float, optional): _description_. Defaults to 1.
        pos_dict (dic, optional): a dictionary in form {
                                "zps_x": [xstart, xend, xstep],
                                "zps_y": [ystart, yend, ystep],
                                "zps_z": [zstart, zend, zstep]
                            }.
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
        cam (ophyd.Device, optional): detector; choose between Andor, MaranaU, and Oryx.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """
    cam = _sel_cam(cam)
    flyer = _sel_flyer(flyer)

    global ZONE_PLATE
    sleep_plan = _schedule_sleep(sleep, num_swing)
    if not sleep_plan:
        print(f"A wrong sleep pattern {sleep=} and {num_swing=} breaks the scan. Quit")
        return

    mots = {"zps_x": zps.sx, "zps_y": zps.sy, "zps_z": zps.sz}
    # mots = [zps.sx, zps.sz]
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
    out_x, out_y, out_z, out_r = out_pos
    (mot_x_out, mot_y_out, mot_z_out, mot_r_out) = FXITomoFlyer.def_abs_out_pos(
        out_x, out_y, out_z, out_r, rel_out_flag
    )
    print("preset done")

    grid_nodes = prep_grid_dic(pos_dict)

    for jj in grid_nodes["pos"]:
        for idx, kk in enumerate(grid_nodes["mots"]):
            yield from mv(mots[kk], jj[idx])
        yield from tomo_zfly(
            scn_mode=scn_mode,
            exp_t=exp_t,
            acq_p=acq_p,
            ang_s=ang_s,
            ang_e=ang_e,
            vel=vel,
            acc_t=acc_t,
            num_swing=1,
            out_pos=out_pos,
            rel_out_flag=rel_out_flag,
            flts=flts,
            rot_back_velo=rot_back_velo,
            bin_fac=bin_fac,
            roi=roi,
            note=note,
            md=md,
            sleep=sleep,
            simu=simu,
            cam=cam,
            flyer=flyer,
        )


def _schedule_sleep(sleep, num_scan):
    sleep_plan = {}
    if num_scan == 1:
        sleep_plan[0] = 0
        return sleep_plan
    elif num_scan > 1:
        if isinstance(sleep, list):
            if len(sleep) != num_scan - 1:
                print(
                    f"The list of sleep time has length {len(sleep)} that is inconsistent \
                        with the number of scans {num_scan}. \
                            The length of sleep time should be {num_scan - 1}"
                )
                return False
            else:
                for ii in range(0, num_scan - 1):
                    sleep_plan[ii] = sleep[ii]
                return sleep_plan
        elif isinstance(sleep, int) or isinstance(sleep, float):
            for ii in range(0, num_scan - 1):
                sleep_plan[ii] = sleep
            return sleep_plan
        else:
            print(f"Unrecognized sleep pattern {sleep}. Quit.")
            return False
    else:
        print(f"Invalid num_scan {num_scan}. Quit!")
        return False


def prep_grid_dic(pos_dict):
    """
    pos_dict: dictionary in form {
                                "zps_x": [xstart, xend, xstep],
                                "zps_y": [ystart, yend, ystep],
                                "zps_z": [zstart, zend, zstep]
                            }.
    """

    def mot_dict(mot_str):
        if mot_str == "zps_x":
            return zps.sx
        elif mot_str == "zps_y":
            return zps.sy
        elif mot_str == "zps_z":
            return zps.sz
        else:
            return None

    grid_nodes = {}
    grid_nodes["mots"] = list(pos_dict.keys())
    tem = []
    for ii in pos_dict.keys():
        if mot_dict(ii) is not None:
            num = (
                int(
                    round(
                        (
                            (
                                pos_dict[ii][1]
                                if pos_dict[ii][1] is not None
                                else mot_dict(ii).position
                            )
                            - (
                                pos_dict[ii][0]
                                if pos_dict[ii][0] is not None
                                else mot_dict(ii).position
                            )
                        )
                        / (pos_dict[ii][2] if pos_dict[ii][2] is not None else 1)
                    )
                )
                + 1
            )
            tem.append(
                np.linspace(
                    (
                        pos_dict[ii][0]
                        if pos_dict[ii][0] is not None
                        else mot_dict(ii).position
                    ),
                    (
                        pos_dict[ii][1]
                        if pos_dict[ii][1] is not None
                        else mot_dict(ii).position
                    ),
                    num,
                    endpoint=True,
                )
            )
        else:
            raise (f"Unrecognized motor name {ii}!")
    print(tem)
    m = np.meshgrid(*tem, indexing="xy")
    pos = list(zip(*(ii.ravel() for ii in m)))
    grid_nodes["pos"] = pos
    return grid_nodes
