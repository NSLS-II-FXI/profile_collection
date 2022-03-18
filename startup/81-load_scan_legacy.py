def export_tomo_scan_legacy(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    scan_type = "tomo_scan"
    scan_id = h.start["scan_id"]
    try:
        x_eng = h.start["XEng"]
    except:
        x_eng = h.start["x_ray_energy"]
    bkg_img_num = h.start["num_bkg_images"]
    dark_img_num = h.start["num_dark_images"]
    chunk_size = h.start["plan_args"]["chunk_size"]
    angle_i = h.start["plan_args"]["start"]
    angle_e = h.start["plan_args"]["stop"]
    angle_n = h.start["plan_args"]["num"]
    exposure_t = h.start["plan_args"]["exposure_time"]
    img = np.array(list(h.data("Andor_image")))
    # img = np.squeeze(img)
    img_dark = img[0:dark_img_num].reshape(-1, img.shape[-2], img.shape[-1])
    img_tomo = img[dark_img_num:-bkg_img_num]
    img_bkg = img[-bkg_img_num:].reshape(-1, img.shape[-2], img.shape[-1])

    s = img_dark.shape
    # img_dark_avg = np.mean(img_dark,axis=0).reshape(1, s[1], s[2])
    # img_bkg_avg = np.mean(img_bkg, axis=0).reshape(1, s[1], s[2])
    img_angle = np.linspace(angle_i, angle_e, angle_n)

    fname = fpath + scan_type + "_id_" + str(scan_id) + ".h5"
    with h5py.File(fname, "w") as hf:
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=img_bkg)
        hf.create_dataset("img_dark", data=img_dark)
        hf.create_dataset("img_tomo", data=img_tomo)
        hf.create_dataset("angle", data=img_angle)
    try:
        write_lakeshore_to_file(h, fname)
    except:
        print("fails to write lakeshore info into {fname}")
    del img
    del img_tomo
    del img_dark
    del img_bkg


def export_fly_scan_legacy(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    uid = h.start["uid"]
    note = h.start["note"]
    scan_type = "fly_scan"
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    x_pos = h.table("baseline")["zps_sx"][1]
    y_pos = h.table("baseline")["zps_sy"][1]
    z_pos = h.table("baseline")["zps_sz"][1]
    r_pos = h.table("baseline")["zps_pi_r"][1]
    zp_z_pos = h.table("baseline")["zp_z"][1]
    DetU_z_pos = h.table("baseline")["DetU_z"][1]
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    try:
        x_eng = h.start["XEng"]
    except:
        x_eng = h.start["x_ray_energy"]
    chunk_size = h.start["chunk_size"]
    # sanity check: make sure we remembered the right stream name
    assert "zps_pi_r_monitor" in h.stream_names
    pos = h.table("zps_pi_r_monitor")
    imgs = np.array(list(h.data("Andor_image")))
    img_dark = imgs[0]
    img_bkg = imgs[-1]
    s = img_dark.shape
    img_dark_avg = np.mean(img_dark, axis=0).reshape(1, s[1], s[2])
    img_bkg_avg = np.mean(img_bkg, axis=0).reshape(1, s[1], s[2])

    imgs = imgs[1:-1]
    s1 = imgs.shape
    imgs = imgs.reshape([s1[0] * s1[1], s1[2], s1[3]])

    with db.reg.handler_context({"AD_HDF5": AreaDetectorHDF5TimestampHandler}):
        chunked_timestamps = list(h.data("Andor_image"))

    chunked_timestamps = chunked_timestamps[1:-1]
    raw_timestamps = []
    for chunk in chunked_timestamps:
        raw_timestamps.extend(chunk.tolist())

    timestamps = convert_AD_timestamps(pd.Series(raw_timestamps))
    pos["time"] = pos["time"].dt.tz_localize("US/Eastern")

    img_day, img_hour = (
        timestamps.dt.day,
        timestamps.dt.hour,
    )
    img_min, img_sec, img_msec = (
        timestamps.dt.minute,
        timestamps.dt.second,
        timestamps.dt.microsecond,
    )
    img_time = (
        img_day * 86400 + img_hour * 3600 + img_min * 60 + img_sec + img_msec * 1e-6
    )
    img_time = np.array(img_time)

    mot_day, mot_hour = (
        pos["time"].dt.day,
        pos["time"].dt.hour,
    )
    mot_min, mot_sec, mot_msec = (
        pos["time"].dt.minute,
        pos["time"].dt.second,
        pos["time"].dt.microsecond,
    )
    mot_time = (
        mot_day * 86400 + mot_hour * 3600 + mot_min * 60 + mot_sec + mot_msec * 1e-6
    )
    mot_time = np.array(mot_time)

    mot_pos = np.array(pos["zps_pi_r"])
    offset = np.min([np.min(img_time), np.min(mot_time)])
    img_time -= offset
    mot_time -= offset
    mot_pos_interp = np.interp(img_time, mot_time, mot_pos)

    pos2 = mot_pos_interp.argmax() + 1
    # img_angle = mot_pos_interp[: pos2 - chunk_size]  # rotation angles
    img_angle = mot_pos_interp[:pos2]
    # img_tomo = imgs[: pos2 - chunk_size]  # tomo images
    img_tomo = imgs[:pos2]

    fname = fpath + scan_type + "_id_" + str(scan_id) + ".h5"

    with h5py.File(fname, "w") as hf:
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=int(scan_id))
        hf.create_dataset("scan_time", data=scan_time)
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=np.array(img_bkg, dtype=np.uint16))
        hf.create_dataset("img_dark", data=np.array(img_dark, dtype=np.uint16))
        hf.create_dataset("img_bkg_avg", data=np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset("img_dark_avg", data=np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset("img_tomo", data=np.array(img_tomo, dtype=np.uint16))
        hf.create_dataset("angle", data=img_angle)
        hf.create_dataset("x_ini", data=x_pos)
        hf.create_dataset("y_ini", data=y_pos)
        hf.create_dataset("z_ini", data=z_pos)
        hf.create_dataset("r_ini", data=r_pos)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(str(pxl_sz) + "nm"))

    try:
        write_lakeshore_to_file(h, fname)
    except:
        print("fails to write lakeshore info into {fname}")

    del img_tomo
    del img_dark
    del img_bkg
    del imgs


def export_xanes_scan_legacy(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    zp_z_pos = h.table("baseline")["zp_z"][1]
    DetU_z_pos = h.table("baseline")["DetU_z"][1]
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = h.start["plan_name"]
    #    scan_type = 'xanes_scan'
    uid = h.start["uid"]
    note = h.start["note"]
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    try:
        x_eng = h.start["XEng"]
    except:
        x_eng = h.start["x_ray_energy"]
    chunk_size = h.start["chunk_size"]
    num_eng = h.start["num_eng"]

    imgs = np.array(list(h.data("Andor_image")))
    img_dark = imgs[0]
    img_dark_avg = np.mean(img_dark, axis=0).reshape(
        [1, img_dark.shape[1], img_dark.shape[2]]
    )
    eng_list = list(h.start["eng_list"])
    s = imgs.shape
    img_xanes_avg = np.zeros([num_eng, s[2], s[3]])
    img_bkg_avg = np.zeros([num_eng, s[2], s[3]])

    if scan_type == "xanes_scan":
        for i in range(num_eng):
            img_xanes = imgs[i + 1]
            img_xanes_avg[i] = np.mean(img_xanes, axis=0)
            img_bkg = imgs[i + 1 + num_eng]
            img_bkg_avg[i] = np.mean(img_bkg, axis=0)
    elif scan_type == "xanes_scan2":
        j = 1
        for i in range(num_eng):
            img_xanes = imgs[j]
            img_xanes_avg[i] = np.mean(img_xanes, axis=0)
            img_bkg = imgs[j + 1]
            img_bkg_avg[i] = np.mean(img_bkg, axis=0)
            j = j + 2
    else:
        print("un-recognized xanes scan......")
        return 0
    img_xanes_norm = (img_xanes_avg - img_dark_avg) * 1.0 / (img_bkg_avg - img_dark_avg)
    img_xanes_norm[np.isnan(img_xanes_norm)] = 0
    img_xanes_norm[np.isinf(img_xanes_norm)] = 0
    img_bkg = np.array(img_bkg, dtype=np.float32)
    #    img_xanes_norm = np.array(img_xanes_norm, dtype=np.float32)
    fname = fpath + scan_type + "_id_" + str(scan_id) + ".h5"
    with h5py.File(fname, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("scan_time", data=scan_time)
        hf.create_dataset("X_eng", data=eng_list)
        hf.create_dataset("img_bkg", data=np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset("img_dark", data=np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset("img_xanes", data=np.array(img_xanes_norm, dtype=np.float32))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")

    try:
        write_lakeshore_to_file(h, fname)
    except:
        print("fails to write lakeshore info into {fname}")

    del img_xanes, img_dark, img_bkg, img_xanes_avg, img_dark_avg
    del img_bkg_avg, imgs, img_xanes_norm


def fly_scan_repeat_legacy(
    exposure_time=0.03,
    start_angle=None,
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
                    yield from fly_scan(
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


def export_multipos_2D_xanes_scan2_legacy(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    scan_type = h.start["plan_name"]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    #    x_eng = h.start['x_ray_energy']
    x_eng = h.start["XEng"]
    chunk_size = h.start["chunk_size"]
    chunk_size = h.start["num_bkg_images"]
    num_eng = h.start["num_eng"]
    num_pos = h.start["num_pos"]
    zp_z_pos = h.table("baseline")["zp_z"][1]
    DetU_z_pos = h.table("baseline")["DetU_z"][1]
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    try:
        repeat_num = h.start["plan_args"]["repeat_num"]
    except:
        repeat_num = 1
    imgs = list(h.data("Andor_image"))

    #    imgs = np.mean(imgs, axis=1)
    img_dark = np.array(imgs[0])
    img_dark = np.mean(img_dark, axis=0, keepdims=True)  # revised here
    eng_list = list(h.start["eng_list"])
    #    s = imgs.shape
    s = img_dark.shape  # revised here e.g,. shape=(1, 2160, 2560)

    #    img_xanes = np.zeros([num_pos, num_eng, imgs.shape[1], imgs.shape[2]])
    img_xanes = np.zeros([num_pos, num_eng, s[1], s[2]])
    img_bkg = np.zeros([num_eng, s[1], s[2]])
    index = 1
    for repeat in range(repeat_num):  # revised here
        try:
            print(f"repeat: {repeat}")
            for i in range(num_eng):
                for j in range(num_pos):
                    img_xanes[j, i] = np.mean(np.array(imgs[index]), axis=0)
                    index += 1
                img_bkg[i] = np.mean(np.array(imgs[index]), axis=0)
                index += 1

            for i in range(num_eng):
                for j in range(num_pos):
                    img_xanes[j, i] = (img_xanes[j, i] - img_dark) / (
                        img_bkg[i] - img_dark
                    )
            # save data
            # fn = os.getcwd() + "/"
            fn = fpath
            for j in range(num_pos):
                fname = (
                    f"{fn}{scan_type}_id_{scan_id}_repeat_{repeat:02d}_pos_{j:02d}.h5"
                )
                print(f"saving {fname}")
                with h5py.File(fname, "w") as hf:
                    hf.create_dataset("uid", data=uid)
                    hf.create_dataset("scan_id", data=scan_id)
                    hf.create_dataset("note", data=str(note))
                    hf.create_dataset("scan_time", data=scan_time)
                    hf.create_dataset("X_eng", data=eng_list)
                    hf.create_dataset(
                        "img_bkg", data=np.array(img_bkg, dtype=np.float32)
                    )
                    hf.create_dataset(
                        "img_dark", data=np.array(img_dark, dtype=np.float32)
                    )
                    hf.create_dataset(
                        "img_xanes", data=np.array(img_xanes[j], dtype=np.float32)
                    )
                    hf.create_dataset("Magnification", data=M)
                    hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")
                try:
                    write_lakeshore_to_file(h, fname)
                except:
                    print("fails to write lakeshore info into {fname}")
        except:
            print(f"fails in export repeat# {repeat}")
    del img_xanes
    del img_bkg
    del img_dark
    del imgs
