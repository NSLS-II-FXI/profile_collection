from tiled.client import from_profile, from_uri
from tiled.queries import Key
import numpy as np
from datetime import datetime
from tqdm import trange
import gc
from skimage.transform import rescale, resize
import h5py
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import io

CATALOG_NAME = "fxi"
tiled_client = from_uri("https://tiled.nsls2.bnl.gov")[CATALOG_NAME]
tiled_client_fxi = tiled_client["raw"]



def timestamp_to_float(t):
    tf = []
    for ts in t:
        tf.append(ts.timestamp())
    return np.array(tf)


def get_fly_scan_angle(scan_id):
    h = tiled_client_fxi[scan_id]
    det_name = h.start["detectors"][0]
    timestamp_tomo = np.array(list(h["primary"]["data"][f"{det_name}_timestamps"]))[0]
    assert "zps_pi_r_monitor" in h
    monitor = h["zps_pi_r_monitor"].read()
    #timestamp_mot = timestamp_to_float(pos["time"])
    timestamp_mot = monitor.coords["time"].values.astype("float64")

    img_ini_timestamp = timestamp_tomo[0]

    # something not correct in rotary stage.
    # we do following correction on 2023_5_16
    # mot_ini_timestamp = timestamp_mot[1]  # timestamp_mot[1] is the time when taking dark image

    n = len(timestamp_mot)
    for idx in range(1, n):
        ts1 = timestamp_mot[idx] - timestamp_mot[idx-1]
        ts2 = timestamp_mot[idx+1] - timestamp_mot[idx]
        if ts1 < 1 and ts2 < 1:
            break
    mot_ini_timestamp = timestamp_mot[idx]
    ## end modifing


    tomo_time = timestamp_tomo - img_ini_timestamp
    mot_time = timestamp_mot - mot_ini_timestamp

    mot_pos = np.array(monitor["zps_pi_r"])
    mot_pos_interp = np.interp(tomo_time, mot_time, mot_pos)

    img_angle = mot_pos_interp
    return img_angle


def write_lakeshore_to_file(h, fname):
    scan_id = h.start["scan_id"]
    for tmp in h.start.keys():
        if "T_" in tmp:
            lakeshore_info = get_lakeshore_param(scan_id, print_flag=0, return_flag=1)
            with h5py.File(fname, "a") as hf:
                for key, value in lakeshore_info.items():
                    hf.create_dataset(key, data=value)
            break


def export_scan(scan_id, scan_id_end=None, binning=4, date_end_by=None, fpath=None, reverse=False, bkg_scan_id=None):
    """
    e.g. load_scan([0001, 0002])
    """
    if scan_id_end is None:
        if isinstance(scan_id, int):
            scan_id = [scan_id]
        for item in scan_id:
            try:
                custom_export(int(item), binning, date_end_by=date_end_by, fpath=fpath, reverse=reverse, bkg_scan_id=bkg_scan_id)
                #dbv0.reg.clear_process_cache()
            except Exception as err:
                print(f'fail to export {item}')
                print(err)
    else:
        for i in range(scan_id, scan_id_end + 1):
            try:
                # export_single_scan(int(i), binning)
                custom_export(int(i), binning, date_end_by=date_end_by, fpath=fpath, reverse=reverse, bkg_scan_id=bkg_scan_id)
                #dbv0.reg.clear_process_cache()
            except Exception as err:
                print(f'fail to export {i}')
                print(err)

def custom_export(scan_id, binning=4, date_end_by=None, fpath=None, reverse=False, bkg_scan_id=None):
    """
    date_end_by: string, e.g., '2020-01-20'
    """
    tmp = list(tiled_client_fxi.search(Key("scan_id")==scan_id))

    if date_end_by is None:
        export_single_scan(scan_id, binning, reverse=reverse, bkg_scan_id=bkg_scan_id)
    else:
        for uid in tmp:
            h = tiled_client_fxi[uid]
            timestamp = h.start["time"]
            ts = pd.to_datetime(timestamp, unit="s").tz_localize("US/Eastern")
            date_end = pd.Timestamp(date_end_by,).tz_localize('US/Eastern')
            if ts < date_end:
                export_single_scan(uid, binning, reverse=reverse)
                break


def export_single_scan(scan_id=-1, binning=4, fpath=None, reverse=False, bkg_scan_id=None):
    import datetime
    #h = dbv0[scan_id]
    h = tiled_client_fxi[scan_id]
    scan_id = h.start["scan_id"]
    scan_type = h.start["plan_name"]
    #    x_eng = h.start['XEng']
    t_new = datetime.datetime(2021, 5, 1)
    t = h.start["time"] - 3600 * 60 * 4  # there are 4hour offset
    t = datetime.datetime.utcfromtimestamp(t)
    if t < t_new:
        scan = "old"
    else:
        scan = "new"

    if scan_type == "tomo_scan":
        print("exporting tomo scan: #{}".format(scan_id))
        if scan == "old":
            export_tomo_scan_legacy(h, fpath)
        else:
            export_tomo_scan(h, fpath)
        print("tomo scan: #{} loading finished".format(scan_id))

    elif scan_type == "fly_scan":
        print("exporting fly scan: #{}".format(scan_id))
        if scan == "old":
            export_fly_scan_legacy(h, fpath)
        else:
            export_fly_scan(h, fpath)
        export_fly_scan(h, fpath)
        print("fly scan: #{} loading finished".format(scan_id))

    elif scan_type == "fly_scan2":
        print("exporting fly scan2: #{}".format(scan_id))
        export_fly_scan2(h, fpath)
        print("fly scan2: #{} loading finished".format(scan_id))
    elif scan_type == "xanes_scan" or scan_type == "xanes_scan2":
        print("exporting xanes scan: #{}".format(scan_id))
        if scan == "old":
            export_xanes_scan_legacy(h, fpath)
        else:
            export_xanes_scan(h, fpath)
        print("xanes scan: #{} loading finished".format(scan_id))

    elif scan_type == 'radiography_scan':
        print("exporting radiography_scan: #{}".format(scan_id))
        export_radiography_scan(h, fpath)
    elif scan_type == "xanes_scan_img_only":
        print("exporting xanes scan image only: #{}".format(scan_id))
        export_xanes_scan_img_only(h, fpath)
        print("xanes scan img only: #{} loading finished".format(scan_id))
    elif scan_type == "z_scan":
        print("exporting z_scan: #{}".format(scan_id))
        export_z_scan(h, fpath)
    elif scan_type == "z_scan2":
        print("exporting z_scan2: #{}".format(scan_id))
        export_z_scan2(h, fpath)
    elif scan_type == "z_scan3":
        print("exporting z_scan3: #{}".format(scan_id))
        export_z_scan2(h, fpath)
    elif scan_type == "test_scan":
        print("exporting test_scan: #{}".format(scan_id))
        export_test_scan(h, fpath)
    elif scan_type == "test_scan2":
        print("exporting test_scan2: #{}".format(scan_id))
        export_test_scan2(h, fpath)
    elif scan_type == "multipos_count":
        print(f"exporting multipos_count: #{scan_id}")
        export_multipos_count(h, fpath)
    elif scan_type == "grid2D_rel":
        print("exporting grid2D_rel: #{}".format(scan_id))
        export_grid2D_rel(h, fpath)
    elif scan_type == "raster_2D":
        print("exporting raster_2D: #{}".format(scan_id))
        export_raster_2D(h, binning, reverse=reverse, bkg_scan_id=bkg_scan_id)
    elif scan_type == "raster_2D_2":
        print("exporting raster_2D_2: #{}".format(scan_id))
        export_raster_2D(h, binning, fpath, reverse=reverse)
    elif scan_type == "count" or scan_type == "delay_count":
        print("exporting count: #{}".format(scan_id))
        export_count_img(h, fpath)
    elif scan_type == "multipos_2D_xanes_scan2":
        print("exporting multipos_2D_xanes_scan2: #{}".format(scan_id))
        export_multipos_2D_xanes_scan2(h, fpath)
    elif scan_type == "multipos_2D_xanes_scan3":
        print("exporting multipos_2D_xanes_scan3: #{}".format(scan_id))
        export_multipos_2D_xanes_scan3(h, fpath)
    elif scan_type == "delay_scan":
        print("exporting delay_scan #{}".format(scan_id))
        export_delay_scan(h, fpath)
    elif scan_type in ("user_fly_only", "dmea_fly_only"):
        print("exporting user_fly_only #{}".format(scan_id))
        export_user_fly_only(h, fpath)
    elif scan_type == "moving_x_scan":
        print("exporting moving_x_scan #{}".format(scan_id))
        export_moving_x_scan(h, fpath)
    else:
        print("Un-recognized scan type ......")


def export_tomo_scan(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    scan_type = "tomo_scan"
    scan_id = h.start["scan_id"]
    try:
        x_eng = h.start["XEng"]
    except:
        x_eng = h.start["x_ray_energy"]

    angle_i = h.start["plan_args"]["start"]
    angle_e = h.start["plan_args"]["stop"]
    angle_n = h.start["plan_args"]["num"]
    #exposure_t = h.start["plan_args"]["exposure_time"]
    img = np.array(list(h["primary"]["data"][f"{det_name}_image"]))
    img_tomo = np.median(img, axis=1)
    img_dark = np.array(list(h["dark"]["data"][f"{det_name}_image"]))[0]
    img_bkg = np.array(list(h["flat"]["data"][f"{det_name}_image"]))[0]

    img_dark_avg = np.median(img_dark, axis=0, keepdims=True)
    img_bkg_avg = np.median(img_bkg, axis=0, keepdims=True)
    img_angle = np.linspace(angle_i, angle_e, angle_n)

    fname = fpath + scan_type + "_id_" + str(scan_id) + ".h5"
    with h5py.File(fname, "w") as hf:
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=img_bkg)
        hf.create_dataset("img_dark", data=img_dark)
        hf.create_dataset("img_bkg_avg", data=img_bkg_avg.astype(np.float32))
        hf.create_dataset("img_dark_avg", data=img_dark_avg.astype(np.float32))
        hf.create_dataset("img_tomo", data=img_tomo)
        hf.create_dataset("angle", data=img_angle)

    del img
    del img_tomo
    del img_dark
    del img_bkg


def export_fly_scan(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_type = "fly_scan"
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    x_pos = h["baseline"]["data"]["zps_sx"][1].item()
    y_pos = h["baseline"]["data"]["zps_sy"][1].item()
    z_pos = h["baseline"]["data"]["zps_sz"][1].item()
    r_pos = h["baseline"]["data"]["zps_pi_r"][1].item()
    relative_rot_angle = h.start['plan_args']['relative_rot_angle']
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    x_eng = h.start["XEng"]
    img_angle = get_fly_scan_angle(uid)
    img_angle = np.array(img_angle)
    id_stop = np.abs(img_angle-(img_angle[0]+relative_rot_angle-1)).argmin()

    img_tomo = np.array(list(h["primary"]["data"][f"{det_name}_image"]))[0][:len(img_angle)]

    s = img_tomo.shape
    try:
        img_dark = np.array(list(h["dark"]["data"][f"{det_name}_image"]))[0]
    except:
        img_dark = np.zeros((1, s[1], s[2]))
    try:
        img_bkg = np.array(list(h["flat"]["data"][f"{det_name}_image"]))[0]
    except:
        img_bkg = np.ones((1, s[1], s[2]))

    img_dark_avg = np.median(img_dark, axis=0, keepdims=True)
    img_bkg_avg = np.median(img_bkg, axis=0, keepdims=True)

    img_tomo = img_tomo[:id_stop]
    img_angle = img_angle[:id_stop]

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
        hf.create_dataset("Pixel Size", data=pxl_sz)

    del img_tomo
    del img_dark
    del img_bkg


def export_fly_scan2(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_type = "fly_scan2"
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    x_pos = h["baseline"]["data"]["zps_sx"][1].item()
    y_pos = h["baseline"]["data"]["zps_sy"][1].item()
    z_pos = h["baseline"]["data"]["zps_sz"][1].item()
    r_pos = h["baseline"]["data"]["zps_pi_r"][1].item()
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M


    x_eng = h.start["XEng"]
    img_angle = get_fly_scan_angle(uid)

    img_tomo = np.array(list(h["primary"]["data"][f"{det_name}_image"]))[0][:len(img_angle)]

    s = img_tomo.shape
    try:
        img_dark = np.array(list(h["dark"]["data"][f"{det_name}_image"]))[0]
    except:
        img_dark = np.zeros((1, s[1], s[2]))
    try:
        img_bkg = np.array(list(h["flat"]["data"][f"{det_name}_image"]))[0]
    except:
        img_bkg = np.ones((1, s[1], s[2]))

    img_dark_avg = np.median(img_dark, axis=0, keepdims=True)
    img_bkg_avg = np.median(img_bkg, axis=0, keepdims=True)

    fname = fpath + "fly_scan_id_" + str(scan_id) + ".h5"

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

    del img_tomo
    del img_dark
    del img_bkg
    # del imgs


def export_xanes_scan(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = h.start["plan_name"]
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

    img_xanes = np.array(list(h["primary"]["data"][f"{det_name}_image"]))
    img_xanes_avg = np.mean(img_xanes, axis=1)
    img_dark = np.array(list(h["dark"]["data"][f"{det_name}_image"]))
    img_dark_avg = np.mean(img_dark, axis=1)
    img_bkg = np.array(list(h["flat"]["data"][f"{det_name}_image"]))
    img_bkg_avg = np.mean(img_bkg, axis=1)

    eng_list = list(h.start["eng_list"])

    img_xanes_norm = (img_xanes_avg - img_dark_avg) * 1.0 / (img_bkg_avg - img_dark_avg)
    img_xanes_norm[np.isnan(img_xanes_norm)] = 0
    img_xanes_norm[np.isinf(img_xanes_norm)] = 0
    n_img = len(img_xanes_norm)
    eng_list = eng_list[:n_img]
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

    del (
        img_dark,
        img_dark_avg,
        img_bkg,
        img_bkg_avg,
        img_xanes,
        img_xanes_avg,
        img_xanes_norm,
    )


def export_xanes_scan_with_binning(h, fpath=None, binning=1):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = h.start["plan_name"]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]

    num_eng = h.start["num_eng"]

    img_xanes_avg = []
    img_bkg_avg = []
    img_list = np.array(list(h["primary"]["data"][f"{det_name}_image"]))
    bkg_list = np.array(list(h["flat"]["data"][f"{det_name}_image"]))
    for i in trange(num_eng):
        img_xanes_sub = np.array(img_list[i])
        img_xanes_sub_avg = np.median(img_xanes_sub, axis=0)
        img_bin1 = rescale(img_xanes_sub_avg, 1/binning)
        img_xanes_avg.append(img_bin1)

        img_bkg_sub = np.array(bkg_list[i])
        img_bkg_sub_avg = np.median(img_bkg_sub, axis=0)
        img_bin2 = rescale(img_bkg_sub_avg, 1/binning)
        img_bkg_avg.append(img_bin2)

    img_dark = np.array(list(h.data(f"{det_name}_image", stream_name="dark")))
    img_dark_avg = np.mean(img_dark, axis=1)[0]
    img_bin = rescale(img_dark_avg, 1/binning)
    img_dark_avg = np.expand_dims(img_bin, axis=0)
    eng_list = list(h.start["eng_list"])

    img_xanes_norm = (img_xanes_avg - img_dark_avg) * 1.0 / (img_bkg_avg - img_dark_avg)
    img_xanes_norm[np.isnan(img_xanes_norm)] = 0
    img_xanes_norm[np.isinf(img_xanes_norm)] = 0
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


    del (
        img_dark_avg,
        img_bkg_avg,
        img_xanes_avg,
        img_xanes_norm,
    )

def export_xanes_scan_img_only(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = h.start["plan_name"]
    #    scan_type = 'xanes_scan'
    uid = h.start["uid"]
    note = h.start["note"]
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]


    img_xanes = np.array(list(h["primary"]["data"]["Andor_image"]))
    img_xanes_avg = np.mean(img_xanes, axis=1)
    img_dark = np.array(list(h["dark"]["data"]["Andor_image"]))
    img_dark_avg = np.mean(img_dark, axis=1)
    img_bkg_avg = np.ones(img_dark_avg.shape)

    eng_list = list(h.start["eng_list"])

    img_xanes_norm = (img_xanes_avg - img_dark_avg) * 1.0
    img_xanes_norm[np.isnan(img_xanes_norm)] = 0
    img_xanes_norm[np.isinf(img_xanes_norm)] = 0
    fname = fpath + scan_type + "_id_" + str(scan_id) + "_img_only.h5"
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

    del (
        img_dark,
        img_dark_avg,
        img_bkg,
        img_bkg_avg,
        img_xanes,
        img_xanes_avg,
        img_xanes_norm,
    )


def export_z_scan(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = h.start["plan_name"]
    scan_id = h.start["scan_id"]
    uid = h.start["uid"]
    try:
        x_eng = h.start["XEng"]
    except:
        x_eng = h.start["x_ray_energy"]
    note = h.start["plan_args"]["note"] if h.start["plan_args"]["note"] else "None"

    img_zscan = np.mean(np.array(list(h["primary"]["data"][f"{det_name}_image"])), axis=1)
    img_bkg = np.mean(np.array(list(h["flat"]["data"][f"{det_name}_image"])), axis=1).squeeze()
    img_dark = np.mean(np.array(list(h["dark"]["data"][f"{det_name}_image"])), axis=1).squeeze()

    img_norm = (img_zscan - img_dark) / (img_bkg - img_dark)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0
    #    fn = h.start['plan_args']['fn']
    fname = fpath + scan_type + "_id_" + str(scan_id) + ".h5"
    with h5py.File(fname, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=img_bkg.astype(np.float32))
        hf.create_dataset("img_dark", data=img_dark.astype(np.float32))
        hf.create_dataset("img", data=img_zscan.astype(np.float32))
        hf.create_dataset("img_norm", data=img_norm.astype(np.float32))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")


    del img_zscan, img_bkg, img_dark, img_norm


def export_z_scan2(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = h.start["plan_name"]
    scan_id = h.start["scan_id"]
    uid = h.start["uid"]
    try:
        x_eng = h.start["XEng"]
    except:
        x_eng = h.start["x_ray_energy"]
    note = h.start["plan_args"]["note"] if h.start["plan_args"]["note"] else "None"
    img = np.mean(np.array(list(h["primary"]["data"][f"{det_name}_image"])), axis=1)
    img = np.squeeze(img)
    img_dark = img[0]
    l1 = np.arange(1, len(img), 2)
    l2 = np.arange(2, len(img), 2)

    img_zscan = img[l1]
    img_bkg = img[l2]

    img_norm = (img_zscan - img_dark) / (img_bkg - img_dark)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0

    fname = fpath + scan_type + "_id_" + str(scan_id) + ".h5"
    with h5py.File(fname, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg",data=img_bkg.astype(np.float32))
        hf.create_dataset("img_dark", data=img_dark.astype(np.float32))
        hf.create_dataset("img", data=img_zscan.astype(np.float32))
        hf.create_dataset("img_norm", data=img_norm.astype(np.float32))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")

    del img, img_zscan, img_bkg, img_dark, img_norm


def export_test_scan(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = h.start["plan_name"]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    try:
        x_eng = h.start["XEng"]
    except:
        x_eng = h.start["x_ray_energy"]
    num = h.start["plan_args"]["num_img"]


    img_list = list(h["primary"]["data"][f"{det_name}_image"])
    n = len(img_list)
    for i in range(n-1, 0, -1):
        try:
            img = np.array(img_list[:i])[:, 0]
        except:
            continue
        break
    if i < n:
        print(f'few images are lost, only {i}/{n} images saved')
    try:
        img_dark = np.array(list(h["dark"]["data"][f"{det_name}_image"]))[0]
        img_dark_avg = np.median(img_dark, axis=0, keepdims=True)
    except:
        img_dark = np.zeros((1, img.shape[1], img.shape[2]))
        img_dark_avg = img_dark
        print('img dark not taken')
    try:
        img_bkg = np.array(list(h["flat"]["data"][f"{det_name}_image"]))[0]
        img_bkg_avg = np.median(img_bkg, axis=0, keepdims=True)
    except:
        img_bkg = np.zeros((1, img.shape[1], img.shape[2]))
        img_bkg_avg = img_bkg
        print('img background not taken')

    img_norm = (img - img_dark_avg) * 1.0 / (img_bkg_avg - img_dark_avg)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0
    fname = fpath + scan_type + "_id_" + str(scan_id) + ".h5"
    with h5py.File(fname, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("scan_time", data=scan_time)
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset("img_dark", data=np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset("img", data=np.array(img, dtype=np.float32))
        hf.create_dataset("img_norm", data=np.array(img_norm, dtype=np.float32))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")

    del (
        img_dark,
        img_dark_avg,
        img_bkg,
        img_bkg_avg,
        #img_xanes,
        #img_xanes_avg,
        img_norm,
        img
    )


def export_test_scan2(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = h.start["plan_name"]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    try:
        x_eng = h.start["XEng"]
    except:
        x_eng = h.start["x_ray_energy"]
    num = h.start["plan_args"]["num_img"]

    img = np.array(list(h["primary"]["data"][f"{det_name}_image"]))[0]
    try:
        img_dark = np.array(list(h["dark"]["data"][f"{det_name}_image"]))[0]
        img_dark_avg = np.mean(img_dark, axis=0, keepdims=True)
    except:
        img_dark = np.zeros((1, img.shape[1], img.shape[2]))
        img_dark_avg = img_dark
    try:
        img_bkg = np.array(list(h["flat"]["data"][f"{det_name}_image"]))[0]
        img_bkg_avg = np.mean(img_bkg, axis=0, keepdims=True)
    except:
        img_bkg = np.ones((1, img.shape[1], img.shape[2]))
        img_bkg_avg = img_bkg

    img_norm = (img - img_dark_avg) * 1.0 / (img_bkg_avg - img_dark_avg)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0
    fname = fpath + scan_type + "_id_" + str(scan_id) + ".h5"
    with h5py.File(fname, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("scan_time", data=scan_time)
        hf.create_dataset("X_eng", data=x_eng)
        hf.create_dataset("img_bkg", data=np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset("img_dark", data=np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset("img", data=np.array(img, dtype=np.float32))
        hf.create_dataset("img_norm", data=np.array(img_norm, dtype=np.float32))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")

    del (
        img_dark,
        img_dark_avg,
        img_bkg,
        img_bkg_avg,
        #img_xanes,
        #img_xanes_avg,
        img_norm,
        img
    )



def export_radiography_scan(h, fpath=None):
    export_test_scan2(h, fpath)



def export_count_img(h, fpath=None):
    """
    load images (e.g. RE(count([KinetixU], 10)) ) and save to .h5 file
    """
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    try:
        zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
        DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
        M = (DetU_z_pos / zp_z_pos - 1) * 10.0
        pxl_sz = 6500.0 / M
    except:
        M = 0
        pxl_sz = 0
        print("fails to calculate magnification and pxl size")

    uid = h.start["uid"]
    #det = h.start["detectors"][0]
    #img = get_img(h, det)
    det_name = h.start["detectors"][0]
    img = np.array(list(h["primary"]["data"][f"{det_name}_image"]))[0]
    scan_id = h.start["scan_id"]
    fn = fpath + "count_id_" + str(scan_id) + ".h5"
    with h5py.File(fn, "w") as hf:
        hf.create_dataset("img", data=img.astype(np.float32))
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")
    try:
        write_lakeshore_to_file(h, fn)
    except:
        print("fails to write lakeshore info into {fname}")


def export_delay_scan(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det = h.start["detectors"][0]
    scan_type = h.start["plan_name"]
    scan_id = h.start["scan_id"]
    uid = h.start["uid"]
    x_eng = h.start["XEng"]
    note = h.start["plan_args"]["note"] if h.start["plan_args"]["note"] else "None"
    mot_name = h.start["plan_args"]["motor"]
    mot_start = h.start["plan_args"]["start"]
    mot_stop = h.start["plan_args"]["stop"]
    mot_steps = h.start["plan_args"]["steps"]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    if det == "detA1" or det == "KinetixU":
        img = np.array(list(h["primary"]["data"][f"{det}_image"]))[0]
        fname = fpath + scan_type + "_id_" + str(scan_id) + ".h5"
        with h5py.File(fname, "w") as hf:
            hf.create_dataset("img", data=np.array(img, dtype=np.float32))
            hf.create_dataset("uid", data=uid)
            hf.create_dataset("scan_id", data=scan_id)
            hf.create_dataset("X_eng", data=x_eng)
            hf.create_dataset("note", data=str(note))
            hf.create_dataset("start", data=mot_start)
            hf.create_dataset("stop", data=mot_stop)
            hf.create_dataset("steps", data=mot_steps)
            hf.create_dataset("motor", data=mot_name)
            hf.create_dataset("Magnification", data=M)
            hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")
        try:
            write_lakeshore_to_file(h, fname)
        except:
            print("fails to write lakeshore info into {fname}")
    else:
        print("no image stored in this scan")


def export_multipos_count(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    scan_type = h.start["plan_name"]
    scan_id = h.start["scan_id"]
    uid = h.start["uid"]
    num_dark = h.start["num_dark_images"]
    num_of_position = h.start["num_of_position"]
    note = h.start["note"]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    #img_raw = list(h.data(f"{det_name}_image"))
    img_raw = list(h["primary"]["data"][f"{det_name}_image"])
    img_dark = np.squeeze(np.array(img_raw[:num_dark]))
    img_dark_avg = np.mean(img_dark, axis=0, keepdims=True)
    num_repeat = np.int(
        (len(img_raw) - 10) / num_of_position / 2
    )  # alternatively image and background

    tot_img_num = num_of_position * 2 * num_repeat
    s = img_dark.shape
    img_group = np.zeros([num_of_position, num_repeat, s[1], s[2]], dtype=np.float32)

    for j in range(num_repeat):
        index = num_dark + j * num_of_position * 2
        print(f"processing #{index} / {tot_img_num}")
        for i in range(num_of_position):
            tmp_img = np.array(img_raw[index + i * 2])
            tmp_bkg = np.array(img_raw[index + i * 2 + 1])
            img_group[i, j] = (tmp_img - img_dark_avg) / (tmp_bkg - img_dark_avg)
    # fn = os.getcwd() + "/"
    fname = fpath + scan_type + "_id_" + str(scan_id) + ".h5"
    with h5py.File(fname, "w") as hf:
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("note", data=str(note))
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")
        for i in range(num_of_position):
            hf.create_dataset(f"img_pos{i+1}", data=np.squeeze(img_group[i]))


def export_grid2D_rel(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    uid = h.start["uid"]
    note = h.start["note"]
    scan_type = "grid2D_rel"
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    x_eng = h.start["XEng"]
    num1 = h.start["plan_args"]["num1"]
    num2 = h.start["plan_args"]["num2"]
    img = np.squeeze(np.array(list(h["primary"]["data"][f"{det_name}_image"])))

    fname = scan_type + "_id_" + str(scan_id)
    # cwd = os.getcwd()
    cwd = fpath
    try:
        os.mkdir(cwd + f"{fname}")
    except:
        print(cwd + f"{fname} existed")
    fout = cwd + f"{fname}"
    for i in range(num1):
        for j in range(num2):
            fname_tif = fout + f"_({ij}).tif"
            img = Image.fromarray(img[i * num1 + j])
            img.save(fname_tif)


def export_raster_2D_2(h, binning=4, fpath=None):
    import tifffile
    from skimage import io

    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_type = "grid2D_rel"
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    num_dark = h.start["plan_args"]["num_dark_images"]
    num_bkg = h.start["plan_args"]["num_bkg_images"]
    x_eng = h.start["XEng"]
    x_range = h.start["plan_args"]["x_range"]
    y_range = h.start["plan_args"]["y_range"]
    img_sizeX = h.start["plan_args"]["img_sizeX"]
    img_sizeY = h.start["plan_args"]["img_sizeY"]
    pix = h.start["plan_args"]["pxl"]
    chunk_size = h.start["plan_args"]["chunk_size"]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    rot_angle = h["baseline"]["data"]["zps_pi_r"][1].item()
    scan_x_flag = h.start['plan_args']['scan_x_flag']
    if scan_x_flag:
        pix = pix * np.cos(rot_angle/180.*np.pi)
    else:
        pix = pix * np.sin(rot_angle/180.*np.pi)
    pix = np.abs(pix)

    img_raw = np.array(list(h.data(f"{det_name}_image", stream_name="primary"))) # (9, chunk_size, 1020, 2014)
    img = np.mean(img_raw, axis=1) # (9, 1020, 1024)
    s = img.shape
    try:
        img_dark = np.array(list(h.data(f"{det_name}_image", stream_name="dark")))[0]
        img_dark_avg = np.mean(img_dark, axis=0, keepdims=True) #(1, 1020, 1024)
    except:
        img_dark_avg = np.zeros((1, *s[1:]))

    try:
        img_bkg = np.array(list(h.data(f"{det_name}_image", stream_name="flat")))[0]
        img_bkg_avg = np.mean(img_bkg, axis=0, keepdims=True) #(1, 1020, 1024)
    except:
        img_bkg_avg = np.ones((1, *s[1:]))

    img = (img - img_dark_avg) / (img_bkg_avg - img_dark_avg)
    s = img.shape

    x_num = round((x_range[1] - x_range[0]) + 1)
    y_num = round((y_range[1] - y_range[0]) + 1)
    # start stitching
    frac = np.round(pix / pxl_sz, 2) # e.g., 10nm/20nm = 0.5
    rl = int(s[1] * frac) # num of pixel (row) in cropped_and_centered image
    rs = s[1]/2 * (1 - frac)
    rs = int(max(0, rs))
    re = rs + rl
    re = int(min(re, s[1]))

    cl = int(s[2] * frac) # num of pixel (column) in cropped_and_centered image
    cs = s[2]/2 *(1 - frac)
    cs = int(max(0, cs))
    ce = cs + cl
    ce = int(min(ce, s[2]))

    x_list = np.linspace(x_range[0], x_range[1], x_num)
    y_list = np.linspace(y_range[0], y_range[1], y_num)
    row_size = y_num * rl
    col_size = x_num * cl
    img_patch = np.zeros([1, row_size, col_size])


    pos_file_for_print = np.zeros([x_num * y_num, 4])
    pos_file = ["cord_x\tcord_y\tx_pos_relative\ty_pos_relative\n"]
    index = 0
    for i in range(int(x_num)):
        for j in range(int(y_num)):
            #img_patch[0, j * s[1] : (j + 1) * s[1], i * s[2] : (i + 1) * s[2]] = img[index, rs:re, cs:ce]
            img_patch[0, j*rl : (j+1)*rl, i*cl : (i+1)*cl] = img[index, rs:re, cs:ce]
            pos_file_for_print[index] = [
                x_list[i],
                y_list[j],
                x_list[i] * pix * img_sizeX / 1000,
                y_list[j] * pix * img_sizeY / 1000,
            ]
            pos_file.append(
                f"{x_list[i]:3.0f}\t{y_list[j]:3.0f}\t{x_list[i]*pix*img_sizeX/1000:3.3f}\t\t{y_list[j]*pix*img_sizeY/1000:3.3f}\n"
            )
            index = index + 1
            print(i,j, index)
    s_patch = img_patch.shape # (1, 3060, 3072)
    try:
        s_bin = (s_patch[0], s_patch[1]//binning*binning, s_patch[2]//binning*binning)
        img_patch_bin = bin_ndarray(
            img_patch[:, :int(s_bin[1]), :int(s_bin[2])], new_shape=(s_bin[0], int(s_bin[1]//binning), int(s_bin[2]//binning))
        )
    except:
        img_patch_bin = img_patch
        binning = 1

    fout_h5 = fpath + f"raster2D_scan_{scan_id}_binning_{binning}.h5"
    fout_tiff = fpath + f"raster2D_scan_{scan_id}_binning_{binning}.tiff"
    fout_txt = fpath + f"raster2D_scan_{scan_id}_cord.txt"
    print(f"{pos_file_for_print}")
    io.imsave(fout_tiff, np.array(img_patch_bin[0], dtype=np.float32))
    with open(f"{fout_txt}", "w+") as f:
        f.writelines(pos_file)
    # tifffile.imsave(fout_tiff, np.array(img_patch_bin, dtype=np.float32))
    num_img = int(x_num) * int(y_num)
    # cwd = os.getcwd()
    # new_dir = f"{cwd}/raster_scan_{scan_id}"
    new_dir = fpath + f"raster_scan_{scan_id}"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    """
    s = img.shape
    tmp = bin_ndarray(img, new_shape=(s[0], int(s[1]/binning), int(s[2]/binning)))
    for i in range(num_img):
        fout = f'{new_dir}/img_{i:02d}_binning_{binning}.tiff'
        print(f'saving {fout}')
        tifffile.imsave(fout, np.array(tmp[i], dtype=np.float32))
    """
    fn_h5_save = f"{new_dir}/img_{i:02d}_binning_{binning}.h5"
    with h5py.File(fn_h5_save, "w") as hf:
        hf.create_dataset("img_patch", data=np.array(img_patch_bin, np.float32))
        hf.create_dataset("img", data=np.array(img, np.float32))
        hf.create_dataset("img_dark", data=np.array(img_dark_avg, np.float32))
        hf.create_dataset("img_bkg", data=np.array(img_bkg_avg, np.float32))
        hf.create_dataset("XEng", data=x_eng)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")
    try:
        write_lakeshore_to_file(h, fn_h5_save)
    except:
        print(f"fails to write lakeshore info into {fn_h5_save}")


def export_raster_2D(h, binning=4, fpath=None, reverse=False, bkg_scan_id=None):

    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"

    det_name = h.start["detectors"][0]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_id = h.start["scan_id"]
    x_eng = h.start["XEng"]
    x_range = h.start["plan_args"]["x_range"]
    y_range = h.start["plan_args"]["y_range"]
    img_sizeX = h.start["plan_args"]["img_sizeX"]
    img_sizeY = h.start["plan_args"]["img_sizeY"]
    pix = h.start["plan_args"]["pxl"]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    rot_angle = h["baseline"]["data"]["zps_pi_r"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    scan_x_flag = h.start['plan_args']['scan_x_flag']
    if scan_x_flag:
        pix = pix * np.cos(rot_angle/180.*np.pi)
    else:
        pix = pix * np.sin(rot_angle/180.*np.pi)

    if not bkg_scan_id is None:
        h_ref = db[norm_bkg_scan_id]
    else:
        h_ref = h

    img_raw = np.array(list(h.data(f"{det_name}_image", stream_name="primary"))) # (9, chunk_size, 1020, 2014)
    img = np.mean(img_raw, axis=1) # (9, 1020, 1024)
    s = img.shape # (9, 1020, 1024)
    try:
        img_dark = np.array(list(h["dark"]["data"][f"{det_name}_image"]))[0]
        img_dark_avg = np.mean(img_dark, axis=0, keepdims=True)
    except:
        img_dark_avg = np.zeros((1, *s[1:]))

    try:
        img_bkg = np.array(list(h["flat"]["data"][f"{det_name}_image"]))[0]
        img_bkg_avg = np.mean(img_bkg, axis=0, keepdims=True)
    except:
        img_bkg_avg = np.ones((1, *s[1:]))

    if reverse:
        img_raw = img_raw[:, ::-1, ::-1]
        img_dark_avg = img_dark_avg[:, ::-1, ::-1]
        img_bkg_avg = img_bkg_avg[:, ::-1, ::-1]


    img = (img - img_dark_avg) / (img_bkg_avg - img_dark_avg)
    x_num = round((x_range[1] - x_range[0]) + 1)
    y_num = round((y_range[1] - y_range[0]) + 1)

    # start stitching
    if pix > pxl_sz:
        warn_msg = f'warning: the setpoint pixel size used in scan ({pix:3.2f} nm) should be smaller than actual pixel size ({pxl_sz:3.2f} nm)'
        pix = pxl_sz
    else:
        warn_msg = ''

    frac = np.round(pix / pxl_sz, 2) # e.g., 10nm/20nm = 0.5
    rl = int(s[1] * frac) # num of pixel (row) in cropped_and_centered image
    rs = s[1]/2 * (1 - frac)
    rs = int(max(0, rs))
    re = rs + rl
    re = int(min(re, s[1]))

    cl = int(s[2] * frac) # num of pixel (column) in cropped_and_centered image
    cs = s[2]/2 *(1 - frac)
    cs = int(max(0, cs))
    ce = cs + cl
    ce = int(min(ce, s[2]))

    x_list = np.linspace(x_range[0], x_range[1], x_num)
    y_list = np.linspace(y_range[0], y_range[1], y_num)
    row_size = y_num * rl
    col_size = x_num * cl
    img_patch = np.zeros([1, row_size, col_size])


    pos_file_for_print = np.zeros([x_num * y_num, 4])
    pos_file = ["cord_x\tcord_y\tx_pos_relative\ty_pos_relative\n"]
    index = 0
    for i in range(int(x_num)):
        for j in range(int(y_num)):
            #img_patch[0, j * s[1] : (j + 1) * s[1], i * s[2] : (i + 1) * s[2]] = img[index, rs:re, cs:ce]
            img_patch[0, j*rl : (j+1)*rl, i*cl : (i+1)*cl] = img[index, rs:re, cs:ce]
            pos_file_for_print[index] = [
                x_list[i],
                y_list[j],
                x_list[i] * pix * img_sizeX / 1000,
                y_list[j] * pix * img_sizeY / 1000,
            ]
            pos_file.append(
                f"{x_list[i]:3.0f}\t{y_list[j]:3.0f}\t{x_list[i]*pix*img_sizeX/1000:3.3f}\t\t{y_list[j]*pix*img_sizeY/1000:3.3f}\n"
            )
            index = index + 1
            print(i,j, index)
    s_patch = img_patch.shape # (1, 3060, 3072)
    try:
        s_bin = (s_patch[0], s_patch[1]//binning*binning, s_patch[2]//binning*binning)
        img_patch_bin = bin_ndarray(
            img_patch[:, :int(s_bin[1]), :int(s_bin[2])], new_shape=(s_bin[0], int(s_bin[1]//binning), int(s_bin[2]//binning))
        )
    except:
        img_patch_bin = img_patch
        binning = 1

    fout_tiff = fpath + f"raster2D_scan_{scan_id}_binning_{binning}.tiff"
    fout_txt = fpath + f"raster2D_scan_{scan_id}_cord.txt"
    print(f"{pos_file_for_print}")
    with open(f"{fout_txt}", "w+") as f:
        f.writelines(pos_file)
    io.imsave(fout_tiff, np.array(img_patch_bin, dtype=np.float32))
    num_img = int(x_num) * int(y_num)

    new_dir = fpath + f"raster_scan_{scan_id}"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    print(warn_msg)

    fn_h5_save = f"{new_dir}/img_{i:02d}_binning_{binning}.h5"
    with h5py.File(fn_h5_save, "w") as hf:
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("Note", data=note)
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("img_patch", data=np.array(img_patch_bin, np.float32))
        hf.create_dataset("img", data=np.array(img, np.float32))
        hf.create_dataset("img_dark", data=np.array(img_dark_avg, np.float32))
        hf.create_dataset("img_bkg", data=np.array(img_bkg_avg, np.float32))
        hf.create_dataset("XEng", data=x_eng)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")

    del img, img_patch, img_raw, img_dark, img_bkg


def export_multipos_2D_xanes_scan2(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    scan_type = h.start["plan_name"]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]

    x_eng = h.start["XEng"]
    num_eng = h.start["num_eng"]
    num_pos = h.start["num_pos"]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    img_xanes = np.array(list(h["primary"]["data"][f"{det_name}_image"]))
    img_dark = np.array(list(h["dark"]["data"][f"{det_name}_image"]))
    img_bkg = np.array(list(h["flat"]["data"][f"{det_name}_image"]))

    img_xanes = np.median(img_xanes, axis=1)
    img_dark = np.median(img_dark, axis=1)
    img_bkg = np.median(img_bkg, axis=1)

    eng_list = list(h.start["eng_list"])

    len_img = len(img_xanes)
    len_bkg = len(img_bkg)

    idx = int(len_img // num_pos)

    id_end = int(min(idx, len_bkg) * num_pos)
    img_xanes = img_xanes[:id_end]
    eng_list = eng_list[:id_end]


    for j in range(num_pos):
        img = img_xanes[j::num_pos]
        img_n = (img - img_dark) / (img_bkg - img_dark)
        fn = fpath
        fname = (f"{fn}{scan_type}_id_{scan_id}_pos_{j:02d}.h5")

        try:
            print(f"saving {fname}")
            with h5py.File(fname, "w") as hf:
                    hf.create_dataset("uid", data=uid)
                    hf.create_dataset("scan_id", data=scan_id)
                    hf.create_dataset("note", data=str(note))
                    hf.create_dataset("scan_time", data=scan_time)
                    hf.create_dataset("X_eng", data=eng_list)
                    hf.create_dataset("img_bkg", data=np.array(img_bkg, dtype=np.float32))
                    hf.create_dataset("img_dark", data=np.array(img_dark, dtype=np.float32))
                    hf.create_dataset("img_xanes", data=np.array(img_n, dtype=np.float32))
                    hf.create_dataset("Magnification", data=M)
                    hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")
        except Exception as err:
            print(err)
    del img_xanes
    del img_bkg


def export_multipos_2D_xanes_scan3(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = h.start["plan_name"]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    num_eng = h.start["num_eng"]
    num_pos = h.start["num_pos"]
    imgs = np.array(list(h["primary"]["data"][f'{det_name}_image']))
    imgs = np.mean(imgs, axis=1)
    img_dark = imgs[0]
    eng_list = list(h.start["eng_list"])
    s = imgs.shape

    img_xanes = np.zeros([num_pos, num_eng, imgs.shape[1], imgs.shape[2]])
    img_bkg = np.zeros([num_eng, imgs.shape[1], imgs.shape[2]])

    index = 1
    for i in range(num_eng):
        for j in range(num_pos):
            img_xanes[j, i] = imgs[index]
            index += 1

    img_bkg = imgs[-num_eng:]

    for i in range(num_eng):
        for j in range(num_pos):
            img_xanes[j, i] = (img_xanes[j, i] - img_dark) / (img_bkg[i] - img_dark)
    # save data
    # fn = os.getcwd() + "/"
    fn = fpath
    for j in range(num_pos):
        fname = f"{fn}{scan_type}_id_{scan_id}_pos_{j}.h5"
        with h5py.File(fname, "w") as hf:
            hf.create_dataset("uid", data=uid)
            hf.create_dataset("scan_id", data=scan_id)
            hf.create_dataset("note", data=str(note))
            hf.create_dataset("scan_time", data=scan_time)
            hf.create_dataset("X_eng", data=eng_list)
            hf.create_dataset("img_bkg", data=np.array(img_bkg, dtype=np.float32))
            hf.create_dataset("img_dark", data=np.array(img_dark, dtype=np.float32))
            hf.create_dataset(
                "img_xanes", data=np.array(img_xanes[j], dtype=np.float32)
            )
            hf.create_dataset("Magnification", data=M)
            hf.create_dataset("Pixel Size", data=str(pxl_sz) + "nm")

    del img_xanes
    del img_bkg
    del img_dark
    del imgs


def export_user_fly_only(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_type = h.start["plan_name"]
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    dark_scan_id = h.start["plan_args"]["dark_scan_id"]
    bkg_scan_id = h.start["plan_args"]["bkg_scan_id"]
    x_pos = h["baseline"]["data"]["zps_sx"][1].item()
    y_pos = h["baseline"]["data"]["zps_sy"][1].item()
    z_pos = h["baseline"]["data"]["zps_sz"][1].item()
    r_pos = h["baseline"]["data"]["zps_pi_r"][1].item()

    try:
        x_eng = h.start["XEng"]
    except:
        x_eng = h.start["x_ray_energy"]
    # sanity check: make sure we remembered the right stream name
    assert "zps_pi_r_monitor" in h.stream_names
    pos = h["zps_pi_r_monitor"].read()
    imgs = np.array(list(h["primary"]["data"][f"{det_name}_image"]))

    s1 = imgs.shape
    chunk_size = s1[1]
    imgs = imgs.reshape(-1, s1[2], s1[3])

    # load darks and bkgs
    img_dark = np.array(list(h["dark"]["data"][f"{det_name}_image"]))[0]
    img_bkg = np.array(list(h["flat"]["data"][f"{det_name}_image"]))[0]
    s = img_dark.shape
    img_dark_avg = np.mean(img_dark, axis=0).reshape(1, s[1], s[2])
    img_bkg_avg = np.mean(img_bkg, axis=0).reshape(1, s[1], s[2])

    chunked_timestamps = list(h["primary"]["data"][f"{det_name}_timestamps"])

    raw_timestamps = []
    for chunk in chunked_timestamps:
        raw_timestamps.extend(chunk.tolist())

    timestamps = convert_AD_timestamps(pd.Series(raw_timestamps))
    pos["time"] = convert_AD_timestamps(pd.Series(pos["time"]))

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
    img_angle = mot_pos_interp[: pos2 - chunk_size]  # rotation angles
    img_tomo = imgs[: pos2 - chunk_size]  # tomo images

    fname = fpath + "fly_scan_id_" + str(scan_id) + ".h5"

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

    del img_tomo
    del img_dark
    del img_bkg
    del imgs



def export_scan_change_expo_time(h, fpath=None, save_range_x=[], save_range_y=[]):

    if fpath is None:
        fpath = os.getcwd()
    if not fpath[-1] == "/":
        fpath += "/"
    det_name = h.start["detectors"][0]
    scan_id = h.start["scan_id"]
    fpath += f"scan_{scan_id}/"
    fpath_t1 = fpath + "t1/"
    fpath_t2 = fpath + "t2/"
    os.makedirs(fpath, exist_ok=True, mode=0o777)
    os.makedirs(fpath_t1, exist_ok=True, mode=0o777)
    os.makedirs(fpath_t2, exist_ok=True, mode=0o777)

    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M
    scan_type = h.start["plan_name"]
    uid = h.start["uid"]
    note = h.start["plan_args"]["note"]

    scan_time = h.start["time"]
    x_eng = h.start["x_ray_energy"]
    t1 = h.start["plan_args"]["t1"]
    t2 = h.start["plan_args"]["t2"]

    img_sizeX = h.start["plan_args"]["img_sizeX"]
    img_sizeY = h.start["plan_args"]["img_sizeY"]
    pxl = h.start["plan_args"]["pxl"]
    step_x = img_sizeX * pxl
    step_y = img_sizeY * pxl

    x_range = h.start["plan_args"]["x_range"]
    y_range = h.start["plan_args"]["y_range"]

    imgs = list(run["primary"]["data"][f"{det_name}_image"])
    s = imgs[0].shape

    if len(save_range_x) == 0:
        save_range_x = [0, s[0]]
    if len(save_range_y) == 0:
        save_range_y = [0, s[1]]

    img_dark_t1 = np.median(np.array(imgs[:5]), axis=0)
    img_dark_t2 = np.median(np.array(imgs[5:10]), axis=0)
    imgs = imgs[10:]

    nx = np.abs(x_range[1] - x_range[0] + 1)
    ny = np.abs(y_range[1] - y_range[0] + 1)
    pos_x = np.zeros(nx * ny)
    pos_y = pos_x.copy()

    idx = 0

    for ii in range(nx):
        if not ii % 100:
            print(f"nx = {ii}")
        for jj in range(ny):
            if not jj % 10:
                print(f"ny = {jj}")
            pos_x[idx] = ii * step_x
            pos_y[idx] = jj * step_y
            idx += 1
            id_c = ii * ny * (5 + 5 + 2) + jj * (5 + 5 + 2)
            img_t1 = imgs[id_c]
            img_t2 = imgs[id_c + 1]
            img_bkg_t1 = imgs[(id_c + 2) : (id_c + 7)]
            img_bkg_t1 = np.median(img_bkg_t1, axis=0)
            img_bkg_t2 = imgs[(id_c + 7) : (id_c + 12)]
            img_bkg_t2 = np.median(img_bkg_t2, axis=0)

            img_t1_n = (img_t1 - img_dark_t1) / (img_bkg_t1 - img_dark_t1)
            img_t2_n = (img_t2 - img_dark_t2) / (img_bkg_t2 - img_dark_t2)

            fsave_t1 = fpath_t1 + f"img_t1_{idx:05d}.tiff"
            fsave_t2 = fpath_t2 + f"img_t2_{idx:05d}.tiff"

            im1 = img_t1_n[
                0, save_range_x[0] : save_range_x[1], save_range_y[0] : save_range_y[1]
            ]
            im2 = img_t2_n[
                0, save_range_x[0] : save_range_x[1], save_range_y[0] : save_range_y[1]
            ]
            io.imsave(fsave_t1, im1.astype(np.float32))
            io.imsave(fsave_t2, im2.astype(np.float32))
    with h5py.File(fpath, "w") as hf:
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("scan_type", data=scan_type)
        hf.create_dataset("uid", data=uid)
        hf.create_dataset("pxl_sz", data=pxl_sz)
        hf.create_dataset("note", data=note)
        hf.create_dataset("XEng", data=x_eng)
        hf.create_dataset("pos_x", data=pos_x)
        hf.create_dataset("pos_y", data=pos_y)


def export_moving_x_scan(h, fpath=None):
    if fpath is None:
        fpath = "./"
    else:
        if not fpath[-1] == "/":
            fpath += "/"
    det_name = h.start["detectors"][0]
    uid = h.start["uid"]
    note = h.start["note"]
    scan_type = "moving_x_scan"
    scan_id = h.start["scan_id"]
    scan_time = h.start["time"]
    x_pos = h["baseline"]["data"]["zps_sx"][1].item()
    y_pos = h["baseline"]["data"]["zps_sy"][1].item()
    z_pos = h["baseline"]["data"]["zps_sz"][1].item()
    r_pos = h["baseline"]["data"]["zps_pi_r"][1].item()
    zp_z_pos = h["baseline"]["data"]["zp_z"][1].item()
    DetU_z_pos = h["baseline"]["data"]["DetU_z"][1].item()
    M = (DetU_z_pos / zp_z_pos - 1) * 10.0
    pxl_sz = 6500.0 / M

    x_eng = h.start["XEng"]
    img = np.array(list(h["primary"]["data"][f"{det_name}_image"]))[0]
    s = img.shape
    try:
        img_dark = np.array(list(h["dark"]["data"][f"{det_name}_image"]))[0]
    except:
        img_dark = np.zeros((1, s[1], s[2]))
    try:
        img_bkg = np.array(list(h["flat"]["data"][f"{det_name}_image"]))[0]
    except:
        img_bkg = np.ones((1, s[1], s[2]))

    img_dark_avg = np.median(img_dark, axis=0, keepdims=True)
    img_bkg_avg = np.median(img_bkg, axis=0, keepdims=True)

    stage_x_pos = get_moving_x_scan_position(scan_id)
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
        hf.create_dataset("img", data=np.array(img, dtype=np.uint16))
        hf.create_dataset("x_pos", data=stage_x_pos)
        hf.create_dataset("x_ini", data=x_pos)
        hf.create_dataset("y_ini", data=y_pos)
        hf.create_dataset("z_ini", data=z_pos)
        hf.create_dataset("r_ini", data=r_pos)
        hf.create_dataset("Magnification", data=M)
        hf.create_dataset("Pixel Size", data=pxl_sz)

    plt.figure()
    plt.plot(stage_x_pos, '.')
    plt.title('motor position (um)')
    del img
    del img_dark
    del img_bkg



def get_moving_x_scan_position(scan_id):
    #h = dbv0[scan_id]
    h = tiled_client_fxi[scan_id]
    det_name = h.start["detectors"][0]
    timestamp_img = list(h["primary"]["data"][f"{det_name}_timestamps"])
    assert "zps_sx_monitor" in h.stream_names
    pos = h["zps_sx_monitor"].read()
    timestamp_mot = convert_AD_timestamps(pd.Series(pos["time"]))

    img_ini_timestamp = timestamp_img[0]
    mot_ini_timestamp = timestamp_mot[0]

    n = len(timestamp_mot)


    img_time = timestamp_img - img_ini_timestamp
    mot_time = timestamp_mot - mot_ini_timestamp

    mot_pos = np.array(pos["zps_sx"])
    n = len(mot_pos)
    idx = 1
    for i in range(1, n):
        if mot_pos[i] - mot_pos[i-1] > 0.1:
            break
        else:
            idx += 1
    mot_time = mot_time[idx:]
    mot_time = mot_time - mot_time[0]
    mot_pos = mot_pos[idx:]
    mot_pos_interp = np.interp(img_time, mot_time, mot_pos)

    img_pos = mot_pos_interp
    return img_pos






















#
