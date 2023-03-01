from pathlib import Path
import datetime
import os


import numpy as np
import matplotlib.pylab as plt
import h5py
import pprint
from operator import attrgetter
from PIL import Image
from scipy.signal import medfilt2d
from datetime import datetime  

def check_latest_scan_id(init_guess=60000, search_size=100):
    sid_from_md = RE.md["scan_id"]
    if len(list(db(scan_id=sid_from_md))) > 0:
        sid_i = max(db[-1].start["scan_id"], sid_from_md, init_guess)
    else:  # current RE.md['scan_id'] is an empty scan, e.g., someone set RE.md['scan_id'] mistakely
        sid_i = max(db[-1].start["scan_id"], init_guess)
    sid = sid_i
    n = len(list(db(scan_id=sid)))
    if len(list(db(scan_id=sid))) == 1:
        for i in range(1, 11):
            if len(list(db(scan_id=sid + i))) == 1:
                break
        if i == 10:
            return print(f'\nThe latest scan_id is {sid}, set RE.md["scan_id"]={sid}')
    while n > 0:
        sid_i = sid
        sid += search_size
        n = len(list(db(scan_id=sid)))
        print(sid)
    sid_n = sid
    sid = int((sid_i + sid_n) / 2)
    print(f"sid_i = {sid_i}, sid_n = {sid_n}")
    while 1:
        print(f"sid_i = {sid_i}, sid_n = {sid_n} --> sid = {sid}")
        n = len(list(db(scan_id=sid)))
        if n > 0:
            sid_i = sid
        else:  # n=0: scan_id is empty
            sid_n = sid
        sid = int((sid_i + sid_n) / 2)
        #    print(f'sid_i = {sid_i}, sid_n = {sid_n} --> sid = {sid}')
        if sid_n - sid_i <= 1:
            break
    tmp = []
    for i in range(10):  # check following 10 scans if any scan_id has not be used ever
        tmp.append(len(list(db(scan_id=sid + i))))
    tmp_len_equal_1 = np.where(np.array(tmp) == 1)[0]
    if len(tmp_len_equal_1):
        sid = sid + tmp_len_equal_1[-1]
    sid = int(sid)
    while not (len(list(db(scan_id=sid))) == 1 and len(list(db(scan_id=sid + 1))) == 0):
        sid += 1
    RE.md["scan_id"] = sid
    return print(f'\nThe latest scan_id is {sid}, set RE.md["scan_id"]={sid}')


def change_hdf5_source(cam, roi_name):
    yield from bps.mov(cam.hdf5.nd_array_port, roi_name)
    yield from bps.abs_set(cam.cam.acquire, 1)
    # TODO set to just 1 frame
    # TODO wait a msarter amount of time
    yield from bps.sleep(1)


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def cal_global_mag(x1, y1, x2, y2, nominal_dist=10000):
    return 650.0 / (nominal_dist / distance(x1, y1, x2, y2))


###################### new record caliber position function ################3


def record_calib_pos_new(n):
    global GLOBAL_MAG, CALIBER

    # CALIBER[f'chi2_pos{n}'] = pzt_dcm_chi2.pos.value
    CALIBER[f"chi2_pos{n}"] = dcm.chi2.position
    CALIBER[f"XEng_pos{n}"] = XEng.position
    CALIBER[f"zp_x_pos{n}"] = zp.x.position
    CALIBER[f"zp_y_pos{n}"] = zp.y.position
    #CALIBER[f"th2_motor_pos{n}"] = th2_motor.position
    CALIBER[f"th2_motor_pos{n}"] = dcm.th2.position
    CALIBER[f"clens_x_pos{n}"] = clens.x.position
    CALIBER[f"clens_y1_pos{n}"] = clens.y1.position
    CALIBER[f"clens_y2_pos{n}"] = clens.y2.position
    CALIBER[f"clens_p_pos{n}"] = clens.p.position
    CALIBER[f"DetU_y_pos{n}"] = DetU.y.position
    CALIBER[f"DetU_x_pos{n}"] = DetU.x.position
    CALIBER[f"aper_x_pos{n}"] = aper.x.position
    CALIBER[f"aper_y_pos{n}"] = aper.y.position
    CALIBER[f"txm_x_pos{n}"] = zps.pi_x.position

    mag = (DetU.z.position / zp.z.position - 1) * GLOBAL_VLM_MAG
    CALIBER[f"mag{n}"] = np.round(mag * 100) / 100.0
    GLOBAL_MAG = CALIBER[f"mag{n}"]

    tmp = {}
    for k in CALIBER.keys():
        if str(n) in k:
            tmp[k] = CALIBER[k]
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(tmp)
    df = pd.DataFrame.from_dict(CALIBER, orient="index")
    df.to_csv("/nsls2/data/fxi-new/legacy/log/calib_new.csv")
    # df.to_csv("/home/xf18id/.ipython/profile_collection/startup/calib_new.csv", sep="\t")
    print(
        f'calib_pos{n} recored: current Magnification = GLOBAL_MAG = {CALIBER[f"mag{n}"]}'
    )


def remove_caliber_pos(n):
    global CALIBER_FLAG, CURRENT_MAG, CALIBER
    df = pd.DataFrame.from_dict(CALIBER, orient="index")
    df.to_csv("/nsls2/data/fxi-new/legacy/log/calib_backup.csv")
    CALIBER_backup = CALIBER.copy()
    try:
        for k in CALIBER_backup.keys():
            if k[-1] == str(n):
                del CALIBER[k]
        df = pd.DataFrame.from_dict(CALIBER, orient="index")
        # df.to_csv("/home/xf18id/.ipython/profile_collection/startup/calib_new.csv", sep="\t")
        df.to_csv("/nsls2/data/fxi-new/legacy/log/calib_new.csv")
    except:
        CALIBER = CALIBER.copy()
        print(f"fails to remove CALIBER postion {n}, or it does not exist")
        print("CALIBER not changed")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(CALIBER)


def print_caliber(print_eng_only=1, pos=-1):
    if print_eng_only:
        for k in CALIBER.keys():
            if "XEng" in k:
                print(f"{k}: {CALIBER[k]:2.5f} keV")
        print(
            'If want to display full list of motor position, use "print_caliber(0, pos=1)"'
        )
        print(
            "e.g., print_caliber(0, pos=-1) will display all recorded position in detail"
        )
        print("e.g., print_caliber(0, pos=1) will display will position 1 only")

    else:
        if pos == -1:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(CALIBER)
        else:
            tmp = {}
            for k in CALIBER.keys():
                if k[-1] == str(pos):
                    tmp[k] = CALIBER[k]
                    print(f"{k:>20s}: {CALIBER[k]:4.8f}")
            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(tmp)


def read_calib_file_new(return_flag=0):
    # fn = "/home/xf18id/.ipython/profile_collection/startup/calib_new.csv"
    fn = "/nsls2/data/fxi-new/legacy/log/calib_new.csv"
    df = pd.read_csv(fn, index_col=0)
    d = df.to_dict("split")
    d = dict(zip(d["index"], d["data"]))

    for k in d.keys():
        CALIBER[k] = np.float(d[k][0])

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(CALIBER)
    print("Energy caliberated at: ")
    print_caliber()
    # except:
    #    print(f'fails to read calibriation file')
    #    CALIBER = {}
    if return_flag:
        return CALIBER


def move_zp_ccd(eng_new, move_flag=1, info_flag=1, move_clens_flag=0, move_det_flag=0):
    """
    move the zone_plate and ccd to the user-defined energy with constant magnification
    use the function as:
        move_zp_ccd_with_const_mag(eng_new=8.0, move_flag=1)
    Note:
        in the above commend, it will use two energy calibration points to calculate the motor
        position of XEng=8.0 keV.
        specfically, one of the calibration points is > 8keV, the other one is < 8keV

    Inputs:
    -------
    eng_new:  float
          User defined energy, in unit of keV
    flag: int
          0: Do calculation without moving real stages
          1: Will move stages
    """
    def find_nearest(data, value):
        data = np.array(data)
        return np.abs(data - value).argmin()

    eng_new = float(eng_new)  # eV, e.g. 9.0
    det = DetU  # upstream detector
    eng_ini = XEng.position
    check_eng_range([eng_ini])
    zp_ini, det_ini, zp_delta, det_delta, zp_final, det_final = cal_zp_ccd_position(
        eng_new, eng_ini, print_flag=0
    )

    assert (det_final) > det.z.low_limit.value and (det_final) < det.z.high_limit.value, print(
        "Trying to move DetU to {0:2.2f}. Movement is out of travel range ({1:2.2f}, {2:2.2f})\nTry to move the bottom stage manually.".format(
            det_final, det.z.low_limit.value, det.z.high_limit.value
        )
    )

    ENG_val = []
    ENG_idx = []
    for k in CALIBER.keys():
        if "XEng" in k:
            ENG_val.append(CALIBER[k])
            ENG_idx.append(int(k[-1]))
    t = find_nearest(eng_new, ENG_val)
    eng1 = ENG_val.pop(t)
    id1 = ENG_idx.pop(t)

    ENG_val_copy = np.array(ENG_val.copy())
    ENG_idx_copy = np.array(ENG_idx.copy())
    if eng_new >= eng1:
        idx = ENG_val_copy > eng_new
    else:
        idx = ENG_val_copy < eng_new

    ENG_val_copy = ENG_val_copy[idx]
    ENG_idx_copy = ENG_idx_copy[idx]
    if len(ENG_val_copy):
        t = find_nearest(eng_new, ENG_val_copy)
        eng2 = ENG_val_copy[t]
        id2 = ENG_idx_copy[t]
    else:
        t = find_nearest(eng_new, ENG_val)
        eng2 = ENG_val[t]
        id2 = ENG_idx[t]

    mag1 = CALIBER[f"mag{id1}"]
    mag2 = CALIBER[f"mag{id2}"]

    if not (np.abs(mag1 / mag2 - 1) < 1e-3):
        print(
            f"mismatch in magfinication:\nmagnificatoin at {eng1:5f} keV = {mag1}\nmagnificatoin at {eng2:2.5f} keV = {mag2}\nwill not move"
        )
        return 0
    else:
        print(
            f"using reference at {eng1:2.5f} keV and {eng2:2.5f} kev to interpolate\n"
        )
        dcm_chi2_eng1 = CALIBER[f"chi2_pos{id1}"]
        zp_x_pos_eng1 = CALIBER[f"zp_x_pos{id1}"]
        zp_y_pos_eng1 = CALIBER[f"zp_y_pos{id1}"]
        th2_motor_eng1 = CALIBER[f"th2_motor_pos{id1}"]
        clens_x_eng1 = CALIBER[f"clens_x_pos{id1}"]
        clens_y1_eng1 = CALIBER[f"clens_y1_pos{id1}"]
        clens_y2_eng1 = CALIBER[f"clens_y2_pos{id1}"]
        clens_p_eng1 = CALIBER[f"clens_p_pos{id1}"]
        DetU_x_eng1 = CALIBER[f"DetU_x_pos{id1}"]
        DetU_y_eng1 = CALIBER[f"DetU_y_pos{id1}"]
        aper_x_eng1 = CALIBER[f"aper_x_pos{id1}"]
        aper_y_eng1 = CALIBER[f"aper_y_pos{id1}"]

        dcm_chi2_eng2 = CALIBER[f"chi2_pos{id2}"]
        zp_x_pos_eng2 = CALIBER[f"zp_x_pos{id2}"]
        zp_y_pos_eng2 = CALIBER[f"zp_y_pos{id2}"]
        th2_motor_eng2 = CALIBER[f"th2_motor_pos{id2}"]
        clens_x_eng2 = CALIBER[f"clens_x_pos{id2}"]
        clens_y1_eng2 = CALIBER[f"clens_y1_pos{id2}"]
        clens_y2_eng2 = CALIBER[f"clens_y2_pos{id2}"]
        clens_p_eng2 = CALIBER[f"clens_p_pos{id2}"]
        DetU_x_eng2 = CALIBER[f"DetU_x_pos{id2}"]
        DetU_y_eng2 = CALIBER[f"DetU_y_pos{id2}"]
        aper_x_eng2 = CALIBER[f"aper_x_pos{id2}"]
        aper_y_eng2 = CALIBER[f"aper_y_pos{id2}"]

        if np.abs(eng1 - eng2) < 1e-5:  # difference less than 0.01 eV
            print(
                f'eng1({eng1:2.5f} eV) and eng2({eng2:2.5f} eV) in "CALIBER" are two close, will not move any motors...'
            )
        else:

            chi2_motor_target = (eng_new - eng2) * (dcm_chi2_eng1 - dcm_chi2_eng2) / (
                eng1 - eng2
            ) + dcm_chi2_eng2
            zp_x_target = (eng_new - eng2) * (zp_x_pos_eng1 - zp_x_pos_eng2) / (
                eng1 - eng2
            ) + zp_x_pos_eng2
            zp_y_target = (eng_new - eng2) * (zp_y_pos_eng1 - zp_y_pos_eng2) / (
                eng1 - eng2
            ) + zp_y_pos_eng2
            th2_motor_target = (eng_new - eng2) * (th2_motor_eng1 - th2_motor_eng2) / (
                eng1 - eng2
            ) + th2_motor_eng2
            clens_x_target = (eng_new - eng2) * (clens_x_eng1 - clens_x_eng2) / (
                eng1 - eng2
            ) + clens_x_eng2
            clens_y1_target = (eng_new - eng2) * (clens_y1_eng1 - clens_y1_eng2) / (
                eng1 - eng2
            ) + clens_y1_eng2
            clens_y2_target = (eng_new - eng2) * (clens_y2_eng1 - clens_y2_eng2) / (
                eng1 - eng2
            ) + clens_y2_eng2
            clens_p_target = (eng_new - eng2) * (clens_p_eng1 - clens_p_eng2) / (
                eng1 - eng2
            ) + clens_p_eng2
            DetU_x_target = (eng_new - eng2) * (DetU_x_eng1 - DetU_x_eng2) / (
                eng1 - eng2
            ) + DetU_x_eng2
            DetU_y_target = (eng_new - eng2) * (DetU_y_eng1 - DetU_y_eng2) / (
                eng1 - eng2
            ) + DetU_y_eng2
            aper_x_target = (eng_new - eng2) * (aper_x_eng1 - aper_x_eng2) / (
                eng1 - eng2
            ) + aper_x_eng2
            aper_y_target = (eng_new - eng2) * (aper_y_eng1 - aper_y_eng2) / (
                eng1 - eng2
            ) + aper_y_eng2

            dcm_chi2_ini = dcm.chi2.position
            zp_x_ini = zp.x.position
            zp_y_ini = zp.y.position
            th2_motor_ini = dcm.th2.position
            clens_x_ini = clens.x.position
            clens_y1_ini = clens.y1.position
            clens_y2_ini = clens.y2.position
            clens_p_ini = clens.p.position
            DetU_x_ini = DetU.x.position
            DetU_y_ini = DetU.y.position
            aper_x_ini = aper.x.position
            aper_y_ini = aper.y.position

            if move_flag:  # move stages
                print("Now moving stages ....")
                if info_flag:
                    print(
                        "Energy: {0:5.2f} keV --> {1:5.2f} keV".format(eng_ini, eng_new)
                    )
                    print(
                        "zone plate position: {0:2.4f} mm --> {1:2.4f} mm".format(
                            zp_ini, zp_final
                        )
                    )
                    print(
                        "CCD position: {0:2.4f} mm --> {1:2.4f} mm".format(
                            det_ini, det_final
                        )
                    )
                    print(
                        "move zp_x: ({0:2.4f} um --> {1:2.4f} um)".format(
                            zp_x_ini, zp_x_target
                        )
                    )
                    print(
                        "move zp_y: ({0:2.4f} um --> {1:2.4f} um)".format(
                            zp_y_ini, zp_y_target
                        )
                    )
                    # print ('move pzt_dcm_th2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_th2_ini, pzt_dcm_th2_target))
                    print(
                        "move dcm_chi2: ({0:2.4f} um --> {1:2.4f} um)".format(
                            dcm_chi2_ini, chi2_motor_target
                        )
                    )
                    print(
                        "move th2_motor: ({0:2.6f} deg --> {1:2.6f} deg)".format(
                            th2_motor_ini, th2_motor_target
                        )
                    )
                    print(
                        "move aper_x_motor: ({0:2.4f} um --> {1:2.4f} um)".format(
                            aper_x_ini, aper_x_target
                        )
                    )
                    print(
                        "move aper_y_motor: ({0:2.4f} um --> {1:2.4f} um)".format(
                            aper_y_ini, aper_y_target
                        )
                    )
                    if move_clens_flag:
                        print(
                            "move clens_x: ({0:2.4f} um --> {1:2.4f} um)".format(
                                clens_x_ini, clens_x_target
                            )
                        )
                        print(
                            "move clens_y1: ({0:2.4f} um --> {1:2.4f} um)".format(
                                clens_y1_ini, clens_y1_target
                            )
                        )
                        print(
                            "move clens_y2: ({0:2.4f} um --> {1:2.4f} um)".format(
                                clens_y1_ini, clens_y2_target
                            )
                        )
                        print(
                            "move clens_p: ({0:2.4f} um --> {1:2.4f} um)".format(
                                clens_p_ini, clens_p_target
                            )
                        )
                    if move_det_flag:
                        print(
                            "move DetU_x: ({0:2.4f} um --> {1:2.4f} um)".format(
                                DetU_x_ini, DetU_x_target
                            )
                        )
                        print(
                            "move DetU_y: ({0:2.4f} um --> {1:2.4f} um)".format(
                                DetU_y_ini, DetU_y_target
                            )
                        )

                yield from mv(zp.x, zp_x_target, zp.y, zp_y_target)
                yield from mv(dcm_th2.feedback_enable, 0)
                yield from mv(dcm_th2.feedback, th2_motor_target)
                yield from mv(dcm_th2.feedback_enable, 1)

                yield from mv(dcm_chi2.feedback_enable, 0)
                yield from mv(dcm_chi2.feedback, chi2_motor_target)
                yield from mv(dcm_chi2.feedback_enable, 1)

                yield from mv(zp.z, zp_final, det.z, det_final, XEng, eng_new)
                yield from mv(aper.x, aper_x_target, aper.y, aper_y_target)
                if move_clens_flag:
                    yield from mv(
                        clens.x,
                        clens_x_target,
                        clens.y1,
                        clens_y1_target,
                        clens.y2,
                        clens_y2_target,
                    )
                    yield from mv(clens.p, clens_p_target)
                if move_det_flag:
                    yield from mv(DetU.x, DetU_x_target)
                    yield from mv(DetU.y, DetU_y_target)
                # yield from mv(pzt_dcm_th2.setpos, pzt_dcm_th2_target, pzt_dcm_chi2.setpos, pzt_dcm_chi2_target)
                # yield from mv(pzt_dcm_chi2.setpos, pzt_dcm_chi2_target)

                yield from bps.sleep(0.1)
                if abs(eng_new - eng_ini) >= 0.005:
                    t = 10 * abs(eng_new - eng_ini)
                    t = min(t, 2)
                    print(f"sleep for {t} sec")
                    yield from bps.sleep(t)
            else:
                print("This is calculation. No stages move")
                print(
                    "Will move Energy: {0:5.2f} keV --> {1:5.2f} keV".format(
                        eng_ini, eng_new
                    )
                )
                print(
                    "will move zone plate down stream by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)".format(
                        zp_delta, zp_ini, zp_final
                    )
                )
                print(
                    "will move CCD down stream by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)".format(
                        det_delta, det_ini, det_final
                    )
                )
                print(
                    "will move zp_x: ({0:2.4f} um --> {1:2.4f} um)".format(
                        zp_x_ini, zp_x_target
                    )
                )
                print(
                    "will move zp_y: ({0:2.4f} um --> {1:2.4f} um)".format(
                        zp_y_ini, zp_y_target
                    )
                )
                print(
                    "will move aper_x: ({0:2.4f} um --> {1:2.4f} um)".format(
                        aper_x_ini, aper_x_target
                    )
                )
                print(
                    "will move aper_y: ({0:2.4f} um --> {1:2.4f} um)".format(
                        aper_y_ini, aper_y_target
                    )
                )
                # print ('will move pzt_dcm_th2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_th2_ini, pzt_dcm_th2_target))
                print(
                    "will move dcm_chi2: ({0:2.4f} um --> {1:2.4f} um)".format(
                        dcm_chi2_ini, chi2_motor_target
                    )
                )
                print(
                    "will move th2_motor: ({0:2.6f} deg --> {1:2.6f} deg)".format(
                        th2_motor_ini, th2_motor_target
                    )
                )
                if move_clens_flag:
                    print(
                        "will move clens_x: ({0:2.4f} um --> {1:2.4f} um)".format(
                            clens_x_ini, clens_x_target
                        )
                    )
                    print(
                        "will move clens_y1: ({0:2.4f} um --> {1:2.4f} um)".format(
                            clens_y1_ini, clens_y1_target
                        )
                    )
                    print(
                        "will move clens_y2: ({0:2.4f} um --> {1:2.4f} um)".format(
                            clens_y1_ini, clens_y2_target
                        )
                    )
                    print(
                        "will move clens_p: ({0:2.4f} um --> {1:2.4f} um)".format(
                            clens_p_ini, clens_p_target
                        )
                    )
                if move_det_flag:
                    print(
                        "will move DetU_x: ({0:2.6f} mm --> {1:2.6f} mm)".format(
                            DetU_x_ini, DetU_x_target
                        )
                    )
                    print(
                        "will move DetU_y: ({0:2.6f} mm --> {1:2.6f} mm)".format(
                            DetU_y_ini, DetU_y_target
                        )
                    )
            return 1


################################


def show_global_para():
    print(f"GLOBAL_MAG = {GLOBAL_MAG} X")  # total magnification
    print(f"GLOBAL_VLM_MAG = {GLOBAL_VLM_MAG} X")  # vlm magnification
    print(f"OUT_ZONE_WIDTH = {OUT_ZONE_WIDTH} nm")  # 30 nm
    print(f"ZONE_DIAMETER = {ZONE_DIAMETER} um")  # 200 um
    for k in CALIBER.keys():
        if "mag" in k:
            print(f"{k} = {CALIBER[k]} X")
    print(f"\nFor Andor camera, current pixel size = {6500./GLOBAL_MAG:3.1f} nm")
    print("\nChange parameters if necessary.\n\n")


# def list_fun():
#    import umacro
#    all_func = inspect.getmembers(umacro, inspect.isfunction)
#    return all_func

################################################################
####################  create new user  #########################
################################################################


def new_user(*, new_pi_name=None, new_proposal_id=None):
    """
    The function creates directory structure for a new user. If ``new_pi_name`` and/or
    ``new_proposal_id`` are ``None``, the function asks the user to type PI name and/or
    Proposal ID.
    """
    now = datetime.datetime.now()
    year = np.str(now.year)

    # this is really cycle not quarter
    qut = f"Q{1 + (now.month - 1) // 4}"

    pre = Path(f"/nsls2/data/fxi-new/legacy/users/{year}{qut}/")
    pre.mkdir(parents=True, exist_ok=True)

    print("\n")

    if new_pi_name is None:
        PI_name = input("PI's name:")
    else:
        PI_name = new_pi_name
    PI_name = PI_name.replace(" ", "_").upper()

    if len(PI_name) == 0 or PI_name[0] == "*":
        print(f"\nstay at current directory: {os.getcwd()}\n")
        return
    elif PI_name[:4] == "COMM":
        PI_name = "FXI_commission"
        fn = pre / PI_name
        print(fn)
    else:
        if new_proposal_id is None:
            proposal_id = input("Proposal ID:")
        else:
            proposal_id = new_proposal_id

        fn = pre / f"{PI_name}_Proposal_{proposal_id}"
        export_pdf(1)
        insert_text(f"New user: {fn}\n")
        export_pdf(1)
    fn.mkdir(parents=True, exist_ok=True)

    os.chdir(fn)
    print("\nUser creating successful!\n\nEntering folder: {}\n".format(os.getcwd()))


def create_new_user(pi_name: str, proposal_id: str):
    """
    The plans create directory structure for the new user. Parameters 'pi_name' and
    'proposal_id' are expected to be strings.

    Parameters
    ----------
    pi_name : str
        PI name
    propsal_id : str
        Proposal ID
    """
    new_user(new_pi_name=pi_name, new_proposal_id=proposal_id)
    yield from sleep(0.1)  # Short pause


################################################################
####################   TXM paramter  ###########################
################################################################


from bluesky.callbacks import CallbackBase


class PdfMaker(CallbackBase):
    def start(self, doc):
        self._last_start = doc
        print("HI")


#    def stop(self, stop):
#        doc = self._last_start
#        scan_id = doc['scan_id']
#        uid = doc['uid']
#        X_eng = f'{h.start["XEng"]:2.4f}'
#        scan_type = doc['plan_name']
#        txt = ''
#        for key, val in doc['plan_args'].items():
#            txt += f'{key}={val}, '
#        txt0 = f'#{scan_id}  (uid: {uid[:6]},  X_Eng: {X_eng} keV)\n'
#        txt = txt0 + scan_type + '(' + txt[:-2] + ')'
#        insert_text(txt)
#        print('this is from callback')


def check_eng_range(eng):
    """
    check energy in range of 4.000-12.000
    Inputs:
    --------
    eng: list
        e.g. [6.000,7.500]
    """
    eng = list(eng)
    high_limit = 12.000
    low_limit = 4.000
    for i in range(len(eng)):
        assert (
            eng[i] >= low_limit and eng[i] <= high_limit
        ), "Energy is outside the range (4.000, 12.000) keV"
    return


def cal_parameter(eng, print_flag=1):
    """
    Calculate parameters for given X-ray energy
    Use as: wave_length, focal_length, NA, DOF = energy_cal(Eng, print_flag=1):

    Inputs:
    -------
    eng: float
         X-ray energy, in unit of keV
    print_flag: int
        0: print outputs
        1: no print

    Outputs:
    --------
    wave_length(nm), focal_length(mm), NA(rad if print_flag=1, mrad if print_flag=0), DOF(mm)
    """

    global OUT_ZONE_WIDTH
    global ZONE_DIAMETER

    h = 6.6261e-34
    c = 3e8
    ec = 1.602e-19

    #    if eng < 4000:    eng = XEng.position * 1000 # current beam energy
    check_eng_range([eng])

    wave_length = h * c / (ec * eng * 1000) * 1e9  # nm
    focal_length = OUT_ZONE_WIDTH * ZONE_DIAMETER / (wave_length) / 1000  # mm
    NA = wave_length / (2 * OUT_ZONE_WIDTH)
    DOF = wave_length / NA**2 / 1000  # um
    if print_flag:
        print(
            "Wave length: {0:2.2f} nm\nFocal_length: {1:2.2f} mm\nNA: {2:2.2f} mrad\nDepth of focus: {3:2.2f} um".format(
                wave_length, focal_length, NA * 1e3, DOF
            )
        )
    else:
        return wave_length, focal_length, NA, DOF


def cal_zp_ccd_position(eng_new, eng_ini=0, print_flag=1):

    """
    calculate the delta amount of movement for zone_plate and CCD whit change energy from ene_ini to eng_new while keeping same magnification
    E.g. delta_zp, delta_det, final_zp, final_det = cal_zp_ccd_with_const_mag(eng_new=8000, eng_ini=0)

    Inputs:
    -------
    eng_new:  float
          User defined energy, in unit of keV
    eng_ini:  float
          if eng_ini < 4.000 (keV), will eng_ini = current Xray energy
    print_flag: int
          0: Do calculation without moving real stages
          1: Will move stages


    Outputs:
    --------
    zp_ini: float
        initial position of zone_plate
    det_ini: float
        initial position of detector
    zp_delta: float
        delta amount of zone_plate movement
        positive means move downstream, and negative means move upstream
    det_delta: float
        delta amount of CCD movement
        positive means move downstream, and negative means move upstream
    zp_final: float
        final position of zone_plate
    det_final: float
        final position of detector

    """

    global GLOBAL_MAG
    global GLOBAL_VLM_MAG

    if eng_ini < 4.000:
        eng_ini = XEng.position  # current beam energy
    check_eng_range([eng_new, eng_ini])

    h = 6.6261e-34
    c = 3e8
    ec = 1.602e-19

    det = DetU  # read current energy and motor position
    mag = GLOBAL_MAG / GLOBAL_VLM_MAG

    zp_ini = zp.z.position  # zone plate position in unit of mm
    zps_ini = zps.sz.position  # sample position in unit of mm
    det_ini = det.z.position  # detector position in unit of mm

    lamda_ini, fl_ini, _, _ = cal_parameter(eng_ini, print_flag=0)
    lamda, fl, _, _ = cal_parameter(eng_new, print_flag=0)

    p_ini = fl_ini * (mag + 1) / mag  # sample distance (mm), relative to zone plate
    q_ini = mag * p_ini  # ccd distance (mm), relative to zone plate

    p_cal = fl * (mag + 1) / mag
    q_cal = mag * p_cal

    zp_delta = p_cal - p_ini
    det_delta = q_cal - q_ini + zp_delta

    zp_final = p_cal
    det_final = p_cal * (mag + 1)

    if print_flag:
        print("Calculation results:")
        print("Change energy from: {0:2.2f} eV to {1:2.2f} eV".format(eng_ini, eng_new))
        print(
            "Need to move zone plate by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)".format(
                zp_delta, zp_ini, zp_final
            )
        )
        print(
            "Need to move CCD by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)".format(
                det_delta, det_ini, det_final
            )
        )
    else:
        return zp_ini, det_ini, zp_delta, det_delta, zp_final, det_final


'''
def move_zp_ccd(eng_new, move_flag=1, info_flag=1, move_clens_flag=0, move_det_flag=0):
    """
    move the zone_plate and ccd to the user-defined energy with constant magnification
    use the function as:
        move_zp_ccd_with_const_mag(eng_new=8.0, move_flag=1):

    Inputs:
    -------
    eng_new:  float
          User defined energy, in unit of keV
    flag: int
          0: Do calculation without moving real stages
          1: Will move stages
    """
    global CALIBER_FLAG
    if CALIBER_FLAG:
        eng_new = float(eng_new) # eV, e.g. 9.0
        det = DetU # upstream detector
        eng_ini = XEng.position
        check_eng_range([eng_ini])
        zp_ini, det_ini, zp_delta, det_delta, zp_final, det_final = cal_zp_ccd_position(eng_new, eng_ini, print_flag=0)

        assert ((det_final) > det.z.low_limit and (det_final) < det.z.high_limit), print ('Trying to move DetU to {0:2.2f}. Movement is out of travel range ({1:2.2f}, {2:2.2f})\nTry to move the bottom stage manually.'.format(det_final, det.z.low_limit, det.z.high_limit))

        eng1 = CALIBER['XEng_pos1']
        eng2 = CALIBER['XEng_pos2']

        #pzt_dcm_th2_eng1 = CALIBER['th2_pos1']
        dcm_chi2_eng1 = CALIBER['chi2_pos1']
        zp_x_pos_eng1 = CALIBER['zp_x_pos1']
        zp_y_pos_eng1 = CALIBER['zp_y_pos1']
        th2_motor_eng1 = CALIBER['th2_motor_pos1']
        clens_x_eng1 = CALIBER['clens_x_pos1']
        clens_y1_eng1 = CALIBER['clens_y1_pos1']
        clens_y2_eng1 = CALIBER['clens_y2_pos1']
        clens_p_eng1 = CALIBER['clens_p_pos1']
        DetU_x_eng1 = CALIBER['DetU_x_pos1']
        DetU_y_eng1 = CALIBER['DetU_y_pos1']
        aper_x_eng1 = CALIBER['aper_x_pos1']
        aper_y_eng1 = CALIBER['aper_y_pos1']

        #pzt_dcm_th2_eng2 = CALIBER['th2_pos2']
        pzt_dcm_chi2_eng2 = CALIBER['chi2_pos2']
        zp_x_pos_eng2 = CALIBER['zp_x_pos2']
        zp_y_pos_eng2 = CALIBER['zp_y_pos2']
        th2_motor_eng2 = CALIBER['th2_motor_pos2']
        clens_x_eng2 = CALIBER['clens_x_pos2']
        clens_y1_eng2 = CALIBER['clens_y1_pos2']
        clens_y2_eng2 = CALIBER['clens_y2_pos2']
        clens_p_eng2 = CALIBER['clens_p_pos2']
        DetU_x_eng2 = CALIBER['DetU_x_pos2']
        DetU_y_eng2 = CALIBER['DetU_y_pos2']
        aper_x_eng2 = CALIBER['aper_x_pos2']
        aper_y_eng2 = CALIBER['aper_y_pos2']

        if np.abs(eng1 - eng2) < 1e-5: # difference less than 0.01 eV
            print(f'eng1({eng1:2.5f} eV) and eng2({eng2:2.5f} eV) in "CALIBER" are two close, will not move any motors...')
        else:
            #pzt_dcm_th2_target = (eng_new - eng2) * (pzt_dcm_th2_eng1 - pzt_dcm_th2_eng2) / (eng1-eng2) + pzt_dcm_th2_eng2
            pzt_dcm_chi2_target = (eng_new - eng2) * (dcm_chi2_eng1 - dcm_chi2_eng2) / (eng1-eng2) + pzt_dcm_chi2_eng2
            zp_x_target = (eng_new - eng2)*(zp_x_pos_eng1 - zp_x_pos_eng2)/(eng1 - eng2) + zp_x_pos_eng2
            zp_y_target = (eng_new - eng2)*(zp_y_pos_eng1 - zp_y_pos_eng2)/(eng1 - eng2) + zp_y_pos_eng2
            th2_motor_target = (eng_new - eng2) * (th2_motor_eng1 -th2_motor_eng2) / (eng1-eng2) + th2_motor_eng2
            clens_x_target = (eng_new - eng2)*(clens_x_eng1 - clens_x_eng2)/(eng1 - eng2) + clens_x_eng2
            clens_y1_target = (eng_new - eng2)*(clens_y1_eng1 - clens_y1_eng2)/(eng1 - eng2) + clens_y1_eng2
            clens_y2_target = (eng_new - eng2)*(clens_y2_eng1 - clens_y2_eng2)/(eng1 - eng2) + clens_y2_eng2
            clens_p_target = (eng_new - eng2)*(clens_p_eng1 - clens_p_eng2)/(eng1 - eng2) + clens_p_eng2
            DetU_x_target = (eng_new - eng2)*(DetU_x_eng1 - DetU_x_eng2)/(eng1 - eng2) + DetU_x_eng2
            DetU_y_target = (eng_new - eng2)*(DetU_y_eng1 - DetU_y_eng2)/(eng1 - eng2) + DetU_y_eng2
            aper_x_target = (eng_new - eng2)*(aper_x_eng1 - aper_x_eng2)/(eng1 - eng2) + aper_x_eng2
            aper_y_target = (eng_new - eng2)*(aper_y_eng1 - aper_y_eng2)/(eng1 - eng2) + aper_y_eng2
            #pzt_dcm_th2_ini = (yield from bps.rd(pzt_dcm_th2.pos))
            pzt_dcm_chi2_ini = (yield from bps.rd(pzt_dcm_chi2.pos.value))
            zp_x_ini = zp.x.position
            zp_y_ini = zp.y.position
            th2_motor_ini = th2_motor.position
            clens_x_ini = clens.x.position
            clens_y1_ini = clens.y1.position
            clens_y2_ini = clens.y2.position
            clens_p_ini = clens.p.position
            DetU_x_ini = DetU.x.position
            DetU_y_ini = DetU.y.position
            aper_x_ini = aper.x.position
            aper_y_ini = aper.y.position

            if move_flag: # move stages
                print ('Now moving stages ....')
                if info_flag:
                    print ('Energy: {0:5.2f} keV --> {1:5.2f} keV'.format(eng_ini, eng_new))
                    print ('zone plate position: {0:2.4f} mm --> {1:2.4f} mm'.format(zp_ini, zp_final))
                    print ('CCD position: {0:2.4f} mm --> {1:2.4f} mm'.format(det_ini, det_final))
                    print ('move zp_x: ({0:2.4f} um --> {1:2.4f} um)'.format(zp_x_ini, zp_x_target))
                    print ('move zp_y: ({0:2.4f} um --> {1:2.4f} um)'.format(zp_y_ini, zp_y_target))
                    #print ('move pzt_dcm_th2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_th2_ini, pzt_dcm_th2_target))
                    print ('move pzt_dcm_chi2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_chi2_ini, pzt_dcm_chi2_target))
                    print ('move th2_motor: ({0:2.6f} deg --> {1:2.6f} deg)'.format(th2_motor_ini, th2_motor_target))
                    print ('move aper_x_motor: ({0:2.4f} um --> {1:2.4f} um)'.format(aper_x_ini, aper_x_target))
                    print ('move aper_y_motor: ({0:2.4f} um --> {1:2.4f} um)'.format(aper_y_ini, aper_y_target))
                    if move_clens_flag:
                        print ('move clens_x: ({0:2.4f} um --> {1:2.4f} um)'.format(clens_x_ini, clens_x_target))
                        print ('move clens_y1: ({0:2.4f} um --> {1:2.4f} um)'.format(clens_y1_ini, clens_y1_target))
                        print ('move clens_y2: ({0:2.4f} um --> {1:2.4f} um)'.format(clens_y1_ini, clens_y2_target))
                        print ('move clens_p: ({0:2.4f} um --> {1:2.4f} um)'.format(clens_p_ini, clens_p_target))
                    if move_det_flag:
                        print ('move DetU_x: ({0:2.4f} um --> {1:2.4f} um)'.format(DetU_x_ini, DetU_x_target))
                        print ('move DetU_y: ({0:2.4f} um --> {1:2.4f} um)'.format(DetU_y_ini, DetU_y_target))

                yield from mv(zp.x, zp_x_target, zp.y, zp_y_target)
#                yield from mv(aper.x, aper_x_target, aper.y, aper_y_target)
                yield from mv(dcm_th2.feedback_enable, 0)
                yield from mv(dcm_th2.feedback, th2_motor_target)
                yield from mv(dcm_th2.feedback_enable, 1)
                yield from mv(zp.z, zp_final,det.z, det_final, XEng, eng_new)
                yield from mv(aper.x,  aper_x_target, aper.y, aper_y_target)
                if move_clens_flag:
                    yield from mv(clens.x, clens_x_target, clens.y1, clens_y1_target, clens.y2, clens_y2_target)
                    yield from mv(clens.p, clens_p_target)
                if move_det_flag:
                    yield from mv(DetU.x, DetU_x_target)
                    yield from mv(DetU.y, DetU_y_target)
                #yield from mv(pzt_dcm_th2.setpos, pzt_dcm_th2_target, pzt_dcm_chi2.setpos, pzt_dcm_chi2_target)
                #yield from mv(pzt_dcm_chi2.setpos, pzt_dcm_chi2_target)

                yield from bps.sleep(0.1)
                if abs(eng_new - eng_ini) >= 0.005:
                    t = 10 * abs(eng_new - eng_ini)
                    t = min(t, 2)
                    print(f'sleep for {t} sec')
                    yield from bps.sleep(t)
            else:
                print ('This is calculation. No stages move')
                print ('Will move Energy: {0:5.2f} keV --> {1:5.2f} keV'.format(eng_ini, eng_new))
                print ('will move zone plate down stream by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)'.format(zp_delta, zp_ini, zp_final))
                print ('will move CCD down stream by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)'.format(det_delta, det_ini, det_final))
                print ('will move zp_x: ({0:2.4f} um --> {1:2.4f} um)'.format(zp_x_ini, zp_x_target))
                print ('will move zp_y: ({0:2.4f} um --> {1:2.4f} um)'.format(zp_y_ini, zp_y_target))
                print ('will move aper_x: ({0:2.4f} um --> {1:2.4f} um)'.format(aper_x_ini, aper_x_target))
                print ('will move aper_y: ({0:2.4f} um --> {1:2.4f} um)'.format(aper_y_ini, aper_y_target))
                #print ('will move pzt_dcm_th2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_th2_ini, pzt_dcm_th2_target))
                print ('will move pzt_dcm_chi2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_chi2_ini, pzt_dcm_chi2_target))
                print ('will move th2_motor: ({0:2.6f} deg --> {1:2.6f} deg)'.format(th2_motor_ini, th2_motor_target))
                if move_clens_flag:
                    print ('will move clens_x: ({0:2.4f} um --> {1:2.4f} um)'.format(clens_x_ini, clens_x_target))
                    print ('will move clens_y1: ({0:2.4f} um --> {1:2.4f} um)'.format(clens_y1_ini, clens_y1_target))
                    print ('will move clens_y2: ({0:2.4f} um --> {1:2.4f} um)'.format(clens_y1_ini, clens_y2_target))
                    print ('will move clens_p: ({0:2.4f} um --> {1:2.4f} um)'.format(clens_p_ini, clens_p_target))
                if move_det_flag:
                    print ('will move DetU_x: ({0:2.6f} mm --> {1:2.6f} mm)'.format(DetU_x_ini, DetU_x_target))
                    print ('will move DetU_y: ({0:2.6f} mm --> {1:2.6f} mm)'.format(DetU_y_ini, DetU_y_target))
    else:
        print('record_calib_pos1() or record_calib_pos2() not excuted successfully...\nWill not move anything')

'''

# def cal_phase_ring_position(eng_new, eng_ini=0, print_flag=1):
#    '''
#    calculate delta amount of phase_ring movement:
#    positive means move down-stream, negative means move up-stream

#    use as:
#        cal_phase_ring_with_const_mag(eng_new=8.0, eng_ini=9.0)

#    Inputs:
#    -------
#    eng_new: float
#        target energy, in unit of keV
#    eng_ini: float
#        initial energy, in unit of keV
#        it will read current Xray energy if eng_ini < 4.0 keV
#
#    '''
#
#    _, fl_ini, _, _ = cal_parameter(eng_ini, print_flag=0)
#    _, fl_new, _, _ = cal_parameter(eng_new, print_flag=0)
#
#    _, _, zp_delta, _, _, _ = cal_zp_ccd_position(eng_new, eng_ini, print_flag=0)
#
#    delta_phase_ring = zp_delta + fl_new - fl_ini
#    if print_flag:
#        print ('Need to move phase ring down-stream by: {0:2.2f} mm'.format(delta_phase_ring))
#        print ('Zone plate position changes by: {0:2.2f} mm'.format(zp_delta))
#        print ('Zone plate focal length changes by: {0:2.2f} mm'.format(fl_new - fl_ini))
#    else:    return delta_phase_ring


# def move_phase_ring(eng_new, eng_ini, flag=1):
#    '''
#    move the phase_ring when change the energy
#
#    use as:
#        move_phase_ring_with_const_mag(eng_new=8.0, eng_ini=9.0, flag=1)

#    Inputs:
#    -------
#    eng_new: float
#        target energy, in unit of keV
#    eng_ini: float
#        initial energy, in unit of keV
#        it will read current Xray energy if eng_ini < 4.0
#    flag: int
#         0: no move
#         1: move stage
#    '''

#    delta_phase_ring = cal_phase_ring_position(eng_new, eng_ini, print_flag=0)
#    if flag:    RE(mvr(phase_ring.z, delta_phase_ring))
#    else:
#        print ('This is calculation. No stages move.')
#        print ('will move phase ring down stream by {0:2.2f} mm'.format(delta_phase_ring))
#    return


def set_ic_dwell_time(dwell_time=1.0):
    if np.abs(dwell_time - 10) < 1e-2:
        ic_rate.value = 3
    elif np.abs(dwell_time - 5) < 1e-2:
        ic_rate.value = 4
    elif np.abs(dwell_time - 2) < 1e-2:
        ic_rate.value = 5
    elif np.abs(dwell_time - 1) < 1e-2:
        ic_rate.value = 6
    elif np.abs(dwell_time - 0.5) < 1e-2:
        ic_rate.value = 7
    elif np.abs(dwell_time - 0.2) < 1e-2:
        ic_rate.value = 8
    elif np.abs(dwell_time - 0.1) < 1e-2:
        ic_rate.value = 9
    else:
        print("dwell_time not in list, set to default value: 1s")
        ic_rate.value = 6


def plot_ssa_ic(scan_id, ic="ic4"):
    h = db[scan_id]
    x = np.array(list(h.data("ssa_v_cen")))
    if len(x) > 0:
        xlabel = "ssa_v_cen"
        xdata = x
    x = np.array(list(h.data("ssa_h_cen")))
    if len(x) > 0:
        xlabel = "ssa_h_cen"
        xdata = x
    x = np.array(list(h.data("ssa_v_gap")))
    if len(x) > 0:
        xlabel = "ssa_v_gap"
        xdata = x
    x = np.array(list(h.data("ssa_h_gap")))
    if len(x) > 0:
        xlabel = "ssa_h_gap"
        xdata = x
    ydata = np.array(list(h.data(ic)))

    plt.figure()
    plt.plot(xdata, -ydata, "r.-")
    plt.xlabel(xlabel)
    plt.ylabel(ic + " counts")
    plt.show()


def read_ic(ics, num, dwell_time=1.0):
    """
    read ion-chamber value
    e.g. RE(read_ic([ic1, ic2], num = 10, dwell_time=0.5))

    Inputs:
    -------
    ics: list of ion-chamber
    num: int
        number of points to record
    dwell_time: float
        in unit of seconds
    """

    set_ic_dwell_time(dwell_time=dwell_time)
    yield from count(ics, num, delay=dwell_time)
    h = db[-1]
    ic_num = len(ics)
    fig = plt.figure()
    x = np.linspace(1, num, num)
    y = np.zeros([ic_num, num])
    for i in range(ic_num):
        y[i] = np.array(list(h.data(ics[i].name)))
        ax = fig.add_subplot(ic_num, 1, i + 1)
        ax.title.set_text(ics[i].name)
        ax.plot(x, y[i], ".-")
    fig.subplots_adjust(hspace=0.5)
    plt.show()
    return y


################################################################
####################   plot scaler  ############################
################################################################


def plot_ic(scan_id=[-1], ics=[]):
    """
    plot ic reading from single/multiple scan(s),
    e.g. plot_ic([-1, -2],['ic3', 'ic4'])
    """
    if type(scan_id) == int:
        scan_id = [scan_id]

    if len(ics) == 0:
        ics = [ic3, ic4]

    plt.figure()
    for sid in scan_id:
        h = db[int(sid)]
        sid = h.start["scan_id"]
        num = int(h.start["plan_args"]["num"])
        try:
            st = h.start["plan_args"]["start"]
            en = h.start["plan_args"]["stop"]
        except:
            try:
                st = h.start["plan_args"]["args"][1]
                en = h.start["plan_args"]["args"][2]
            except:
                st = 1
                en = num
        x = np.linspace(st, en, num)
        y = []
        for ic in ics:
            y = list(h.data(ic.name))
            plt.plot(x, y, ".-", label=f"scan: {sid}, ic: {ic.name}")
            plt.legend()
    plt.title(f"reading of ic: {[ic.name for ic in ics]}")
    plt.show()


def plot2dsum(scan_id=-1, fn="Det_Image", save_flag=0):
    """
    valid only if the scan using Andor or detA1 camera
    """
    h = db[scan_id]
    if scan_id == -1:
        scan_id = h.start["scan_id"]
    if "Andor" in h.start["detectors"]:
        det = "Andor_image"
        find_areaDet = 1
    elif "detA1" in h.start["detectors"]:
        det = "detA1_image"
        find_areaDet = 1
    else:
        find_areaDet = 0
    if find_areaDet:
        img = np.array(list(h.data(det)))
        if len(img.shape) == 4:
            img = np.mean(img, axis=1)
        img[img < 20] = 0
        img_sum = np.sum(np.sum(img, axis=1), axis=1)
        num = int(h.start["plan_args"]["steps"])
        st = h.start["plan_args"]["start"]
        en = h.start["plan_args"]["stop"]
        x = np.linspace(st, en, num)
        plt.figure()
        plt.plot(x, img_sum, "r-.")
        plt.title("scanID: " + str(scan_id) + ":  " + det + "_sum")

        if save_flag:
            with h5py.File(fn, "w") as hf:
                hf.create_dataset("data", data=img)
    else:
        print("AreaDetector is not used in the scan")


def plot1d(scan_id=-1, detectors=[], plot_time_stamp=0, return_flag=0):
    h = db[scan_id]
    scan_id = h.start["scan_id"]
    n = len(detectors)
    if n == 0:
        detectors = h.start["detectors"]
        n = len(detectors)
    pos = h.table()
    try:
        st = h.start["plan_args"]["start"]
        en = h.start["plan_args"]["stop"]
        num = int(h.start["plan_args"]["steps"])
        flag = 0
    except:
        flag = 1
    if flag or plot_time_stamp:
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
        x = mot_time - mot_time[0]
    else:
        x = np.linspace(st, en, num)
    fig = plt.figure()
    y_sig = {}
    for i in range(n):
        det_name = detectors[i]
        if det_name == "detA1" or det_name == "Andor":
            det_name = det_name + "_stats1_total"
        y = np.abs(np.array(list(h.data(det_name))))
        title_txt = f"scan#{scan_id}:   {det_name}"
        ax = fig.add_subplot(n, 1, i + 1)
        y_sig[i] = y
        ax.plot(x, y)
        ax.title.set_text(title_txt)
    if flag:
        plt.xlabel("time (s)")
    else:
        plt.xlabel(f'{h.start["plan_args"]["motor"]} position')
    fig.subplots_adjust(hspace=1)
    plt.show()
    if return_flag:
        return x, y_sig


################################################################
####################    read and save  #########################
################################################################


def readtiff(fn_pre="", num=1, x=[], bkg=0, roi=[]):
    if len(x) == 0:
        x = np.arange(num)
    fn = fn_pre + "_" + "{:03d}".format(1) + ".tif"
    img = np.array(Image.open(fn))
    s = img.shape
    if len(roi) == 0:
        roi = [0, s[0], 0, s[1]]
    img_stack = np.zeros([num, s[0], s[1]])
    img_stack[0] = img
    for i in range(1, num):
        fn = fn_pre + "_" + "{:03d}".format(i + 1) + ".tif"
        img_stack[i] = Image.open(fn)
    bkg = bkg * s[0] * s[1]
    img_stack_roi = img_stack[:, roi[0] : roi[1], roi[2] : roi[3]]
    img_sum = np.sum(np.sum(img_stack_roi, axis=1), axis=1)
    img_sum = img_sum - bkg
    plt.figure()
    plt.plot(x, img_sum, ".-")
    return img_stack, img_stack_roi


def save_hdf_file(fn, *args):
    n = len(args)
    assert n % 2 == 0, "even number of args only"
    n = int(n / 2)
    j = 0
    with h5py.File(fn, "w") as hf:
        for i in range(n):
            j = int(2 * i)
            tmp = args[j + 1]
            hf.create_dataset(args[j], data=tmp)


def export_ic3(scan_id=-1, return_flag=1, save_flag=0, plot_flag=0):
    h = db[scan_id]
    scan_id = h.start["scan_id"]
    pos = h.table()
    scan_id = h.start["scan_id"]
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
    x = mot_time - mot_time[0]
    y = list(h.data("ic3"))

    if save_flag:
        fn = f"ic3_scan_{scan_id}.txt"
        data = np.zeros([len(x), 2])
        data[:, 0] = x
        data[:, 1] = y
        np.savetxt(fn, data)
        print(f"{fn} is saved")
    if plot_flag:
        plt.figure()
        plt.plot(x, y)
    if return_flag:
        return x, y


########################################################################
########################################################################


def print_baseline_list():
    a = list(db[-1].table("baseline"))
    with open("/home/xf18id/Documents/FXI_manual/FXI_baseline_record.txt", "w") as tx:
        i = 1
        for txt in a:
            if not i % 3:
                tx.write("\n")
            tx.write(f"\t{txt:<30}")
            i += 1


def get_img(h, det="Andor", sli=[]):
    "Take in a Header and return a numpy array of detA1 image(s)."
    det_name = f"{det}_image"
    if len(sli) == 2:
        img = np.array(list(h.data(det_name))[sli[0] : sli[1]])
    else:
        img = np.array(list(h.data(det_name)))
    return np.squeeze(img)


def get_scan_parameter(scan_id=-1, print_flag=0):
    h = db[scan_id]
    scan_id = h.start["scan_id"]
    uid = h.start["uid"]
    try:
        X_eng = f'{h.start["XEng"]:2.4f}'
    except:
        X_eng = "n/a"
    scan_type = h.start["plan_name"]
    scan_time = datetime.fromtimestamp(h.start["time"]).strftime("%D %H:%M")

    txt = ""
    for key, val in h.start["plan_args"].items():
        if key == "zone_plate":
            continue
        txt += f"{key}={val}, "
    txt0 = f"#{scan_id}  (uid: {uid[:6]},  X_Eng: {X_eng} keV,  Time: {scan_time})\n"
    txt = txt0 + scan_type + f"({txt[:-2]})\n"
    try:
        txt_tmp = ""
        for zone_plate_key in h.start["plan_args"]["zone_plate"].keys():
            txt_tmp += f"{zone_plate_key}: {val[zone_plate_key]};    "
        txt = txt + "Zone Plate info:  " + txt_tmp
    except:
        pass
    if print_flag:
        print(txt)
    return txt


def get_scan_timestamp_legacy(scan_id, return_flag=0):
    h = db[scan_id]
    scan_id = h.start["scan_id"]
    timestamp = h.start["time"]
    timestamp_conv = convert_AD_timestamps(pd.Series(timestamp))
    scan_year = int(timestamp_conv.dt.year)
    scan_mon = int(timestamp_conv.dt.month)
    scan_day = int(timestamp_conv.dt.day)
    scan_day, scan_hour = int(timestamp_conv.dt.day), int(timestamp_conv.dt.hour)
    scan_min, scan_sec, scan_msec = (
        int(timestamp_conv.dt.minute),
        int(timestamp_conv.dt.second),
        int(timestamp_conv.dt.microsecond),
    )
    scan_time = f"scan#{scan_id}: {scan_year-20:04d}-{scan_mon:02d}-{scan_day:02d}   {scan_hour:02d}:{scan_min:02d}:{scan_sec:02d}"
    print(scan_time)
    if return_flag:
        return scan_time.split("#")[-1]


def get_scan_timestamp(scan_id, return_flag=0):      
    h = db[scan_id]
    scan_id = h.start["scan_id"]
    timestamp = h.start["time"]
    dt = datetime.fromtimestamp(timestamp)
    t = dt.strftime('%Y-%m-%d %H:%M:%S')
    print(t)
    if return_flag:
        return t


def get_scan_file_name(scan_id):
    hdr = db[scan_id]
    #    print(scan_id, hdr.stop['exit_status'])
    res_uids = list(db.get_resource_uids(hdr))
    for i, uid in enumerate(res_uids):
        res_doc = db.reg.resource_given_uid(uid)
    #        print("   ", i, res_doc)
    fpath_root = res_doc["root"]
    fpath_relative = res_doc["resource_path"]
    fpath = fpath_root + "/" + fpath_relative
    fpath_remote = "/nsls2/xf18id1/backup/DATA/Andor/" + fpath_relative
    return print(f"local path: {fpath}\nremote path: {fpath_remote}")


def get_scan_motor_pos(scan_id):
    df = db[scan_id].table("baseline").T
    mot = BlueskyMagics.positioners
    for i in mot:
        try:
            mot_name = i.name
            if mot_name[:3] == "pzt":
                print(f"{mot_name:>16s}  :: {df[1][mot_name]: 14.6f} ")
            else:
                mot_parent_name = i.parent.name
                offset_name = f"{mot_name}_user_offset"
                offset_dir = f"{mot_name}_user_offset_dir"
                offset_val = db[scan_id].config_data(mot_parent_name)["baseline"][0][
                    offset_name
                ]
                offset_dir_val = db[scan_id].config_data(mot_parent_name)["baseline"][
                    0
                ][offset_dir]
                print(
                    f"{mot_name:>16s}  :: {df[1][mot_name]: 14.6f} {i.motor_egu.value:>4s}  --->  {df[2][mot_name]: 14.6f} {i.motor_egu.value:>4s}      offset = {offset_val: 14.6f}    {offset_dir_val: 1d}"
                )
        except:
            pass
    try:
        for tmp in db[
            scan_id
        ].start.keys():  # if 'T_enabled' has been assigned in md in scan()'
            if "T_" in tmp:
                get_lakeshore_param(scan_id, print_flag=1)
                break
    except:
        pass


def reprint_scan(scan_id):
    from bluesky.callbacks.best_effort import BestEffortCallback

    mybec = BestEffortCallback()
    h = db[scan_id]
    for name, doc in h.documents():
        mybec(name, doc)


def get_lakeshore_param(scan_id, print_flag=0, return_flag=0):
    h = db[scan_id]
    df = h.table("baseline").T
    mot_info = {}
    for mot in motor_lakeshore:
        mot_info[f"{mot.name}"] = df[1][mot.name]
        if print_flag:
            print(f"{mot.name:32s}::   {df[1][mot.name]:>4.1f}")
    if return_flag:
        return mot_info


def split_fly_scan(fn, num=1):
    f = h5py.File(fn, 'r')
    img_bkg = np.array(f['img_bkg'])
    img_bkg_avg = np.array(f['img_bkg_avg'])
    img_dark = np.array(f['img_dark'])
    img_dark_avg = np.array(f['img_dark_avg'])
    x_eng = np.float32(f['X_eng'])
    sid = np.int32(f['scan_id'])
    uid = str(f['uid'])
    pix = f['Pixel Size']

    ang = np.array(f['angle'])
    n_ang = len(ang)
    ang_max = np.max(np.abs(ang))
    ang_min = np.min(np.abs(ang))
    if ang[-1] > ang[0]:
        direction = 1
    else:
        direction = -1
    t = ang_max - ang_min - 180
    if t <= 0:
        print('angle spans less than 180 degrees, will not do anything')
        return 0
    ang_start = np.linspace(0, t, num, endpoint=True)
    for i in range(num):
        id_s = find_nearest(ang, ang[0] + ang_start[i] * direction)
        id_e = find_nearest(ang, ang[0] + (ang_start[i] + 180) * direction)
        id_e = np.min([id_e, n_ang])
        img_t = np.array(f['img_tomo'][id_s:id_e])
        ang_t = ang[id_s:id_e]
        fsave = f'fly_scan_id_{sid:d}_sub_{i:d}.h5'
        print(f'angle: {ang_t[0]:4.2f} - {ang_t[-1]:4.2f}: saved to {fsave}')
        with h5py.File(fsave, 'w') as hf:
            hf.create_dataset('Pixel Size', data=pix)
            hf.create_dataset('scan_id', data=sid)
            hf.create_dataset('uid', data=uid)
            hf.create_dataset('img_tomo', data=img_t.astype(np.float32))
            hf.create_dataset('img_bkg', data=img_bkg.astype(np.float32))
            hf.create_dataset('img_bkg_avg', data=img_bkg_avg.astype(np.float32))
            hf.create_dataset('img_dark', data=img_dark.astype(np.float32))
            hf.create_dataset('img_dark_avg', data=img_dark_avg.astype(np.float32))
            hf.create_dataset('angle', data=ang_t.astype(np.float32))
            hf.create_dataset('X_eng', data=x_eng)
            
    f.close()
    del img_t
            
            
            
        


class IndexTracker(object):
    def __init__(self, ax, X, clim):
        self.ax = ax
        self._indx_txt = ax.set_title(" ", loc="center")
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices // 2
        if len(clim)==2:
            self.im = ax.imshow(self.X[self.ind, :, :], cmap="gray", clim=clim)
        else:
            self.im = ax.imshow(self.X[self.ind, :, :], cmap="gray")
        self.update()

    def onscroll(self, event):
        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        # self.ax.set_ylabel('slice %s' % self.ind)
        self._indx_txt.set_text(f"frame {self.ind + 1} of {self.slices}")
        self.im.axes.figure.canvas.draw()


def image_scrubber(data, clim, *, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
    else:
        fig = ax.figure
    tracker = IndexTracker(ax, data, clim)
    # monkey patch the tracker onto the figure to keep it alive
    fig._tracker = tracker
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)
    return tracker
