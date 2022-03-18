from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import h5py
import tomopy
import os
import tomopy
from scipy.signal import medfilt2d
from skimage import io
from PIL import Image
from scipy.interpolate import interp1d


def find_nearest(data, value):
    data = np.array(data)
    return np.abs(data - value).argmin()


def find_rot(fn, thresh=0.05, norm_flag=1):

    f = h5py.File(fn, "r")
    img_bkg = np.squeeze(np.array(f["img_bkg_avg"]))
    img_dark = np.squeeze(np.array(f["img_dark_avg"]))
    ang = np.array(list(f["angle"]))

    idx0 = np.argmin(np.abs(ang))
    idx180 = np.abs(ang - 180).argmin()
    img0 = np.array(list(f["img_tomo"][idx0]))
    img180_raw = np.array(list(f["img_tomo"][idx180]))
    f.close()
    s = np.squeeze(img0.shape)
    if norm_flag:
        img0 = (img0 - img_dark) / (img_bkg - img_dark)
        img180_raw = (img180_raw - img_dark) / (img_bkg - img_dark)
        img180 = img180_raw[:, ::-1]
        im1 = -np.log(img0)
        im2 = -np.log(img180)
    else:
        img180 = img180_raw[:, ::-1]
        im1 = img0.astype(np.float32)
        im2 = img180.astype(np.float32)
    im1[np.isnan(im1)] = 0
    im1[np.isinf(im1)] = 0
    im2[np.isnan(im2)] = 0
    im2[np.isinf(im2)] = 0
    im1[im1 < thresh] = 0
    im2[im2 < thresh] = 0
    im1 = im1[50:-50]
    im2 = im2[50:-50]
    im1 = medfilt2d(im1, 3)
    im2 = medfilt2d(im2, 3)
    im1_fft = np.fft.fft2(im1)
    im2_fft = np.fft.fft2(im2)
    results = dftregistration(im1_fft, im2_fft)
    row_shift = results[2]
    col_shift = results[3]
    rot_cen = s[1] / 2 + col_shift / 2 - 1
    return rot_cen


def rotcen_test2(
    fn,
    start=None,
    stop=None,
    steps=None,
    sli=0,
    block_list=[],
    return_flag=0,
    print_flag=1,
    bkg_level=0,
    txm_normed_flag=0,
    denoise_flag=0,
    fw_level=9,
    algorithm="gridrec",
    n_iter=5,
    circ_mask_ratio=0.95,
    options={},
    atten=None,
    clim=[],
    dark_scale=1,
    filter_name="None",
):
    import tomopy

    if not atten is None:
        ref_ang = atten[:, 0]
        ref_atten = atten[:, 1]
        fint = interp1d(ref_ang, ref_atten)

    f = h5py.File(fn, "r")
    tmp = np.array(f["img_tomo"][0])
    s = [1, tmp.shape[0], tmp.shape[1]]

    if denoise_flag:
        addition_slice = 100
    else:
        addition_slice = 0

    if sli == 0:
        sli = int(s[1] / 2)
    sli_exp = [
        np.max([0, sli - addition_slice // 2]),
        np.min([sli + addition_slice // 2 + 1, s[1]]),
    ]
    tomo_angle = np.array(f["angle"])
    theta = tomo_angle / 180.0 * np.pi
    img_tomo = np.array(f["img_tomo"][:, sli_exp[0] : sli_exp[1], :])

    if txm_normed_flag:
        prj_norm = img_tomo
    else:
        img_bkg = np.array(f["img_bkg_avg"][:, sli_exp[0] : sli_exp[1], :])
        img_dark = (
            np.array(f["img_dark_avg"][:, sli_exp[0] : sli_exp[1], :]) / dark_scale
        )
        prj = (img_tomo - img_dark) / (img_bkg - img_dark)
        if not atten is None:
            for i in range(len(tomo_angle)):
                att = fint(tomo_angle[i])
                prj[i] = prj[i] / att
        prj_norm = -np.log(prj)
    f.close()

    prj_norm = denoise(prj_norm, denoise_flag)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0

    prj_norm -= bkg_level

    prj_norm = tomopy.prep.stripe.remove_stripe_fw(
        prj_norm, level=fw_level, wname="db5", sigma=1, pad=True
    )
    """    
    if denoise_flag == 1: # denoise using wiener filter
        ss = prj_norm.shape
        for i in range(ss[0]):
           prj_norm[i] = skr.wiener(prj_norm[i], psf=psf, reg=reg, balance=balance, is_real=is_real, clip=clip)
    elif denoise_flag == 2:
        from skimage.filters import gaussian as gf
        prj_norm = gf(prj_norm, [0, 1, 1])
    """
    s = prj_norm.shape
    if len(s) == 2:
        prj_norm = prj_norm.reshape(s[0], 1, s[1])
        s = prj_norm.shape

    if theta[-1] > theta[1]:
        pos = find_nearest(theta, theta[0] + np.pi)
    else:
        pos = find_nearest(theta, theta[0] - np.pi)
    block_list = list(block_list) + list(np.arange(pos + 1, len(theta)))
    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]
    if start == None or stop == None or steps == None:
        start = int(s[2] / 2 - 50)
        stop = int(s[2] / 2 + 50)
        steps = 26
    cen = np.linspace(start, stop, steps)
    img = np.zeros([len(cen), s[2], s[2]])
    for i in range(len(cen)):
        if print_flag:
            print("{}: rotcen {}".format(i + 1, cen[i]))
            if algorithm == "gridrec":
                img[i] = tomopy.recon(
                    prj_norm[:, addition_slice : addition_slice + 1],
                    theta,
                    center=cen[i],
                    algorithm="gridrec",
                    filter_name=filter_name,
                )
            elif "astra" in algorithm:
                img[i] = tomopy.recon(
                    prj_norm[:, addition_slice : addition_slice + 1],
                    theta,
                    center=cen[i],
                    algorithm=tomopy.astra,
                    options=options,
                )
            else:
                img[i] = tomopy.recon(
                    prj_norm[:, addition_slice : addition_slice + 1],
                    theta,
                    center=cen[i],
                    algorithm=algorithm,
                    num_iter=n_iter,
                    filter_name=filter_name,
                )
    fout = "center_test.h5"
    with h5py.File(fout, "w") as hf:
        hf.create_dataset("img", data=img)
        hf.create_dataset("rot_cen", data=cen)
    img = tomopy.circ_mask(img, axis=0, ratio=circ_mask_ratio)
    tracker = image_scrubber(img, clim=clim)
    if return_flag:
        return img, cen


def rotcen_test(
    fn, start=None, stop=None, steps=None, sli=0, block_list=[], filter_name="none"
):

    f = h5py.File(fn)
    tmp = np.array(f["img_bkg_avg"])
    s = tmp.shape
    if sli == 0:
        sli = int(s[1] / 2)
    img_tomo = np.array(f["img_tomo"][:, sli, :])
    img_bkg = np.array(f["img_bkg_avg"][:, sli, :])
    img_dark = np.array(f["img_dark_avg"][:, sli, :])
    theta = np.array(f["angle"]) / 180.0 * np.pi
    f.close()
    prj = (img_tomo - img_dark) / (img_bkg - img_dark)
    prj_norm = -np.log(prj)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0
    s = prj_norm.shape
    prj_norm = prj_norm.reshape(s[0], 1, s[1])
    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]
    if start == None or stop == None or steps == None:
        start = int(s[1] / 2 - 50)
        stop = int(s[1] / 2 + 50)
        steps = 31
    cen = np.linspace(start, stop, steps)
    img = np.zeros([len(cen), s[1], s[1]])
    for i in range(len(cen)):
        print("{}: rotcen {}".format(i + 1, cen[i]))
        img[i] = tomopy.recon(
            prj_norm, theta, center=cen[i], algorithm="gridrec", filter_name=filter_name
        )
    fout = "center_test.h5"
    with h5py.File(fout, "w") as hf:
        hf.create_dataset("img", data=img)
        hf.create_dataset("rot_cen", data=cen)


def recon_sub(
    img,
    theta,
    rot_cen,
    block_list=[],
    rm_stripe=False,
    stripe_remove_level=9,
    algorithm="gridrec",
    num_iter=20,
    options={},
    filter_name="none",
):
    prj_norm = img
    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]
    if rm_stripe:
        prj_norm = tomopy.prep.stripe.remove_stripe_fw(
            prj_norm, level=stripe_remove_level, wname="db5", sigma=1, pad=True
        )
        # prj_norm = tomopy.prep.stripe.remove_all_stripe_tomo(prj_norm, 3, 81, 31)

    if algorithm == "gridrec":
        rec = tomopy.recon(
            prj_norm,
            theta,
            center=rot_cen,
            algorithm="gridrec",
            filter_name=filter_name,
        )
    elif "astra" in algorithm:
        rec = tomopy.recon(
            prj_norm, theta, center=rot_cen, algorithm=tomopy.astra, options=options
        )
    else:
        rec = tomopy.recon(
            prj_norm,
            theta,
            center=rot_cen,
            algorithm=algorithm,
            num_iter=num_iter,
            filter_name=filter_name,
        )

    return rec


def recon(
    fn,
    rot_cen,
    sli=[],
    col=[],
    binning=None,
    zero_flag=0,
    tiff_flag=0,
    block_list=[],
    rm_stripe=True,
    stripe_remove_level=9,
    filter_name="none",
):
    """
    reconstruct 3D tomography
    Inputs:
    --------
    fn: string
        filename of scan, e.g. 'fly_scan_0001.h5'
    rot_cen: float
        rotation center
    algorithm: string
        choose from 'gridrec' and 'mlem'
    sli: list
        a range of slice to recontruct, e.g. [100:300]
    num_iter: int
        iterations for 'mlem' algorithm
    bingning: int
        binning the reconstruted 3D tomographic image
    zero_flag: bool
        if 1: set negative pixel value to 0
        if 0: keep negative pixel value

    """
    import tomopy
    from PIL import Image

    f = h5py.File(fn, "r")
    tmp = np.array(f["img_bkg_avg"])
    s = tmp.shape
    slice_info = ""
    bin_info = ""
    col_info = ""

    if len(sli) == 0:
        sli = [0, s[1]]
    elif len(sli) == 1 and sli[0] >= 0 and sli[0] <= s[1]:
        sli = [sli[0], sli[0] + 1]
        slice_info = "_slice_{}_".format(sli[0])
    elif len(sli) == 2 and sli[0] >= 0 and sli[1] <= s[1]:
        slice_info = "_slice_{}_{}_".format(sli[0], sli[1])
    else:
        print("non valid slice id, will take reconstruction for the whole object")

    if len(col) == 0:
        col = [0, s[2]]
    elif len(col) == 1 and col[0] >= 0 and col[0] <= s[2]:
        col = [col[0], col[0] + 1]
        col_info = "_col_{}_".format(col[0])
    elif len(col) == 2 and col[0] >= 0 and col[1] <= s[2]:
        col_info = "col_{}_{}_".format(col[0], col[1])
    else:
        col = [0, s[2]]
        print("invalid col id, will take reconstruction for the whole object")

    rot_cen = rot_cen - col[0]

    scan_id = np.array(f["scan_id"])
    img_tomo = np.array(f["img_tomo"][:, sli[0] : sli[1], :])
    img_tomo = np.array(img_tomo[:, :, col[0] : col[1]])
    img_bkg = np.array(f["img_bkg_avg"][:, sli[0] : sli[1], col[0] : col[1]])
    img_dark = np.array(f["img_dark_avg"][:, sli[0] : sli[1], col[0] : col[1]])
    theta = np.array(f["angle"]) / 180.0 * np.pi
    if theta[-1] > theta[1]:
        pos_180 = find_nearest(theta, theta[0] + np.pi)
    else:
        pos_180 = find_nearest(theta, theta[0] - np.pi)
    block_list = list(block_list) + list(np.arange(pos_180 + 1, len(theta)))

    eng = np.array(f["X_eng"])
    f.close()

    s = img_tomo.shape
    if not binning == None:
        img_tomo = bin_ndarray(
            img_tomo, (s[0], int(s[1] / binning), int(s[2] / binning)), "sum"
        )
        img_bkg = bin_ndarray(
            img_bkg, (1, int(s[1] / binning), int(s[2] / binning)), "sum"
        )
        img_dark = bin_ndarray(
            img_dark, (1, int(s[1] / binning), int(s[2] / binning)), "sum"
        )
        rot_cen = rot_cen * 1.0 / binning
        bin_info = "bin{}".format(int(binning))

    prj = (img_tomo - img_dark) / (img_bkg - img_dark)
    prj_norm = -np.log(prj)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0

    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]

    if rm_stripe:
        prj_norm = tomopy.prep.stripe.remove_stripe_fw(
            prj_norm, level=stripe_remove_level, wname="db5", sigma=1, pad=True
        )
    fout = (
        "recon_scan_" + str(scan_id) + str(slice_info) + str(col_info) + str(bin_info)
    )

    if tiff_flag:
        cwd = os.getcwd()
        try:
            os.mkdir(cwd + f"/{fout}")
        except:
            print(cwd + f"/{fout} existed")
        for i in range(prj_norm.shape[1]):
            print(f"recon slice: {i:04d}/{prj_norm.shape[1]-1}")
            rec = tomopy.recon(
                prj_norm[:, i : i + 1, :], theta, center=rot_cen, algorithm="gridrec"
            )

            if zero_flag:
                rec[rec < 0] = 0
            fout_tif = cwd + f"/{fout}" + f"/{i+sli[0]:04d}.tiff"
            io.imsave(fout_tif, rec[0])
            # img = Image.fromarray(rec[0])
            # img.save(fout_tif)
    else:
        rec = tomopy.recon(
            prj_norm,
            theta,
            center=rot_cen,
            algorithm="gridrec",
            filter_name=filter_name,
        )
        if zero_flag:
            rec[rec < 0] = 0
        fout_h5 = fout + ".h5"
        with h5py.File(fout_h5, "w") as hf:
            hf.create_dataset("img", data=rec)
            hf.create_dataset("scan_id", data=scan_id)
            hf.create_dataset("X_eng", data=eng)
        print("{} is saved.".format(fout))
    del rec
    del img_tomo
    del prj_norm


"""
def batch_recon(file_path='.', file_prefix='fly', file_type='.h5', sli=[], col=[], block_list=[], binning=1, rm_stripe=True, stripe_remove_level=9):
    path = os.path.abspath(file_path)
    files = pyxas.retrieve_file_type(file_path, file_prefix, file_type)
    num_file = len(files)
    for i in range(num_file):
        fn = files[i].split('/')[-1]
        tmp = pyxas.get_img_from_hdf_file(fn, 'angle')
        angle = tmp['angle']
        pos_180 = pyxas.find_nearest(angle, angle[0]+180)
        block_list = list(block_list) + list(np.arange(pos_180+1, len(angle)))
        rotcen = pyxas.find_rot(fn)
        pyxas.recon(fn, rotcen, binning=binning, sli=sli, col=col, block_list=block_list, rm_stripe=rm_stripe, stripe_remove_level=stripe_remove_level)
"""


def batch_find_rotcen(files, block_list, index=0):
    img = []
    r = []
    for i in range(len(files)):
        fn = files[i]
        r.append(find_rot(fn))
        print(f"#{i} {fn}: rotcen = {r[-1]}")
        tmp = recon(
            fn,
            r[-1],
            sli=[index],
            block_list=block_list,
            binning=None,
            tiff_flag=0,
            h5_flag=0,
            return_flag=1,
        )
        img.append(np.squeeze(tmp))
    img = np.array(img, dtype=np.float32)
    with h5py.File("batch_rotcen.h5", "w") as hf:
        hf.create_dataset("img", data=img)
        hf.create_dataset("rotcen", data=r)
    return img, r


###########
def recon2(
    fn,
    rot_cen,
    sli=[],
    col=[],
    binning=None,
    algorithm="gridrec",
    zero_flag=0,
    block_list=[],
    bkg_level=0,
    txm_normed_flag=0,
    read_full_memory=0,
    denoise_flag=0,
    fw_level=9,
    num_iter=20,
    dark_scale=1,
    atten=[],
    norm_empty_sli=[],
    options={
        "proj_type": "cuda",
        "method": "SIRT_CUDA",
        "num_iter": 200,
    },
    filter_name="None",
    ncore=4,
):
    """
    reconstruct 3D tomography
    Inputs:
    --------
    fn: string
        filename of scan, e.g. 'fly_scan_0001.h5'
    rot_cen: float
        rotation center
    sli: list
        a range of slice to recontruct, e.g. [100:300]
    col:
        a range of column to reconstruct, e.g, [300,800]
    algorithm:
        if using astra, algorithm="astra", and will use "options" provided
    bingning: int
        binning the reconstruted 3D tomographic image
    zero_flag: bool
        if 1: set negative pixel value to 0
        if 0: keep negative pixel value
    block_list: list
        a list of index for the projections that will not be considered in reconstruction
    denoise_flag: int
        0: no denoising on projection image
        1: wiener denoising
        2: gaussian denoising
    """

    from PIL import Image

    f = h5py.File(fn, "r")
    tmp = np.array(f["img_tomo"][0])
    s = [1, tmp.shape[0], tmp.shape[1]]
    slice_info = ""
    bin_info = ""
    col_info = ""
    sli_step = 40
    if len(sli) == 0:
        sli = [0, s[1]]
        sli[1] = int((sli[1] - sli[0]) // sli_step * sli_step)
    elif len(sli) == 1 and sli[0] >= 0 and sli[0] <= s[1]:
        sli = [sli[0], sli[0] + 1]
        slice_info = "_slice_{}".format(sli[0])
    elif len(sli) == 2 and sli[0] >= 0 and sli[1] <= s[1]:
        sli[1] = int((sli[1] - sli[0]) // sli_step * sli_step) + sli[0]
        slice_info = "_slice_{}_{}".format(sli[0], sli[1])
    else:
        print("non valid slice id, will take reconstruction for the whole object")

    if len(col) == 0:
        col = [0, s[2]]
    elif len(col) == 1 and col[0] >= 0 and col[0] <= s[2]:
        col = [col[0], col[0] + 1]
        col_info = "_col_{}".format(col[0])
    elif len(col) == 2 and col[0] >= 0 and col[1] <= s[2]:
        col_info = "_col_{}_{}".format(col[0], col[1])
    else:
        col = [0, s[2]]
        print("invalid col id, will take reconstruction for the whole object")

    scan_id = np.array(f["scan_id"])
    theta = np.array(f["angle"]) / 180.0 * np.pi
    eng = np.array(f["X_eng"])

    if theta[-1] > theta[1]:
        pos = find_nearest(theta, theta[0] + np.pi)
    else:
        pos = find_nearest(theta, theta[0] - np.pi)
    block_list = list(block_list) + list(np.arange(pos + 1, len(theta)))
    allow_list = list(set(np.arange(len(theta))) - set(block_list))
    theta = theta[allow_list]
    tmp = np.squeeze(np.array(f["img_tomo"][0]))
    tmp = tmp[:, col[0] : col[1]]
    s = tmp.shape
    f.close()

    sli_total = np.arange(sli[0], sli[1])
    binning = binning if binning else 1
    bin_info = f"_bin_{binning}"

    n_steps = int(len(sli_total) / sli_step)
    rot_cen = (rot_cen * 1.0 - col[0]) / binning

    if n_steps == 0:
        read_full_memory = 1
    if read_full_memory:
        sli_step = sli[1] - sli[0]
        n_steps = 1

    # optional
    if denoise_flag:
        add_slice = min(sli_step // 2, 20)
    else:
        add_slice = 0

    try:
        rec = np.zeros(
            [sli_step * n_steps // binning, s[1] // binning, s[1] // binning],
            dtype=np.float32,
        )
    except:
        print("Cannot allocate memory")

    for i in range(n_steps):
        time_s = time.time()
        if i == 0:
            sli_sub = [sli_total[0], sli_total[0] + sli_step]
            current_sli = sli_sub
        elif i == n_steps - 1:
            sli_sub = [i * sli_step + sli_total[0], len(sli_total) + sli[0]]
            current_sli = sli_sub
        else:
            sli_sub = [i * sli_step + sli_total[0], (i + 1) * sli_step + sli_total[0]]
            current_sli = [sli_sub[0] - add_slice, sli_sub[1] + add_slice]
        print(f"recon {i+1}/{n_steps}:    sli = [{sli_sub[0]}, {sli_sub[1]}] ... ")

        prj_norm = proj_normalize(
            fn,
            current_sli,
            txm_normed_flag,
            binning,
            allow_list,
            bkg_level,
            fw_level=fw_level,
            denoise_flag=denoise_flag,
            dark_scale=dark_scale,
            atten=atten,
            norm_empty_sli=norm_empty_sli,
        )
        prj_norm = prj_norm[:, :, col[0] // binning : col[1] // binning]
        if i != 0 and i != n_steps - 1:
            prj_norm = prj_norm[
                :, add_slice // binning : sli_step // binning + add_slice // binning
            ]
        if algorithm == "gridrec":
            rec_sub = tomopy.recon(
                prj_norm,
                theta,
                center=rot_cen,
                algorithm="gridrec",
                ncore=ncore,
                filter_name=filter_name,
            )
        elif "astra" in algorithm:
            rec_sub = tomopy.recon(
                prj_norm,
                theta,
                center=rot_cen,
                algorithm=tomopy.astra,
                options=options,
                ncore=ncore,
            )
        else:
            rec_sub = tomopy.recon(
                prj_norm,
                theta,
                center=rot_cen,
                algorithm=algorithm,
                num_iter=num_iter,
                ncore=ncore,
                filter_name=filter_name,
            )
        rec[
            i * sli_step // binning : i * sli_step // binning + rec_sub.shape[0]
        ] = rec_sub
        time_e = time.time()
        print(f"takeing {time_e-time_s:3.1f} sec")
    bin_info = f"_bin{int(binning)}"
    fout = f"recon_scan_{str(scan_id)}{str(slice_info)}{str(col_info)}{str(bin_info)}"
    if zero_flag:
        rec[rec < 0] = 0
    fout_h5 = f"{fout}.h5"
    with h5py.File(fout_h5, "w") as hf:
        hf.create_dataset("img", data=np.array(rec, dtype=np.float32))
        hf.create_dataset("scan_id", data=scan_id)
        hf.create_dataset("X_eng", data=eng)
        hf.create_dataset("rot_cen", data=rot_cen)
        hf.create_dataset("binning", data=binning)
    print(f"{fout} is saved.")
    del rec
    del prj_norm


def denoise(prj, denoise_flag):
    if denoise_flag == 1:  # Wiener denoise
        import skimage.restoration as skr

        ss = prj.shape
        psf = np.ones([2, 2]) / (2**2)
        reg = None
        balance = 0.3
        is_real = True
        clip = True
        for j in range(ss[0]):
            prj[j] = skr.wiener(
                prj[j], psf=psf, reg=reg, balance=balance, is_real=is_real, clip=clip
            )
    elif denoise_flag == 2:  # Gaussian denoise
        from skimage.filters import gaussian as gf

        prj = gf(prj, [0, 1, 1])
    return prj


def proj_normalize(
    fn,
    sli,
    txm_normed_flag,
    binning,
    allow_list=[],
    bkg_level=0,
    fw_level=9,
    denoise_flag=0,
    dark_scale=1,
    atten=[],
    norm_empty_sli=[],
):
    f = h5py.File(fn, "r")
    img_tomo = np.array(f["img_tomo"][:, sli[0] : sli[1], :])
    tomo_angle = np.array(f["angle"])
    if len(norm_empty_sli) == 2:
        print("norm_empty_sli")
        t_tomo = np.array(f["img_tomo"][:, norm_empty_sli[0] : norm_empty_sli[1]])
        t_bkg = np.array(f["img_bkg_avg"][:, norm_empty_sli[0] : norm_empty_sli[1]])
        t_dark = (
            np.array(f["img_dark_avg"][:, norm_empty_sli[0] : norm_empty_sli[1]])
            / dark_scale
        )
        t = (t_tomo - t_dark) / (t_bkg - t_dark)
        t_mean = np.mean(t, axis=1)
        t_mean = np.expand_dims(t_mean, 1)
    else:
        t_mean = 1
    try:
        img_bkg = np.array(f["img_bkg_avg"][:, sli[0] : sli[1]])
    except:
        img_bkg = []
    try:
        img_dark = np.array(f["img_dark_avg"][:, sli[0] : sli[1]]) / dark_scale
    except:
        img_dark = []
    if len(img_dark) == 0 or len(img_bkg) == 0 or txm_normed_flag == 1:
        prj = img_tomo
    else:
        prj = (img_tomo - img_dark) / (img_bkg - img_dark)
        prj = prj / t_mean
        if len(atten):
            fint = interp1d(atten[:, 0], atten[:, 1])
            atten_interp = fint(tomo_angle)
            for i in range(len(tomo_angle)):
                prj[i] = prj[i] / atten_interp[i]

    prj = denoise(prj, denoise_flag)
    s = prj.shape

    prj = bin_ndarray(prj, (s[0], int(s[1] / binning), int(s[2] / binning)), "mean")
    if not txm_normed_flag:
        prj_norm = -np.log(prj)
        prj_norm[np.isnan(prj_norm)] = 0
        prj_norm[np.isinf(prj_norm)] = 0
    else:
        prj_norm = prj
    prj_norm[prj_norm < 0] = 0
    prj_norm = prj_norm[allow_list]
    prj_norm = tomopy.prep.stripe.remove_stripe_fw(
        prj_norm, level=fw_level, wname="db5", sigma=1, pad=True
    )
    prj_norm -= bkg_level
    f.close()
    del img_tomo
    del img_bkg
    del img_dark
    del prj
    return prj_norm


def show_image_slice(fn, sli=0):
    f = h5py.File(fn, "r")
    try:
        img = np.squeeze(np.array(f["img_tomo"][sli]))
        plt.figure()
        plt.imshow(img)
    except:
        try:
            img = np.squeeze(np.array(f["img_xanes"][sli]))
            plt.imshow(img)
        except:
            print("cannot display image")
    finally:
        f.close()


class IndexTracker(object):
    def __init__(self, ax, X, clim):
        self.ax = ax
        self._indx_txt = ax.set_title(" ", loc="center")
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices // 2
        if not len(clim):
            im_min = np.min(self.X[self.ind, :, :])
            im_max = np.max(self.X[self.ind, :, :])
        else:
            im_min, im_max = clim

        self.im = ax.imshow(self.X[self.ind, :, :], cmap="gray", clim=[im_min, im_max])
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


def image_scrubber(data, ax=None, clim=[]):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    tracker = IndexTracker(ax, data, clim)
    # monkey patch the tracker onto the figure to keep it alive
    fig._tracker = tracker
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)
    return tracker


def test():
    fn = "/home/mingyuan/Work/tomo_recon/data/fly_scan_id_66030.h5"
    options = {
        "proj_type": "cuda",
        "method": "FBP_CUDA",
        "num_iter": 80,
    }
    recon2(
        fn, 647, sli=[500], col=[300, 900], algorithm="astra", ncore=8, options=options
    )
