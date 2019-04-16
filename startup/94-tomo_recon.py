import numpy as np

def find_nearest(data, value):
    data = np.array(data)
    return np.abs(data - value).argmin()


def find_rot(fn, thresh=0.05):
    from pystackreg import StackReg
    sr = StackReg(StackReg.TRANSLATION) 
    f = h5py.File(fn, 'r')
    img_bkg = np.squeeze(np.array(f['img_bkg_avg']))
    ang = np.array(list(f['angle']))
    tmp = np.abs(ang - ang[0] -180).argmin() 
    img0 = np.array(list(f['img_tomo'][0]))
    img180_raw = np.array(list(f['img_tomo'][tmp]))
    f.close()
    img0 = img0 / img_bkg
    img180_raw = img180_raw / img_bkg
    img180 = img180_raw[:,::-1] 
    s = np.squeeze(img0.shape)
    im1 = -np.log(img0)
    im2 = -np.log(img180)
    im1[np.isnan(im1)] = 0
    im2[np.isnan(im2)] = 0
    im1[im1 < thresh] = 0
    im2[im2 < thresh] = 0
    im1 = medfilt2d(im1,5)
    im2 = medfilt2d(im2, 5)
    im1_fft = np.fft.fft2(im1)
    im2_fft = np.fft.fft2(im2)
    results = dftregistration(im1_fft, im2_fft)
    row_shift = results[2]
    col_shift = results[3]
    rot_cen = s[1]/2 + col_shift/2 - 1 

    tmat = sr.register(im1, im2) 
    rshft = -tmat[1, 2]
    cshft = -tmat[0, 2]
    rot_cen0 = s[1]/2 + cshft/2 - 1

    print(f'rot_cen = {rot_cen} or {rot_cen0}')
    return rot_cen


def rotcen_test(fn, start=None, stop=None, steps=None, sli=0, block_list=[], return_flag=0, print_flag=1):  
    import tomopy 
    f = h5py.File(fn)
    tmp = np.array(f['img_bkg_avg'])
    s = tmp.shape
    if sli == 0: sli = int(s[1]/2)
    img_tomo = np.array(f['img_tomo'][:, sli, :])
    img_bkg = np.array(f['img_bkg_avg'][:, sli, :])
    img_dark = np.array(f['img_dark_avg'][:, sli, :])
    theta = np.array(f['angle']) / 180.0 * np.pi
    f.close()
    prj = (img_tomo - img_dark) / (img_bkg - img_dark)
    prj_norm = -np.log(prj)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0    
    s = prj_norm.shape  
    prj_norm = prj_norm.reshape(s[0], 1, s[1])
    prj_norm = tomopy.prep.stripe.remove_stripe_fw(prj_norm,level=9, wname='db5', sigma=1, pad=True)
    pos = find_nearest(theta, theta[0]+np.pi)
    block_list = list(block_list) + list(np.arange(pos+1, len(theta)))
    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]
    if start==None or stop==None or steps==None:
        start = int(s[1]/2-50)
        stop = int(s[1]/2+50)
        steps = 26
    cen = np.linspace(start, stop, steps)          
    img = np.zeros([len(cen), s[1], s[1]])
    for i in range(len(cen)):
        if print_flag:
            print('{}: rotcen {}'.format(i+1, cen[i]))
        img[i] = tomopy.recon(prj_norm, theta, center=cen[i], algorithm='gridrec')    
    fout = 'center_test.h5'
    with h5py.File(fout, 'w') as hf:
        hf.create_dataset('img', data=img)
        hf.create_dataset('rot_cen', data=cen)
    img = tomopy.circ_mask(img, axis=0, ratio=0.6)
    tracker = image_scrubber(img)
    if return_flag:
        return img, cen


def img_variance(img):
    import tomopy
    s = img.shape
    variance = np.zeros(s[0])
    img = tomopy.circ_mask(img, axis=0, ratio=0.6)
    for i in range(s[0]):
        img[i] = medfilt2d(img[i], 5)
        img_ = img[i].flatten()
        t = img_>0
        img_ = img_[t]
        t = np.mean(img_)
        variance[i] = np.sqrt(np.sum(np.power(np.abs(img_ - t), 2))/len(img_-1))
    return variance


def recon(fn, rot_cen, sli=[], col=[], binning=None, zero_flag=0, tiff_flag=0, block_list=[]):
    '''
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
        
    '''
    import tomopy
    from PIL import Image
    f = h5py.File(fn)
    tmp = np.array(f['img_bkg_avg'])
    s = tmp.shape
    slice_info = ''
    bin_info = ''
    col_info = ''

    if len(sli) == 0:
        sli = [0, s[1]]
    elif len(sli) == 1 and sli[0] >=0 and sli[0] <= s[1]:
        sli = [sli[0], sli[0]+1]
        slice_info = '_slice_{}'.format(sli[0])
    elif len(sli) == 2 and sli[0] >=0 and sli[1] <= s[1]:
        slice_info = '_slice_{}_{}'.format(sli[0], sli[1])
    else:
        print('non valid slice id, will take reconstruction for the whole object')    
    if len(col) == 0:
        col = [0, s[2]]
    elif len(col) == 1 and col[0] >=0 and col[0] <= s[2]:
        col = [col[0], col[0]+1]
        col_info = '_col_{}'.format(col[0])
    elif len(col) == 2 and col[0] >=0 and col[1] <= s[2]:
        col_info = '_col_{}_{}'.format(col[0], col[1])
    else:
        col = [0, s[2]]
        print('invalid col id, will take reconstruction for the whole object')

    rot_cen = rot_cen - col[0]    
    scan_id = np.array(f['scan_id'])
    img_tomo = np.array(f['img_tomo'][:, sli[0]:sli[1], :])
    img_tomo = np.array(img_tomo[:, :, col[0]:col[1]])
    img_bkg = np.array(f['img_bkg_avg'][:, sli[0]:sli[1], col[0]:col[1]])
    img_dark = np.array(f['img_dark_avg'][:, sli[0]:sli[1], col[0]:col[1]])
    theta = np.array(f['angle']) / 180.0 * np.pi
    eng = np.array(f['X_eng'])
    f.close() 
    s = img_tomo.shape
    if not binning == None:
        img_tomo = bin_ndarray(img_tomo, (s[0], int(s[1]/binning), int(s[2]/binning)), 'sum')
        img_bkg = bin_ndarray(img_bkg, (1, int(s[1]/binning), int(s[2]/binning)), 'sum')
        img_dark = bin_ndarray(img_dark, (1, int(s[1]/binning), int(s[2]/binning)), 'sum')
        rot_cen = rot_cen * 1.0 / binning 
        bin_info = f'_bin{int(binning)}'
    prj = (img_tomo - img_dark) / (img_bkg - img_dark)
    prj_norm = -np.log(prj)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0   

    pos = find_nearest(theta, theta[0]+np.pi)
    block_list = list(block_list) + list(np.arange(pos+1, len(theta)))
    if len(block_list):
        allow_list = list(set(np.arange(len(prj_norm))) - set(block_list))
        prj_norm = prj_norm[allow_list]
        theta = theta[allow_list]

    prj_norm = tomopy.prep.stripe.remove_stripe_fw(prj_norm,level=9, wname='db5', sigma=1, pad=True)
    fout = f'recon_scan_{str(scan_id)}{str(slice_info)}{str(col_info)}{str(bin_info)}'
  
    if tiff_flag:
        cwd = os.getcwd()
        try:
            os.mkdir(cwd+f'/{fout}')
        except:
            print(cwd+f'/{fout} existed')
        for i in range(prj_norm.shape[1]):
            print(f'recon slice: {i:04d}/{prj_norm.shape[1]-1}')
            rec = tomopy.recon(prj_norm[:, i:i+1,:], theta, center=rot_cen, algorithm='gridrec')
            
            if zero_flag:
                rec[rec<0] = 0
            img = Image.fromarray(rec[0])
            fout_tif = cwd + f'/{fout}' + f'/{i+sli[0]:04d}.tiff' 
            img.save(fout_tif)
    else:
        rec = tomopy.recon(prj_norm, theta, center=rot_cen, algorithm='gridrec')
        if zero_flag:
            rec[rec<0] = 0
        fout_h5 = f'{fout}.h5'
        with h5py.File(fout_h5, 'w') as hf:
            hf.create_dataset('img', data=rec)
            hf.create_dataset('scan_id', data=scan_id)        
            hf.create_dataset('X_eng', data=eng)
        print(f'{fout} is saved.') 
    del rec
    del img_tomo
    del prj_norm



def show_image_slice(fn, sli=0):
    f=h5py.File(fn,'r')
    try:
        img = np.squeeze(np.array(f['img_tomo'][sli]))
        plt.figure()
        plt.imshow(img)
    except:
        try:
            img = np.squeeze(np.array(f['img_xanes'][sli]))
            plt.imshow(img)
        except:
            print('cannot display image')
    finally:
        f.close()

















