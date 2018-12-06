def find_rot(fn, thresh=0.05):
    f = h5py.File(fn, 'r')
    img_bkg = np.squeeze(np.array(f['img_bkg_avg']))
    ang = np.array(list(f['angle']))
    
    tmp = np.abs(ang - ang[0] -180).argmin() 
    img0 = np.array(list(f['img_tomo'][0]))
    img180_raw = np.array(list(f['img_tomo'][tmp]))
    f.close()
#    img0, img180_raw = img[0], img[tmp]
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
#    C = np.abs(np.fft.ifft2(im1_fft * np.conj(im2_fft))) 
    results = dftregistration(im1_fft, im2_fft)
    row_shift = results[2]
    col_shift = results[3]
    rot_cen = s[1]/2 + col_shift/2 - 1 
    return rot_cen


def rotcen_test(fn, start=None, stop=None, steps=None, sli=0):
   
   
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
    if start==None or stop==None or steps==None:
        start = int(s[1]/2-50)
        stop = int(s[1]/2+50)
        steps = 31
    cen = np.linspace(start, stop, steps)          
    img = np.zeros([len(cen), s[1], s[1]])
    for i in range(len(cen)):
        print('{}: rotcen {}'.format(i+1, cen[i]))
        img[i] = tomopy.recon(prj_norm, theta, center=cen[i], algorithm='gridrec')    
    fout = 'center_test.h5'
    with h5py.File(fout, 'w') as hf:
        hf.create_dataset('img', data=img)
        hf.create_dataset('rot_cen', data=cen)
    
    
def recon(fn, rot_cen, sli=[], col=[], binning=None, zero_flag=0, tiff_flag=0):
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
        slice_info = '_slice_{}_'.format(sli[0])
    elif len(sli) == 2 and sli[0] >=0 and sli[1] <= s[1]:
        slice_info = '_slice_{}_{}_'.format(sli[0], sli[1])
    else:
        print('non valid slice id, will take reconstruction for the whole object')
    
    if len(col) == 0:
        col = [0, s[2]]
    elif len(col) == 1 and col[0] >=0 and col[0] <= s[2]:
        col = [col[0], col[0]+1]
        col_info = '_col_{}_'.format(col[0])
    elif len(col) == 2 and col[0] >=0 and col[1] <= s[2]:
        col_info = 'col_{}_{}_'.format(col[0], col[1])
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
    f.close() 

    s = img_tomo.shape
    if not binning == None:
        img_tomo = bin_ndarray(img_tomo, (s[0], int(s[1]/binning), int(s[2]/binning)), 'sum')
        img_bkg = bin_ndarray(img_bkg, (1, int(s[1]/binning), int(s[2]/binning)), 'sum')
        img_dark = bin_ndarray(img_dark, (1, int(s[1]/binning), int(s[2]/binning)), 'sum')
        rot_cen = rot_cen * 1.0 / binning 
        bin_info = 'bin{}'.format(int(binning))

    prj = (img_tomo - img_dark) / (img_bkg - img_dark)
    prj_norm = -np.log(prj)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0   

    fout = 'recon_scan_' + str(scan_id) + str(slice_info) + str(col_info) + str(bin_info)
    '''
    if algorithm == 'gridrec':
        rec = tomopy.recon(prj_norm, theta, center=rot_cen, algorithm='gridrec')
    elif algorithm == 'mlem' or algorithm == 'ospml_hybrid':
        rec = tomopy.recon(prj_norm, theta, center=rot_cen, algorithm=algorithm, num_iter=num_iter)
    else:
        print('algorithm not recognized')
    rec = tomopy.misc.corr.remove_ring(rec, rwidth=3)
    '''    
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
        fout_h5 = fout +'.h5'
        with h5py.File(fout_h5, 'w') as hf:
            hf.create_dataset('img', data=rec)
            hf.create_dataset('scan_id', data=scan_id)        
        print('{} is saved.'.format(fout)) 
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
            img = np.squeeze(np.array(f['img_xames'][sli]))
            plt.imshow(img)
        except:
            print('cannot display image')
    finally:
        f.close()

def find_3Dshift(f_ref, f_need_align, threshold=0.2):
    '''
    f1 is the reference
    f2 is the file need to shift
    '''
    f = h5py.File(f_ref, 'r')
    img1 = np.squeeze(np.array(f['img_tomo'][0]))
    bkg1 = np.squeeze(np.array(f['img_bkg_avg']))
    f.close()

    f = h5py.File(f_need_align, 'r')
    img2 = np.squeeze(np.array(f['img_tomo'][0]))
    bkg2 = np.squeeze(f['img_bkg_avg'])
    f.close()

    img_norm1 = -np.log(img1 / bkg1)
    img_norm2 = -np.log(img2 / bkg2)

    mask = np.ones(img_norm1.shape)
    mask[img_norm1<threshold] = 0
    mask = medfilt2d(mask, 7)
#    mask[:, 1600:] = 0

    img_norm1_shift, rshift, cshift = align_img(mask, img_norm2)

    return rshift, cshift



def recon_with_align(fn, rot_cen, rshift, cshift, sli=[], col=[], binning=None, tiff_flag=0):
    import tomopy
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
        slice_info = '_slice_{}_'.format(sli[0])
    elif len(sli) == 2 and sli[0] >=0 and sli[1] <= s[1]:
        slice_info = '_slice_{}_{}_'.format(sli[0], sli[1])
    else:
        sli = [0, s[1]]
        print('invalid slice id, will take reconstruction for the whole object')

    if len(col) == 0:
        col = [0, s[2]]
    elif len(col) == 1 and col[0] >=0 and col[0] <= s[2]:
        col = [col[0], col[0]+1]
        col_info = 'col_{}_'.format(col[0])
    elif len(col) == 2 and col[0] >=0 and col[1] <= s[2]:
        col_info = 'col_{}_{}_'.format(col[0], col[1])
    else:
        col = [0, s[2]]
        print('invalid col id, will take reconstruction for the whole object')
    rot_cen = rot_cen - col[0]


    scan_id = np.array(f['scan_id'])
    img_tomo = np.array(f['img_tomo'][:, sli[0]:sli[1], :])
    img_tomo = img_tomo[:, :, col[0]:col[1]]
    img_bkg = np.array(f['img_bkg_avg'][:, sli[0]:sli[1], col[0]:col[1]])
    img_dark = np.array(f['img_dark_avg'][:, sli[0]:sli[1], col[0]:col[1]])
    theta = np.array(f['angle']) / 180.0 * np.pi
    f.close() 
    
    s = img_tomo.shape
    if not binning == None:
        img_tomo = bin_ndarray(img_tomo, (s[0], int(s[1]/binning), int(s[2]/binning)), 'sum')
        img_bkg = bin_ndarray(img_bkg, (1, int(s[1]/binning), int(s[2]/binning)), 'sum')
        img_dark = bin_ndarray(img_dark, (1, int(s[1]/binning), int(s[2]/binning)), 'sum')
        rot_cen = rot_cen * 1.0 / binning 
        rshift = rshift * 1.0 / binning
        bin_info = 'bin{}'.format(int(binning))

    prj = (img_tomo - img_dark) / (img_bkg - img_dark)
    prj_norm = -np.log(prj)
    prj_norm[np.isnan(prj_norm)] = 0
    prj_norm[np.isinf(prj_norm)] = 0
    prj_norm[prj_norm < 0] = 0   

    
    print('shift imaging ...')
    prj_norm = shift(prj_norm, [0, rshift, 0], mode='constant', cval=0, order=0)     

    fout = 'recon_scan_' + str(scan_id) + str(slice_info) + str(col_info) + str(bin_info)
    if tiff_flag:
        cwd = os.getcwd()
        try:
            os.mkdir(cwd+f'/{fout}')
        except:
            print(cwd+f'/{fout} existed')
        for i in range(prj_norm.shape[1]):
            print(f'recon slice: {i:04d}/{prj_norm.shape[1]}')
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
        fout_h5 = fout +'.h5'
        with h5py.File(fout_h5, 'w') as hf:
            hf.create_dataset('img', data=rec)
            hf.create_dataset('scan_id', data=scan_id)        
        print('{} is saved.'.format(fout)) 






def align3D(f_ref, f_ali, tag_ref='recon', tag_ali='recon'):
    '''
    align two sets of 3D reconstruction data with same image size

    Inputs:
    --------
    f_ref: file name of 1st tomo-reconstruction, use as reference
    
    f_ali: file name of 2nd tomo-reconstruction, need to be aligned

    tag_ref: tag of 3D data in .h5 file for 1st 3D data

    tag_ali: tag of 3D data in .h5 file for 2nd 3D data

    Outputs:
    --------
    3D array of aligned data
    
    '''
    f = h5py.File(f_ref, 'r')
    img_ref = np.array(f[tag_ref])
    f.close()
    f = h5py.File(f_ali)
    img = np.array(f[tag_ali])  
    f.close()
    img_ali = deepcopy(img)
    s = img_ref.shape

 #   img[img < 0.01 * np.max(img)] = 0
 #   img_ali[img_ali < 0.01 * np.max(img_ali)] = 0

    img_ref_prj1 = np.sum(img_ref, axis=1) # project to front of cube
    img_prj1 = np.sum(img, axis=1)
    _, r1, c1 = align_img(img_ref_prj1, img_prj1)

    img_ref_prj2 = np.sum(img_ref, axis=2) # project to right of cube
    img_prj2 = np.sum(img, axis=2)
    _, r2, c2 = align_img(img_ref_prj2, img_prj2)
    
    r = (r1 + r2) / 2
    print('1st-dimension shift: {}\n2nd-dimension shift: {}\n3rd-dimension shift: {}'.format(r, c2, c1))
    print('aligning 3d stack ...')   
    for i in range(s[1]):
#        if not i%20: print('{}'.format(i))
        temp = np.squeeze(img[:,i,:])
        temp = shift(temp, [r1, c1], mode='constant', cval=0)
        img_ali[:, i, :] = temp

    for i in range(s[2]):
#        if not i%20: print('{}'.format(i))
        temp = np.squeeze(img_ali[:,:, i])
        temp = shift(temp, [0, c1], mode='constant', cval=0)
        img_ali[:, :, i] = temp
    return img_ali
    

def dif3D(f_ref, f_ali, tag_ref='recon', tag_ali='recon', output_name='dif3D.h5'):
    '''
    calculate the difference of two sets of 3D tomo-reconstruction

    Inputs:
    --------
    f_ref: file name of 1st tomo-reconstruction, use as reference
    
    f_ali: file name of 2nd tomo-reconstruction, need to be aligned

    tag_ref: tag of 3D data in .h5 file for 1st 3D data

    tag_ali: tag of 3D data in .h5 file for 2nd 3D data

    output_name: filename of output

    --------
    '''
    
    img_ali = align3D(f_ref, f_ali, tag_ref, tag_ali)
    f = h5py.File(f_ref, 'r')
    img_ref = np.array(f[tag_ref])
    f.close()
    img_dif = img_ali - img_ref
    
    with h5py.File(output_name, 'w') as hf:
        hf.create_dataset('dif3D', data = img_dif)
        hf.create_dataset('img_ref', data = f_ref)
        hf.create_dataset('img_ali', data = f_ali)

    print('\'{} \' saved.'. format(output_name))



def batch_recon(scan_list=[], sli=[], col=[], binning=2):
    summary = {}
    summary_scan_id = []
    summary_rotcen = []
    summary_binning = []
    for scan_id in scan_list:
        h = db[int(scan_id)]
        scan_id = h.start['scan_id']
        if h.start['plan_name'] == 'fly_scan':
            try:
                load_scan([scan_id])
            except:
                print('loading fails...')
        
            try:
                fn = f'fly_scan_id_{scan_id}.h5'
                r = find_rot(fn)
                print(f'scan #{scan_id}: rot_cen = {r:3.2f}')
                recon(fn, rot_cen=r, sli=sli, col=[], binning=binning)
                summary_scan_id.append(scan_id)
                summary_rotcen.append(r)
                summary_binning.append(binning)
            except:
                print(f'scan # {scan_id}: recon/writting fails ... ')
        else:
            print('skipping non-fly-scan #{scan_id}')
    summary['scan_id'] = summary_scan_id
    summary['rotcen'] = summary_rotcen
    summary['binning'] = summary_binning

    df = pd.DataFrame(summary)
    df.to_csv(f'recon_summary_{summary_scan_id[0]}_to_{summary_scan_id[-1]}.txt',sep='\t')
    print('a summary file have been saved to current directory')

















