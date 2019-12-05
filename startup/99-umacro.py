
new_user()
show_global_para()
run_pdf()
#read_calib_file()
read_calib_file_new()
    
###################################

def load_xanes_ref(*arg):
    '''
    load reference spectrum, use it as:    ref = load_xanes_ref(Ni, Ni2, Ni3)
    each spectrum is two-column array, containing: energy(1st column) and absortion(2nd column)

    It returns a dictionary, which can be used as: spectrum_ref['ref0'], spectrum_ref['ref1'] ....
    '''

    num_ref = len(arg)
    assert num_ref >1, "num of reference should larger than 1"
    spectrum_ref = {}    
    for i in range(num_ref):
        spectrum_ref[f'ref{i}'] = arg[i]  
    return spectrum_ref


def fit_2D_xanes_non_iter(img_xanes, eng, spectrum_ref, error_thresh=0.1):
    '''
    Solve equation of Ax=b, where:

    Inputs:
    ----------
    A: reference spectrum (2-colume array: xray_energy vs. absorption_spectrum)
    X: fitted coefficient of each ref spectrum
    b: experimental 2D XANES data 

    Outputs:
    ----------
    fit_coef: the 'x' in the equation 'Ax=b': fitted coefficient of each ref spectrum
    cost: cost between fitted spectrum and raw data
    '''

    num_ref = len(spectrum_ref)
    spec_interp = {}
    comp = {}   
    A = [] 
    s = img_xanes.shape
    for i in range(num_ref):
        tmp = interp1d(spectrum_ref[f'ref{i}'][:,0], spectrum_ref[f'ref{i}'][:,1], kind='cubic')
        A.append(tmp(eng).reshape(1, len(eng)))
        spec_interp[f'ref{i}'] = tmp(eng).reshape(1, len(eng))
        comp[f'A{i}'] = spec_interp[f'ref{i}'].reshape(len(eng), 1)
        comp[f'A{i}_t'] = comp[f'A{i}'].T
    # e.g., spectrum_ref contains: ref1, ref2, ref3
    # e.g., comp contains: A1, A2, A3, A1_t, A2_t, A3_t
    #       A1 = ref1.reshape(110, 1)
    #       A1_t = A1.T
    A = np.squeeze(A).T
    M = np.zeros([num_ref+1, num_ref+1])
    for i in range(num_ref):
        for j in range(num_ref):
            M[i,j] = np.dot(comp[f'A{i}_t'], comp[f'A{j}'])
        M[i, num_ref] = 1
    M[num_ref] = np.ones((1, num_ref+1))
    M[num_ref, -1] = 0
    # e.g.
    # M = np.array([[float(np.dot(A1_t, A1)), float(np.dot(A1_t, A2)), float(np.dot(A1_t, A3)), 1.],
    #                [float(np.dot(A2_t, A1)), float(np.dot(A2_t, A2)), float(np.dot(A2_t, A3)), 1.],
    #                [float(np.dot(A3_t, A1)), float(np.dot(A3_t, A2)), float(np.dot(A3_t, A3)), 1.],
    #                [1., 1., 1., 0.]])
    M_inv = np.linalg.inv(M)
    
    b_tot = img_xanes.reshape(s[0],-1)
    B = np.ones([num_ref+1, b_tot.shape[1]])
    for i in range(num_ref):
        B[i] = np.dot(comp[f'A{i}_t'], b_tot)
    x = np.dot(M_inv, B)
    x = x[:-1]    
    x[x<0] = 0
    x_sum = np.sum(x, axis=0, keepdims=True)
    x = x / x_sum  

    cost = np.sum((np.dot(A, x) - b_tot)**2, axis=0)/s[0]
    cost = cost.reshape(s[1], s[2])

    x = x.reshape(num_ref, s[1], s[2])  
    # cost = compute_xanes_fit_cost(img_xanes, x, spec_interp)
    
    mask = compute_xanes_fit_mask(cost, error_thresh)
    mask = mask.reshape(s[1], s[2])
    mask_tile = np.tile(mask, (x.shape[0], 1, 1))

    x = x * mask_tile
    cost = cost * mask
    return x, cost



def fit_2D_xanes_iter(img_xanes, eng, spectrum_ref, coef0=None, learning_rate=0.005, n_iter=10, bounds=[0,1], error_thresh=0.1):
    '''
    Solve the equation A*x = b iteratively


    Inputs:
    -------
    img_xanes: 3D xanes image stack

    eng: energy list of xanes

    spectrum_ref: dictionary, obtained from, e.g. spectrum_ref = load_xanes_ref(Ni2, Ni3)

    coef0: initial guess of the fitted coefficient, 
           it has dimention of [num_of_referece, img_xanes.shape[1], img_xanes.shape[2]]

    learning_rate: float

    n_iter: int

    bounds: [low_limit, high_limit]
          can be 'None', which give no boundary limit

    error_thresh: float
          used to generate a mask, mask[fitting_cost > error_thresh] = 0

    Outputs:
    ---------
    w: fitted 2D_xanes coefficient
       it has dimention of [num_of_referece, img_xanes.shape[1], img_xanes.shape[2]]

    cost: 2D fitting cost
    '''

    num_ref = len(spectrum_ref)
    A = [] 
    for i in range(num_ref):
        tmp = interp1d(spectrum_ref[f'ref{i}'][:,0], spectrum_ref[f'ref{i}'][:,1], kind='cubic')
        A.append(tmp(eng).reshape(1, len(eng)))
    A = np.squeeze(A).T
    Y = img_xanes.reshape(img_xanes.shape[0], -1)
    if not coef0 is None:
        W = coef0.reshape(coef0.shape[0], -1)
    w, cost = lsq_fit_iter2(A, Y, W, learning_rate, n_iter, bounds, print_flag=1)
    w = w.reshape(len(w), img_xanes.shape[1], img_xanes.shape[2])
    cost = cost.reshape(cost.shape[0], img_xanes.shape[1], img_xanes.shape[2])
    mask = compute_xanes_fit_mask(cost[-1], error_thresh)
    mask_tile = np.tile(mask, (w.shape[0], 1, 1))
    w = w * mask_tile
    mask_tile2 = np.tile(mask, (cost.shape[0], 1, 1))
    cost = cost * mask_tile2
    return w, cost


def compute_xanes_fit_cost(img_xanes, fit_coef, spec_interp):
    # compute the cost
    num_ref = len(spec_interp)
    y_fit = np.zeros(img_xanes.shape)
    for i in range(img_xanes.shape[0]):
        for j in range(num_ref):
            y_fit[i] = y_fit[i] + fit_coef[j]*np.squeeze(spec_interp[f'ref{j}'])[i] 
    y_dif = np.power(y_fit - img_xanes, 2)
    cost = np.sum(y_dif, axis=0) / img_xanes.shape[0]
    return cost


def compute_xanes_fit_mask(cost, error_thresh=0.1):
    mask = np.ones(cost.shape)
    mask[cost > error_thresh] = 0
    return mask


def xanes_fit_demo():
    f = h5py.File('img_xanes_normed.h5', 'r')
    img_xanes = np.array(f['img'])
    eng = np.array(f['X_eng'])
    f.close()
    img_xanes= bin_ndarray(img_xanes, (img_xanes.shape[0], int(img_xanes.shape[1]/2), int(img_xanes.shape[2]/2)))

    Ni = np.loadtxt('/NSLS2/xf18id1/users/2018Q1/MING_Proposal_000/xanes_ref/Ni_xanes_norm.txt')
    Ni2 = np.loadtxt('/NSLS2/xf18id1/users/2018Q1/MING_Proposal_000/xanes_ref/NiO_xanes_norm.txt')
    Ni3 = np.loadtxt('/NSLS2/xf18id1/users/2018Q1/MING_Proposal_000/xanes_ref/LiNiO2_xanes_norm.txt')

    spectrum_ref = load_xanes_ref(Ni2, Ni3)
    w1, c1 = fit_2D_xanes_non_iter(img_xanes, eng, spectrum_ref, error_thresh=0.1)
    plt.figure()
    plt.subplot(121); plt.imshow(w1[0])
    plt.subplot(122); plt.imshow(w1[1])    



def temp():
    #os.mkdir('recon_image')
    scan_id = np.arange(15198, 15256)
    n = len(scan_id)
    for i in range(n):
        fn = f'fly_scan_id_{int(scan_id[i])}.h5'
        print(f'reconstructing: {fn} ... ')
        img = get_img(db[int(scan_id[i])], sli=[0,1])
        s = img.shape
        if s[-1] > 2000:
            sli = [200, 1900]
            binning = 2
        else:
            sli = [100, 950]
            binning = 1
        rot_cen = find_rot(fn)
        recon(fn, rot_cen, sli=sli, binning=binning)
        try:
            f_recon = f'recon_scan_{int(scan_id[i])}_sli_{sli[0]}_{sli[1]}_bin{binning}.h5'
            f = h5py.File(f_recon, 'r')
            sli_choose = int((sli[0]+sli[1])/2)
            img_recon = np.array(f['img'][sli_choose], dtype=np.float32)
            sid = scan_id[i]
            f.close()
            fn_img_save = f'recon_image/recon_{int(sid)}_sli_{sli_choose}.tiff'
            print(f'saving {fn_img_save}\n')
            io.imsave(fn_img_save, img_recon)
        except:
            pass


def multipos_tomo(exposure_time, x_list, y_list, z_list, out_x, out_y, out_z, out_r, rs, relative_rot_angle = 185, period=0.05, relative_move_flag=0, traditional_sequence_flag=1, repeat=1, sleep_time=0, note=''):
    n = len(x_list)
    txt = f'starting multiposition_flyscan: (repeating for {repeat} times)'
    insert_text(txt)
    for rep in range(repeat):
        for i in range(n):
            txt = f'\n################\nrepeat #{rep+1}:\nmoving to the {i+1} position: x={x_list[i]}, y={y_list[i]}, z={z_list[i]}'
            print(txt)
            insert_text(txt)
            yield from mv(zps.sx, x_list[i], zps.sy, y_list[i], zps.sz, z_list[i])
            yield from fly_scan(exposure_time=exposure_time, relative_rot_angle=relative_rot_angle, period=period, chunk_size=20, out_x=out_x, out_y=out_y, out_z=out_z,  out_r=out_r, rs=rs,
 simu=False, relative_move_flag=relative_move_flag, traditional_sequence_flag=traditional_sequence_flag, note=note, md=None)
        print(f'sleeping for {sleep_time:3.1f} s')
        yield from bps.sleep(sleep_time)


def create_lists(x0, y0, z0, dx, dy, dz, Nx, Ny, Nz):
    Ntotal=Nx*Ny*Nz
    x_list=np.zeros(Ntotal)
    y_list=np.zeros(Ntotal)
    z_list=np.zeros(Ntotal)

    j = 0
    for iz in range(Nz):
        for ix in range(Nx):
            for iy in range(Ny):
                j = iy + ix*Ny + iz * Ny * Nx #!!!
                y_list[j] = y0 + dy * iy
                x_list[j] = x0 + dx * ix
                z_list[j] = z0 + dz * iz

    return x_list, y_list, z_list


def fan_scan(eng_list, x_list_2d, y_list_2d, z_list_2d, r_list_2d, x_list_3d, y_list_3d, z_list_3d, r_list_3d, out_x, out_y, out_z, out_r, relative_rot_angle, rs=3, exposure_time=0.05, chunk_size=4, sleep_time=0, repeat=1, relative_move_flag=True,  note=''):
    export_pdf(1)    
    insert_text('start multiposition 2D xanes and 3D xanes')
    for i in range(repeat):
        print(f'\nrepeat # {i+1}')
        #print(f'start xanes 2D scan:')
        #yield from multipos_2D_xanes_scan2(eng_list, x_list_2d, y_list_2d, z_list_2d, r_list_2d, out_x, out_y, out_z, out_r, repeat_num=1, exposure_time=exposure_time,  sleep_time=1, chunk_size=chunk_size, simu=False, relative_move_flag=relative_move_flag, note=note, md=None)

        print('\n\nstart multi 3D xanes:')
        yield from multi_pos_xanes_3D(eng_list, x_list_3d, y_list_3d, z_list_3d, r_list_3d, exposure_time=exposure_time, relative_rot_angle=relative_rot_angle, period=exposure_time, out_x=out_x, out_y=out_y, out_z=out_z, out_r=out_r, rs=rs, simu=False, relative_move_flag=relative_move_flag, traditional_sequence_flag=1, note=note, sleep_time=0, repeat=1)
    insert_text('finished multiposition 2D xanes and 3D xanes')
    export_pdf(1)


Ni_eng_list = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Ni/eng_list_Ni_xanes_standard.txt')
Ni_eng_list_short = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Ni/eng_list_Ni_s_xanes_standard.txt')

Mn_eng_list = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Mn/eng_list_Mn_xanes_standard.txt')
Mn_eng_list_short = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Mn/eng_list_Mn_s_xanes_standard.txt')
Co_eng_list = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Co/eng_list_Co_xanes_standard.txt')
Co_eng_list_short = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Co/eng_list_Co_s_xanes_standard.txt')
Fe_eng_list = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Fe/eng_list_Fe_xanes_standard.txt')
Fe_eng_list_short = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Fe/eng_list_Fe_s_xanes_standard.txt')
V_eng_list = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/V/eng_list_V_xanes_standard.txt')
V_eng_list_short = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/V/eng_list_V_s_xanes_standard.txt')
Cr_eng_list = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Cr/eng_list_Cr_xanes_standard.txt')
Cr_eng_list_short = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Cr/eng_list_Cr_s_xanes_standard.txt')
Cu_eng_list = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Cu/eng_list_Cu_xanes_standard.txt')
Cu_eng_list_short = np.genfromtxt('/NSLS2/xf18id1/SW/xanes_ref/Cu/eng_list_Cu_s_xanes_standard.txt')


#def scan_3D_2D_overnight(n):
#
#    Ni_eng_list_insitu = np.arange(8.344, 8.363, 0.001)
#    pos1 = [30, -933, -578]
#    pos2 = [-203, -1077, 563]
#    x_list = [pos1[0]]
#    y_list = [pos1[1], pos2[1]]
#    z_list = [pos1[2], pos2[2]]
#    r_list = [-71, -71]
#
#    
#   
#        
#    #RE(multipos_2D_xanes_scan2(Ni_eng_list_insitu, x_list, y_list, z_list, [-40, -40], out_x=None, out_y=None, out_z=-2500, out_r=-90, repeat_num=1, exposure_time=0.1,  sleep_time=1, chunk_size=5, simu=False, relative_move_flag=0, note='NC_insitu'))
#
#    RE(mv(zps.sx, pos1[0], zps.sy, pos1[1], zps.sz, pos1[2], zps.pi_r, 0))
#    RE(xanes_scan2(Ni_eng_list_insitu, exposure_time=0.1, chunk_size=5, out_x=None, out_y=None, out_z=-3000, out_r=-90, simu=False, relative_move_flag=0, note='NC_insitu')        
#
#    pos1 = [30, -929, -568]
#    pos_cen = [-191, -813, -563]
#    for i in range(5):
#        print(f'repeating {i+1}/{5}')
#    
#        RE(mv(zps.sx, pos1[0], zps.sy, pos1[1], zps.sz, pos1[2], zps.pi_r, -72))
#        RE(xanes_3D(Ni_eng_list_insitu, exposure_time=0.1, relative_rot_angle=138, period=0.1, out_x=None, out_y=None, out_z=-3000, out_r=-90, rs=2, simu=False, relative_move_flag=0, traditional_sequence_flag=1, note='NC_insitu'))   
#        
#    
#        RE(mv(zps.sx, pos_cen[0], zps.sy, pos_cen[1], zps.sz, pos_cen[2], zps.pi_r, 0))
#        RE(raster_2D_scan(x_range=[-1,1],y_range=[-1,1],exposure_time=0.1, out_x=None, out_y=None, out_z=-3000, out_r=-90, img_sizeX=640,img_sizeY=540,pxl=80, simu=False, relative_move_flag=0,rot_first_flag=1,note='NC_insitu'))
#        
#        RE(raster_2D_xanes2(Ni_eng_list_insitu, x_range=[-1,1],y_range=[-1,1],exposure_time=0.1, out_x=None, out_y=None, out_z=-3000, out_r=-90, img_sizeX=640, img_sizeY=540, pxl=80, simu=False, relative_move_flag=0, rot_first_flag=1,note='NC_insitu'))
#    
#        RE(mv(zps.sx, pos1[0], zps.sy, pos1[1], zps.sz, pos1[2], zps.pi_r, -72))
#        RE(fly_scan(exposure_time=0.1, relative_rot_angle =138, period=0.1, chunk_size=20, out_x=None, out_y=None, out_z=-3000, out_r=-90, rs=1.5, simu=False, relative_move_flag=0, traditional_sequence_flag=0, note='NC_insitu'))
#        
#        RE(bps.sleep(600))
#        print('sleep for 600sec')
###############################
        
        
    

