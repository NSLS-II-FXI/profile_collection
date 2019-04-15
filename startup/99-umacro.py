
new_user()
show_global_para()
run_pdf()
read_calib_file()
    
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




