from scipy.signal import medfilt
import numpy as np
from lsq_fit import lsq_fit_iter, lsq_fit_iter2
from scipy.interpolate import interp1d
from copy import deepcopy
from util import find_nearest, fit_curve



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


def fit_2D_xanes_non_iter(img_xanes, eng, spectrum_ref):
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
    cost = cost.reshape(1, s[1], s[2])
    x = x.reshape(num_ref, s[1], s[2])
    return x, cost



def fit_2D_xanes_iter(img_xanes, eng, spectrum_ref, coef0=None, learning_rate=0.005, n_iter=10, bounds=[0,1]):
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
    s = img_xanes.shape
    A = [] 
    for i in range(num_ref):
        tmp = interp1d(spectrum_ref[f'ref{i}'][:,0], spectrum_ref[f'ref{i}'][:,1], kind='cubic')
        A.append(tmp(eng).reshape(1, len(eng)))
    A = np.squeeze(A).T
    Y = img_xanes.reshape(img_xanes.shape[0], -1)
    if coef0 is None:
        W = None
    else:
        W = coef0.reshape(coef0.shape[0], -1)
    w, cost = lsq_fit_iter2(A, Y, W, learning_rate, n_iter, bounds, print_flag=1)
    w = w.reshape(len(w), img_xanes.shape[1], img_xanes.shape[2])
    cost = cost[-1].reshape(1, s[1], s[2])

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


def normalize_2D_xanes(img_stack, xanes_eng, pre_edge, post_edge):
    pre_s, pre_e = pre_edge
    post_s, post_e = post_edge
    img_norm = deepcopy(img_stack)
    s0 = img_norm.shape
    x_eng = xanes_eng
    try:
        # pre_edge
        xs, xe = find_nearest(x_eng, pre_s), find_nearest(x_eng, pre_e)
        if xs == xe:
            eng_pre = x_eng[xs]
            img_pre = img_norm[xs]
            img_norm = img_norm - img_pre
        elif xe > xs:
            eng_pre = x_eng[xs:xe]
            img_pre = img_norm[xs:xe]
            s = img_pre.shape
            x_pre = eng_pre.reshape(len(eng_pre), 1)
            x_bar_pre = np.mean(x_pre)
            x_dif_pre = x_pre - x_bar_pre
            SSx_pre = np.dot(x_dif_pre.T, x_dif_pre)
            y_bar_pre = np.mean(img_pre, axis=0)
            p = img_pre - y_bar_pre
            for i in range(s[0]):
                p[i] = p[i] * x_dif_pre[i]
            SPxy_pre = np.sum(p, axis=0)
            b0_pre = y_bar_pre - SPxy_pre / SSx_pre * x_bar_pre
            b1_pre = SPxy_pre / SSx_pre
            for i in range(s0[0]):
                if not i%10:
                    print(f'current image: {i}')
                img_norm[i] = img_norm[i] - (b0_pre + b1_pre * x_eng[i])
        else:
            print('check pre-edge/post-edge energy')

        # post_edge
        xs, xe = find_nearest(x_eng, post_s), find_nearest(x_eng, post_e)
        if xs == xe:
            eng_post = x_eng[xs]
            img_post = img_norm[xs]
            img_norm = img_norm / img_post
            img_norm[np.isnan(img_norm)] = 0
            img_norm[np.isinf(img_norm)] = 0
        elif xe > xs:
            eng_post = x_eng[xs:xe]
            img_post = img_norm[xs:xe]
            s = img_post.shape
            x_post = eng_post.reshape(len(eng_post), 1)
            x_bar_post = np.mean(x_post)
            x_dif_post = x_post - x_bar_post
            SSx_post = np.dot(x_dif_post.T, x_dif_post)
            y_bar_post = np.mean(img_post, axis=0)
            p = img_post - y_bar_post
            for i in range(s[0]):
                p[i] = p[i] * x_dif_post[i]
            SPxy_post = np.sum(p, axis=0)
            b0_post = y_bar_post - SPxy_post / SSx_post * x_bar_post
            b1_post = SPxy_post / SSx_post
            for i in range(s0[0]):
                img_norm[i] = img_norm[i] / (b0_post + b1_post * x_eng[i])
        else:
            print('check pre-edge/post-edge energy')
        img_norm[np.isnan(img_norm)] = 0
        img_norm[np.isinf(img_norm)] = 0
    except:
        print('Normalization fails ...')
    finally:
        return img_norm


def normalize_1D_xanes(xanes_spec, xanes_eng, pre_edge, post_edge):

    pre_s, pre_e = pre_edge
    post_s, post_e = post_edge
    x_eng = xanes_eng
    xanes_spec_fit = deepcopy(xanes_spec)
    xs, xe = find_nearest(x_eng, pre_s), find_nearest(x_eng, pre_e)
    pre_eng = x_eng[xs:xe]
    pre_spec = xanes_spec[xs:xe]
    print(f'{pre_spec.shape}')
    if len(pre_eng) > 1:
        y_pre_fit = fit_curve(pre_eng, pre_spec, x_eng)
        xanes_spec_tmp = xanes_spec - y_pre_fit
        pre_fit_flag = True
    elif len(pre_eng) <= 1:
        y_pre_fit = np.ones(x_eng.shape) * xanes_spec[xs]
        xanes_spec_tmp = xanes_spec - y_pre_fit
        pre_fit_flag = True
    else:
        print('invalid pre-edge assignment')

    # fit post-edge
    xs, xe = find_nearest(x_eng, post_s), find_nearest(x_eng, post_e)
    post_eng = x_eng[xs:xe]
    post_spec = xanes_spec_tmp[xs:xe]
    if len(post_eng) > 1:
        y_post_fit = fit_curve(post_eng, post_spec, x_eng)
        post_fit_flag = True
    elif len(post_eng) <= 1:
        y_post_fit = np.ones(x_eng.shape) * xanes_spec_tmp[xs]
        post_fit_flag = True
    else:
        print('invalid pre-edge assignment')


    if pre_fit_flag and post_fit_flag:
        xanes_spec_fit = xanes_spec_tmp * 1.0 / y_post_fit
        xanes_spec_fit[np.isnan(xanes_spec_fit)] = 0
        xanes_spec_fit[np.isinf(xanes_spec_fit)] = 0

    return xanes_spec_fit, y_pre_fit, y_post_fit





