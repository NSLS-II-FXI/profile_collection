import numpy as np
import matplotlib.pylab as plt
import h5py
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial.polynomial import polyval
from align_image import align_img_stack
from image_binning import bin_ndarray
from scipy import signal


fn = '/home/mingyuan/Work/pyqt_work/tomo_backup/LMO_4.3V_XANES_prj_76.h5'
f = h5py.File(fn, 'r')
prj = np.array(f['t0/channel0'])
f.close()

prj = align_img_stack(prj)
prj = prj[1:]
prj_bin = bin_ndarray(prj, (prj.shape[0], int(prj.shape[1]/2), int(prj.shape[2]/2)))
prj_bin[prj_bin<0]=0
prj_bin[np.isnan(prj_bin)]=0
prj_bin[np.isinf(prj_bin)]=0
fout = 'LMO_XANES_align.h5'
with h5py.File(fout, 'w') as hf:
    hf.create_dataset('xanes_image', data=prj_bin)


prj_sum = np.sum(np.sum(prj_bin,axis=1),axis=1)
plt.plot(prj_sum,'r.-');plt.show()
