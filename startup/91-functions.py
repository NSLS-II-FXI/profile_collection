import numpy as np
import matplotlib.pylab as plt
import h5py
from operator import attrgetter
from PIL import Image
from scipy.signal import medfilt2d

#from image_binning import bin_ndarray

GLOBAL_MAG = 377 # total magnification
GLOBAL_VLM_MAG = 10 # vlm magnification
OUT_ZONE_WIDTH = 30 # 30 nm
ZONE_DIAMETER = 200 # 200 um
CURRENT_MAG_1 = GLOBAL_MAG
CURRENT_MAG_2 = GLOBAL_MAG
CALIBER_FLAG = 1



CALIBER = {}

def record_calib_pos1():
    global CALIBER_FLAG
    CALIBER['th2_pos1'] = pzt_dcm_th2.pos.value
    CALIBER['chi2_pos1'] = pzt_dcm_chi2.pos.value
    CALIBER['XEng_pos1'] = XEng.position
    CALIBER['zp_x_pos1'] = zp.x.position
    CALIBER['zp_y_pos1'] = zp.y.position
    print(f'pzt_dcm_th2_{CALIBER["XEng_pos1"]:2.4f}\t: {CALIBER["th2_pos1"]:2.4f}')
    print(f'pzt_dcm_chi2_{CALIBER["XEng_pos1"]:2.4f}\t: {CALIBER["chi2_pos1"]:2.4f}')
    print(f'zp_x_{CALIBER["XEng_pos1"]:2.4f}\t\t: {CALIBER["zp_x_pos1"]:2.4f}')
    print(f'zp_y_{CALIBER["XEng_pos1"]:2.4f}\t\t: {CALIBER["zp_y_pos1"]:2.4f}')

    df = pd.DataFrame.from_dict(CALIBER, orient='index')
    CALIBER_FLAG = 1 - CALIBER_FLAG
    df.to_csv('/home/xf18id/.ipython/profile_collection/startup/calib.csv', sep='\t')
    CURRENT_MAG_1 = (DetU.z.position / zp.z.position - 1) * GLOBAL_VLM_MAG
    if CALIBER_FLAG:
        if np.abs(CURRENT_MAG_1 - CURRENT_MAG_2) > 0.1:
            print('MAGNIFICATION between two calibration points does not match!!')
            print(f'MAGNIFICATION_1 = {CURRENT_MAG_1}\nMAGNIFICATION_2 = {CURRENT_MAG_2}')
        else:
            GLOBAL_MAG = np.round(CURRENT_MAG_1 * 100) / 100.
    print(f'calib_pos1 recored: current Magnification = {CURRENT_MAG_1}')


def record_calib_pos2():
    global CALIBER_FLAG
    CALIBER['th2_pos2'] = pzt_dcm_th2.pos.value
    CALIBER['chi2_pos2'] = pzt_dcm_chi2.pos.value
    CALIBER['XEng_pos2'] = XEng.position
    CALIBER['zp_x_pos2'] = zp.x.position
    CALIBER['zp_y_pos2'] = zp.y.position
    print(f'pzt_dcm_th2_{CALIBER["XEng_pos2"]:2.4f}\t: {CALIBER["th2_pos2"]:2.4f}')
    print(f'pzt_dcm_chi2_{CALIBER["XEng_pos2"]:2.4f}\t: {CALIBER["chi2_pos2"]:2.4f}')
    print(f'zp_x_{CALIBER["XEng_pos2"]:2.4f}\t\t: {CALIBER["zp_x_pos2"]:2.4f}')
    print(f'zp_y_{CALIBER["XEng_pos2"]:2.4f}\t\t: {CALIBER["zp_y_pos2"]:2.4f}')
 
    df = pd.DataFrame.from_dict(CALIBER, orient='index')
    df.to_csv('/home/xf18id/.ipython/profile_collection/startup/calib.csv', sep='\t')
    CALIBER_FLAG = 1 - CALIBER_FLAG
    CURRENT_MAG_2 = (DetU.z.position / zp.z.position - 1) * GLOBAL_VLM_MAG
    if CALIBER_FLAG:
        if np.abs(CURRENT_MAG_1 - CURRENT_MAG_2) > 0.1:
            print('MAGNIFICATION between two calibration points does not match!!')
            print(f'MAGNIFICATION_1 = {CURRENT_MAG_1}\nMAGNIFICATION_2 = {CURRENT_MAG_2}')
        else:
            GLOBAL_MAG = np.round(CURRENT_MAG_2 * 100) / 100.
    print(f'calib_pos2 recored: current Magnification = {CURRENT_MAG_2}')


def read_calib_file():
    fn = '/home/xf18id/.ipython/profile_collection/startup/calib.csv'
    try:

        df = pd.Series.from_csv(fn, header=0).to_dict()
        CALIBER['th2_pos1'] = np.float(df['th2_pos1'])
        CALIBER['chi2_pos1'] = np.float(df['chi2_pos1'])
        ALIBER['XEng_pos1'] = np.float(df['XEng_pos1'])
        CALIBER['zp_x_pos1'] = np.float(df['zp_x_pos1'])
        CALIBER['zp_y_pos1'] = np.float(df['zp_y_pos1'])

        CALIBER['th2_pos2'] = np.float(df['th2_pos2'])
        CALIBER['chi2_pos2'] = np.float(df['chi2_pos2'])
        CALIBER['XEng_pos2'] = np.float(df['XEng_pos2'])
        CALIBER['zp_x_pos2'] = np.float(df['zp_x_pos2'])
        CALIBER['zp_y_pos2'] = np.float(df['zp_y_pos2'])
    
        print(f'pzt_dcm_th2_{CALIBER["XEng_pos1"]:2.4f}\t: {CALIBER["th2_pos1"]:2.4f}')
        print(f'pzt_dcm_chi2_{CALIBER["XEng_pos1"]:2.4f}\t: {CALIBER["chi2_pos1"]:2.4f}')
        print(f'zp_x_{CALIBER["XEng_pos1"]:2.4f}\t\t: {CALIBER["zp_x_pos1"]:2.4f}')
        print(f'zp_y_{CALIBER["XEng_pos1"]:2.4f}\t\t: {CALIBER["zp_y_pos1"]:2.4f}')
        print('\n')
        print(f'pzt_dcm_th2_{CALIBER["XEng_pos2"]:2.4f}\t: {CALIBER["th2_pos2"]:2.4f}')
        print(f'pzt_dcm_chi2_{CALIBER["XEng_pos2"]:2.4f}\t: {CALIBER["chi2_pos2"]:2.4f}')
        print(f'zp_x_{CALIBER["XEng_pos2"]:2.4f}\t\t: {CALIBER["zp_x_pos2"]:2.4f}')
        print(f'zp_y_{CALIBER["XEng_pos2"]:2.4f}\t\t: {CALIBER["zp_y_pos2"]:2.4f}')
    except:
        print('reading calibration file: {fn} fails...\n Please optimize optics at two energy points, and using record_calib_pos1() and record_calib_pos2() after optimizing each energy points ')


def show_global_para():
    print(f'GLOBAL_MAG = {GLOBAL_MAG} X') # total magnification
    print(f'GLOBAL_VLM_MAG = {GLOBAL_VLM_MAG} X') # vlm magnification
    print(f'OUT_ZONE_WIDTH = {OUT_ZONE_WIDTH} nm') # 30 nm
    print(f'ZONE_DIAMETER = {ZONE_DIAMETER} um') # 200 um
    print(f'\nFor Andor camera, current pixel size = {6500./GLOBAL_MAG:3.1f} nm')
    print('\nChange parameters if necessary.')


#def list_fun():
#    import umacro
#    all_func = inspect.getmembers(umacro, inspect.isfunction)
#    return all_func     

################################################################
####################  create new user  #########################
################################################################

def new_user():
    now = datetime.now()
    year = np.str(now.year)
    mon  = '{:02d}'.format(now.month)

    if now.month >= 1 and now.month <=4:    qut = 'Q1'
    elif now.month >= 5 and now.month <=8:  qut = 'Q2'
    else: qut = 'Q3'

    pre = '/NSLS2/xf18id1/users/' + year + qut + '/'
    try:        os.mkdir(pre)
    except Exception:    pass
    print('\n')

    PI_name = input('PI\'s name:')  
    PI_name = PI_name.replace(' ', '_').upper()

    if PI_name[0] == '*':
        cwd = os.getcwd()
        print(f'stay at current directory: {cwd}')
        return
    if PI_name[:4] == 'COMM':
        PI_name = 'FXI_commission'
        fn = pre + PI_name
    else:
        proposal_id = input('Proposal ID:')
        fn = pre + PI_name + '_Proposal_' + proposal_id
    try:        os.mkdir(fn)
    except Exception:
        print('Found (user, proposal) existed\nEntering folder: {}'.format(os.getcwd()))
        os.chdir(fn)       
        pass
    os.chdir(fn)
    print ('\nUser creating successful!\n\nEntering folder: {}\n'.format(os.getcwd()))


################################################################
####################   TXM paramter  ###########################
################################################################


def get_img(h, det='Andor'):
    "Take in a Header and return a numpy array of detA1 image(s)."
    img = list(h.data('Andor_image'))
    return np.squeeze(np.array(img))


def get_scan_parameter(scan_id=-1):
    h=db[scan_id]
    scan_id = h.start['scan_id']
    uid = h.start['uid']
    X_eng = f'{h.start["XEng"]:2.4f}'
    scan_type = h.start['plan_name']
    txt = ''
    for key, val in h.start['plan_args'].items():
        txt += f'{key}={val}, '
    txt0 = f'#{scan_id}  (uid: {uid[:6]},  X_Eng: {X_eng} keV)\n'
    txt = txt0 + scan_type + '(' + txt[:-2] + ')'
    return txt

from bluesky.callbacks import CallbackBase

class PdfMaker(CallbackBase):
    def start(self, doc):
        self._last_start = doc
        print('HI')

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
    '''
    check energy in range of 4.000-12.000
    Inputs:
    --------
    eng: list
        e.g. [6.000,7.500]
    '''
    eng = list(eng)
    high_limit = 12.000
    low_limit = 4.000
    for i in range(len(eng)):
        assert(eng[i] >= low_limit and eng[i] <= high_limit), 'Energy is outside the range (4.000, 12.000) keV'
    return 
        
def cal_parameter(eng, print_flag=1):
    '''
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
    '''
    
    global OUT_ZONE_WIDTH 
    global ZONE_DIAMETER 

    h = 6.6261e-34
    c = 3e8
    ec = 1.602e-19

#    if eng < 4000:    eng = XEng.position * 1000 # current beam energy
    check_eng_range([eng])

    wave_length = h * c / (ec * eng*1000) * 1e9 # nm
    focal_length = OUT_ZONE_WIDTH * ZONE_DIAMETER / (wave_length) / 1000 # mm 
    NA = wave_length / (2 * OUT_ZONE_WIDTH)
    DOF = wave_length / NA**2 / 1000 # um
    if  print_flag:
        print ('Wave length: {0:2.2f} nm\nFocal_length: {1:2.2f} mm\nNA: {2:2.2f} mrad\nDepth of focus: {3:2.2f} um'.format(wave_length, focal_length, NA*1e3, DOF))
    else:
        return wave_length, focal_length, NA, DOF
    

def cal_zp_ccd_position(eng_new, eng_ini=0, print_flag=1):

    '''
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

    '''

    global GLOBAL_MAG
    global GLOBAL_VLM_MAG
    
    if eng_ini < 4.000:    
        eng_ini = XEng.position # current beam energy
    check_eng_range([eng_new, eng_ini])
  
    
    h = 6.6261e-34
    c = 3e8
    ec = 1.602e-19
  
    det = DetU    # read current energy and motor position
    mag = GLOBAL_MAG / GLOBAL_VLM_MAG

    zp_ini = zp.z.position # zone plate position in unit of mm
    zps_ini = zps.sz.position # sample position in unit of mm
    det_ini = det.z.position # detector position in unit of mm

    lamda_ini, fl_ini, _, _ = cal_parameter(eng_ini, print_flag=0)
    lamda, fl, _, _ = cal_parameter(eng_new, print_flag=0)
    
    p_ini = fl_ini * (mag+1) / mag # sample distance (mm), relative to zone plate
    q_ini = mag * p_ini  # ccd distance (mm), relative to zone plate

    p_cal = fl * (mag+1) / mag
    q_cal = mag * p_cal

    zp_delta = p_cal - p_ini
    det_delta = q_cal - q_ini + zp_delta

    zp_final = p_cal
    det_final = p_cal*(mag+1)

    if print_flag:    
        print ('Calculation results:')
        print ('Change energy from: {0:2.2f} eV to {1:2.2f} eV'.format(eng_ini, eng_new))
        print ('Need to move zone plate by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)'
                .format(zp_delta, zp_ini, zp_final))
        print ('Need to move CCD by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)'
                .format(det_delta, det_ini, det_final)) 
    else:
        return zp_ini, det_ini, zp_delta, det_delta, zp_final, det_final


def move_zp_ccd(eng_new, move_flag=1, info_flag=1):
    '''
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
    '''
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
        
        pzt_dcm_th2_eng1 = CALIBER['th2_pos1']
        pzt_dcm_chi2_eng1 = CALIBER['chi2_pos1']
        zp_x_pos_eng1 = CALIBER['zp_x_pos1']
        zp_y_pos_eng1 = CALIBER['zp_y_pos1']


        pzt_dcm_th2_eng2 = CALIBER['th2_pos2']
        pzt_dcm_chi2_eng2 = CALIBER['chi2_pos2']
        zp_x_pos_eng2 = CALIBER['zp_x_pos2']
        zp_y_pos_eng2 = CALIBER['zp_y_pos2']

        
        pzt_dcm_th2_target = (eng_new - eng2) * (pzt_dcm_th2_eng1 - pzt_dcm_th2_eng2) / (eng1-eng2) + pzt_dcm_th2_eng2
        pzt_dcm_chi2_target = (eng_new - eng2) * (pzt_dcm_chi2_eng1 - pzt_dcm_chi2_eng2) / (eng1-eng2) + pzt_dcm_chi2_eng2
        zp_x_target = (eng_new - eng2)*(zp_x_pos_eng1 - zp_x_pos_eng2)/(eng1 - eng2) + zp_x_pos_eng2
        zp_y_target = (eng_new - eng2)*(zp_y_pos_eng1 - zp_y_pos_eng2)/(eng1 - eng2) + zp_y_pos_eng2

        pzt_dcm_th2_ini = pzt_dcm_th2.pos.value
        pzt_dcm_chi2_ini = pzt_dcm_chi2.pos.value
        zp_x_ini = zp.x.position    
        zp_y_ini = zp.y.position
        

    
        if move_flag: # move stages
            print ('Now moving stages ....')     
            if info_flag: 
                print ('Energy: {0:5.2f} keV --> {1:5.2f} keV'.format(eng_ini, eng_new))
                print ('zone plate position: {0:2.4f} mm --> {1:2.4f} mm'.format(zp_ini, zp_final))
                print ('CCD position: {0:2.4f} mm --> {1:2.4f} mm'.format(det_ini, det_final)) 
                print ('move zp_x: ({0:2.4f} um --> {1:2.4f} um)'.format(zp_x_ini, zp_x_target))
                print ('move zp_y: ({0:2.4f} um --> {1:2.4f} um)'.format(zp_y_ini, zp_y_target))
                print ('move pzt_dcm_th2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_th2_ini, pzt_dcm_th2_target))
                print ('move pzt_dcm_chi2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_chi2_ini, pzt_dcm_chi2_target))
            yield from mv(zp.z, zp_final,det.z, det_final, XEng, eng_new)
            yield from mv(pzt_dcm_th2.setpos, pzt_dcm_th2_target, pzt_dcm_chi2.setpos, pzt_dcm_chi2_target)
            yield from mv(zp.x, zp_x_target, zp.y, zp_y_target)
        else:
            print ('This is calculation. No stages move') 
            print ('Will move Energy: {0:5.2f} keV --> {1:5.2f} keV'.format(eng_ini, eng_new))
            print ('will move zone plate down stream by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)'.format(zp_delta, zp_ini, zp_final))
            print ('will move CCD down stream by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)'.format(det_delta, det_ini, det_final))
            print ('will move zp_x: ({0:2.4f} um --> {1:2.4f} um)'.format(zp_x_ini, zp_x_target))
            print ('will move zp_y: ({0:2.4f} um --> {1:2.4f} um)'.format(zp_y_ini, zp_y_target))
            print ('will move pzt_dcm_th2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_th2_ini, pzt_dcm_th2_target))
            print ('will move pzt_dcm_chi2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_chi2_ini, pzt_dcm_chi2_target))
    else:
        print('record_calib_pos1() or record_calib_pos2() not excuted...\nWill not move anything')

def cal_phase_ring_position(eng_new, eng_ini=0, print_flag=1):
    '''
    calculate delta amount of phase_ring movement:
    positive means move down-stream, negative means move up-stream
    
    use as:
        cal_phase_ring_with_const_mag(eng_new=8.0, eng_ini=9.0)

    Inputs:
    -------
    eng_new: float
        target energy, in unit of keV
    eng_ini: float
        initial energy, in unit of keV
        it will read current Xray energy if eng_ini < 4.0 keV

    '''

    _, fl_ini, _, _ = cal_parameter(eng_ini, print_flag=0)
    _, fl_new, _, _ = cal_parameter(eng_new, print_flag=0) 
       
    _, _, zp_delta, _, _, _ = cal_zp_ccd_position(eng_new, eng_ini, print_flag=0)

    delta_phase_ring = zp_delta + fl_new - fl_ini
    if print_flag:    
        print ('Need to move phase ring down-stream by: {0:2.2f} mm'.format(delta_phase_ring))
        print ('Zone plate position changes by: {0:2.2f} mm'.format(zp_delta))
        print ('Zone plate focal length changes by: {0:2.2f} mm'.format(fl_new - fl_ini))
    else:    return delta_phase_ring


def move_phase_ring(eng_new, eng_ini, flag=1):
    '''
    move the phase_ring when change the energy
        
    use as:
        move_phase_ring_with_const_mag(eng_new=8.0, eng_ini=9.0, flag=1)

    Inputs:
    -------
    eng_new: float
        target energy, in unit of keV
    eng_ini: float
        initial energy, in unit of keV
        it will read current Xray energy if eng_ini < 4.0
    flag: int
         0: no move
         1: move stage
    '''

    delta_phase_ring = cal_phase_ring_position(eng_new, eng_ini, print_flag=0)
    if flag:    RE(mvr(phase_ring.z, delta_phase_ring))
    else: 
        print ('This is calculation. No stages move.')
        print ('will move phase ring down stream by {0:2.2f} mm'.format(delta_phase_ring)) 
    return 
            

def go_det(det):
    if det == 'manta_u':
        pos_x = -87.6
        pos_y = 0.85
        DetU.x.move(pos_x, wait = 'False')
        DetU.y.move(pos_y, wait = 'False')
        print('move DetU.x to {0:2.2f}\nmove DetU.y to {1:2.2f}'.format(pos_x, pos_y))
    else:
        print('Detector not defined...')



def set_ic_dwell_time(dwell_time=1.):
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
        print('dwell_time not in list, set to default value: 1s')
        ic_rate.value = 6

def plot_ssa_ic(scan_id, ic='ic4'):
    h = db[scan_id]
    x = np.array(list(h.data('ssa_v_cen')))
    if len(x) > 0:
        xlabel = 'ssa_v_cen'
        xdata = x
    x = np.array(list(h.data('ssa_h_cen')))
    if len(x) > 0:
        xlabel = 'ssa_h_cen'
        xdata = x
    x = np.array(list(h.data('ssa_v_gap')))
    if len(x) > 0:
        xlabel = 'ssa_v_gap'
        xdata = x
    x = np.array(list(h.data('ssa_h_gap')))
    if len(x) > 0:
        xlabel = 'ssa_h_gap'
        xdata = x
    ydata = np.array(list(h.data(ic)))

    plt.figure()
    plt.plot(xdata, -ydata, 'r.-')
    plt.xlabel(xlabel)
    plt.ylabel(ic + ' counts')
    plt.show()
    

def read_ic(ics, num, dwell_time=1.):
    '''
    read ion-chamber value
    e.g. RE(read_ic([ic1, ic2], num = 10, dwell_time=0.5))

    Inputs:
    -------
    ics: list of ion-chamber
    num: int
        number of points to record
    dwell_time: float
        in unit of seconds
    '''

    set_ic_dwell_time(dwell_time=dwell_time)
    yield from count(ics, num, delay = dwell_time)
    h = db[-1]
    ic_num = len(ics)
    fig = plt.figure()
    x = np.linspace(1, num, num)
    y = np.zeros([ic_num, num])
    for i in range(ic_num):
        y[i] = np.array(list(h.data(ics[i].name)))
        ax = fig.add_subplot(ic_num, 1, i+1)
        ax.title.set_text(ics[i].name)        
        ax.plot(x, y[i], '.-')
    fig.subplots_adjust(hspace=.5)
    plt.show()
    return y


def go_eng(eng):
    '''
    Move DCM to specifc energy
    E.g. RE(go_eng(9.0)) # Note that: energy is in unit of keV
    '''
    check_eng_range([eng])
    yield from mv(XEng, eng)

################################################################
####################    load scan  #############################
################################################################

def load_scan(scan_id):
    '''
    e.g. load_scan([0001, 0002]) 
    '''
    for item in scan_id:        
        load_single_scan(int(item))  
        

def load_single_scan(scan_id=-1):
    h = db[scan_id]
    scan_id = h.start['scan_id']
    scan_type = h.start['plan_name']
#    x_eng = h.start['XEng']
     
    if scan_type == 'tomo_scan':
        print('loading tomo scan: #{}'.format(scan_id))
        load_tomo_scan(h)
        print('tomo scan: #{} loading finished'.format(scan_id))
    elif scan_type == 'fly_scan':
        print('loading fly scan: #{}'.format(scan_id))
        load_fly_scan(h)
        print('fly scan: #{} loading finished'.format(scan_id))
    elif scan_type == 'xanes_scan' or scan_type == 'xanes_scan2':
        print('loading xanes scan: #{}'.format(scan_id))
        load_xanes_scan(h)
        print('xanes scan: #{} loading finished'.format(scan_id))
    elif scan_type == 'z_scan':
        print('loading z_scan: #{}'.format(scan_id))
        load_z_scan(h)
    elif scan_type == 'test_scan':
        print('loading test_scan: #{}'.format(scan_id))
        load_test_scan(h)    
    elif scan_type == 'multipos_count':
        print(f'loading multipos_count: #{scan_id}')
        load_multipos_count(h)
    elif scan_type == 'grid2D_rel':
        print('loading grid2D_rel: #{}'.format(scan_id))
        load_grid2D_rel(h)
    elif scan_type == 'count':
        print('loading count: #{}'.format(scan_id))
        load_count_img(h)
    else:
        print('Un-recognized scan type ......')
        pass
        

def load_tomo_scan(h):
    scan_type = 'tomo_scan'
    scan_id = h.start['scan_id']
    x_eng = h.start['x_ray_energy']
    bkg_img_num = h.start['num_bkg_images']
    dark_img_num = h.start['num_dark_images']
    angle_i = h.start['plan_args']['start']
    angle_e = h.start['plan_args']['stop']
    angle_n = h.start['plan_args']['num'] 
    exposure_t = h.start['plan_args']['exposure_time'] 
    img = np.array(list(h.data('Andor_image')))
    img = np.squeeze(img)
    img_dark = img[0:dark_img_num]
    img_tomo = img[dark_img_num : -bkg_img_num]
    img_bkg = img[-bkg_img_num:]
    

    s = img_dark.shape
    img_dark_avg = np.mean(img_dark,axis=0).reshape(1, s[1], s[2])
    img_bkg_avg = np.mean(img_bkg, axis=0).reshape(1, s[1], s[2])
    img_angle = np.linspace(angle_i, angle_e, angle_n)
        
    fname = scan_type + '_id_' + str(scan_id) + '.h5'
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('X_eng', data = x_eng)
        hf.create_dataset('img_bkg', data = img_bkg)
        hf.create_dataset('img_dark', data = img_dark)
        hf.create_dataset('img_tomo', data = img_tomo)
        hf.create_dataset('angle', data = img_angle)
    del img
    del img_tomo
    del img_dark
    del img_bkg


def load_fly_scan(h): 
    uid = h.start['uid']
    note = h.start['note']
    scan_type = 'fly_scan'
    scan_id = h.start['scan_id']   
    scan_time = h.start['time'] 
    x_eng = h.start['XEng']
    chunk_size = h.start['chunk_size']
    # sanity check: make sure we remembered the right stream name
    assert 'zps_pi_r_monitor' in h.stream_names
    pos = h.table('zps_pi_r_monitor')
    imgs = np.array(list(h.data('Andor_image')))
    img_dark = imgs[0]
    img_bkg = imgs[-1]
    s = img_dark.shape
    img_dark_avg = np.mean(img_dark, axis=0).reshape(1, s[1], s[2])
    img_bkg_avg = np.mean(img_bkg, axis=0).reshape(1, s[1], s[2])

    imgs = imgs[1:-1]
    s1 = imgs.shape
    imgs =imgs.reshape([s1[0]*s1[1], s1[2], s1[3]])

    with db.reg.handler_context({'AD_HDF5': AreaDetectorHDF5TimestampHandler}):
        chunked_timestamps = list(h.data('Andor_image'))
    
    chunked_timestamps = chunked_timestamps[1:-1]
    raw_timestamps = []
    for chunk in chunked_timestamps:
        raw_timestamps.extend(chunk.tolist())

    timestamps = convert_AD_timestamps(pd.Series(raw_timestamps))
    pos['time'] = pos['time'].dt.tz_localize('US/Eastern')

    img_day, img_hour = timestamps.dt.day, timestamps.dt.hour, 
    img_min, img_sec, img_msec = timestamps.dt.minute, timestamps.dt.second, timestamps.dt.microsecond
    img_time = img_day * 86400 + img_hour * 3600 + img_min * 60 + img_sec + img_msec * 1e-6
    img_time = np.array(img_time)

    mot_day, mot_hour = pos['time'].dt.day, pos['time'].dt.hour, 
    mot_min, mot_sec, mot_msec = pos['time'].dt.minute, pos['time'].dt.second, pos['time'].dt.microsecond
    mot_time = mot_day * 86400 + mot_hour * 3600 + mot_min * 60 + mot_sec + mot_msec * 1e-6
    mot_time =  np.array(mot_time)

    mot_pos = np.array(pos['zps_pi_r'])
    offset = np.min([np.min(img_time), np.min(mot_time)])
    img_time -= offset
    mot_time -= offset
    mot_pos_interp = np.interp(img_time, mot_time, mot_pos)
    
    pos2 = mot_pos_interp.argmax() + 1
    img_angle = mot_pos_interp[:pos2-chunk_size] # rotation angles
    img_tomo = imgs[:pos2-chunk_size]  # tomo images
    
    fname = scan_type + '_id_' + str(scan_id) + '.h5'
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('note', data = note)
        hf.create_dataset('uid', data = uid)
        hf.create_dataset('scan_id', data = int(scan_id))
        hf.create_dataset('scan_time', data = scan_time)
        hf.create_dataset('X_eng', data = x_eng)
        hf.create_dataset('img_bkg', data = img_bkg)
        hf.create_dataset('img_dark', data = img_dark)
        hf.create_dataset('img_bkg_avg', data = img_bkg_avg)
        hf.create_dataset('img_dark_avg', data = img_dark_avg)
        hf.create_dataset('img_tomo', data = img_tomo)
        hf.create_dataset('angle', data = img_angle)
    del img_tomo
    del img_dark
    del img_bkg
    del imgs
    

def load_xanes_scan(h):
    scan_type = h.start['plan_name']
#    scan_type = 'xanes_scan'
    uid = h.start['uid']
    note = h.start['note']
    scan_id = h.start['scan_id']  
    scan_time = h.start['time']
    x_eng = h.start['x_ray_energy']
#    chunk_size = h.start['chunk_size']
    chunk_size = h.start['num_bkg_images']
    num_eng = h.start['num_eng']
    
    imgs = np.array(list(h.data('Andor_image')))
    img_dark = imgs[0]
    img_dark_avg = np.mean(img_dark, axis=0).reshape([1,img_dark.shape[1], img_dark.shape[2]])
    eng_list = list(h.start['eng_list'])
    s = imgs.shape
    img_xanes_avg = np.zeros([num_eng, s[2], s[3]])   
    img_bkg_avg = np.zeros([num_eng, s[2], s[3]])  

    if scan_type == 'xanes_scan':
        for i in range(num_eng):
            img_xanes = imgs[i+1]
            img_xanes_avg[i] = np.mean(img_xanes, axis=0)
            img_bkg = imgs[i+1 + num_eng]
            img_bkg_avg[i] = np.mean(img_bkg, axis=0)
    elif scan_type == 'xanes_scan2':
        j = 1
        for i in range(num_eng):
            img_xanes = imgs[j]
            img_xanes_avg[i] = np.mean(img_xanes, axis=0)
            img_bkg = imgs[j+1]
            img_bkg_avg[i] = np.mean(img_bkg, axis=0)
            j = j+2
    else:
        print('un-recognized xanes scan......')
        return 0
    img_xanes_norm = (img_xanes_avg - img_dark_avg) * 1.0 / (img_bkg_avg - img_dark_avg)
    img_xanes_norm[np.isnan(img_xanes_norm)] = 0
    img_xanes_norm[np.isinf(img_xanes_norm)] = 0        
    img_bkg = np.array(img_bkg, dtype=np.float32)
#    img_xanes_norm = np.array(img_xanes_norm, dtype=np.float32)
    fname = scan_type + '_id_' + str(scan_id) + '.h5'
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('uid', data = uid)
        hf.create_dataset('scan_id', data = scan_id)
        hf.create_dataset('note', data = note)
        hf.create_dataset('scan_time', data = scan_time)
        hf.create_dataset('X_eng', data = eng_list)
        hf.create_dataset('img_bkg', data = img_bkg_avg)
        hf.create_dataset('img_dark', data = img_dark_avg)
        hf.create_dataset('img_xanes', data = img_xanes_norm)
    del img_xanes, img_dark, img_bkg, img_xanes_avg, img_dark_avg
    del img_bkg_avg, imgs, img_xanes_norm


def load_z_scan(h):
    scan_type = h.start['plan_name']
    scan_id = h.start['scan_id']
    uid = h.start['uid']
    num = h.start['plan_args']['steps']
    chunk_size = h.start['plan_args']['chunk_size']
    note = h.start['plan_args']['note'] if h.start['plan_args']['note'] else 'None'
    img = np.array(list(h.data('Andor_image')))
    img_zscan = np.mean(img[:num], axis=1)
    img_bkg = np.mean(img[num], axis=0, keepdims=True)
    img_dark = np.mean(img[-1], axis=0, keepdims=True)
    img_norm = (img_zscan - img_dark) / (img_bkg - img_dark)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0
#    fn = h.start['plan_args']['fn']
    fname = scan_type + '_id_' + str(scan_id) + '.h5'
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('uid', data = uid)
        hf.create_dataset('scan_id', data = scan_id)
        hf.create_dataset('note', data = note)
        hf.create_dataset('img_bkg', data = img_bkg)
        hf.create_dataset('img_dark', data = img_dark)
        hf.create_dataset('img', data = img_zscan)
        hf.create_dataset('img_norm', data=img_norm)
    del img, img_zscan, img_bkg, img_dark, img_norm

    
def load_test_scan(h):
    import tifffile
    scan_type = h.start['plan_name']
    scan_id = h.start['scan_id']
    uid = h.start['uid']   
    num = h.start['plan_args']['num_img']
    num_bkg = h.start['plan_args']['num_bkg']
    note = h.start['plan_args']['note'] if h.start['plan_args']['note'] else 'None'
    img = np.squeeze(np.array(list(h.data('Andor_image'))))
    assert len(img.shape) == 3, 'load test_scan fails...'
    img_test = img[:num]
    img_bkg = np.mean(img[num: num + num_bkg], axis=0, keepdims=True)
    img_dark = np.mean(img[-num_bkg:], axis=0, keepdims=True)
    img_norm = (img_test - img_dark) / (img_bkg - img_dark)
    img_norm[np.isnan(img_norm)] = 0
    img_norm[np.isinf(img_norm)] = 0
#    fn = h.start['plan_args']['fn']
    fname = scan_type + '_id_' + str(scan_id) + '.h5'
    fname_tif = scan_type + '_id_' + str(scan_id) + '.tif'
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('uid', data = uid)
        hf.create_dataset('scan_id', data = scan_id)
        hf.create_dataset('note', data = note)
        hf.create_dataset('img_bkg', data = img_bkg)
        hf.create_dataset('img_dark', data = img_dark)
        hf.create_dataset('img', data = img_test)
        hf.create_dataset('img_norm', data=img_norm)
#    tifffile.imsave(fname_tif, img_norm)
    del img, img_test, img_bkg, img_dark, img_norm



def load_count_img(scan_id, fn='img_test.h5'):
    '''
    load images (e.g. RE(count([Andor], 10)) ) and save to .h5 file
    '''    
    uid = h.start['uid']
    img = get_img(h)
    scan_id = h.start['scan_id']
    fn = 'count_id_' + str(scan_id) + '.h5'
    with h5py.File(fn, 'w') as hf:
        hf.create_dataset('img',data=img)
        hf.create_dataset('uid',data=uid)
        hf.create_dataset('scan_id',data=scan_id)



def load_multipos_count(h):
    scan_type = h.start['plan_name']
    scan_id = h.start['scan_id']
    uid = h.start['uid']
    num_dark = h.start['num_dark_images']
    num_of_position = h.start['num_of_position']
    note = h.start['note']

    img_raw = list(h.data('Andor_image'))
    img_dark = np.squeeze(np.array(img_raw[:num_dark]))
    img_dark_avg = np.mean(img_dark, axis=0, keepdims=True)
    num_repeat = np.int((len(img_raw) - 10)/num_of_position/2) # alternatively image and background

    tot_img_num = num_of_position * 2 * num_repeat
    s = img_dark.shape
    img_group = np.zeros([num_of_position, num_repeat, s[1], s[2]], dtype=np.float32)

    for j in range(num_repeat):
        index = num_dark + j*num_of_position*2
        print(f'processing #{index} / {tot_img_num}' )
        for i in range(num_of_position):
            tmp_img = np.array(img_raw[index+i*2])
            tmp_bkg = np.array(img_raw[index+i*2+1])
            img_group[i, j] = (tmp_img - img_dark_avg)/(tmp_bkg - img_dark_avg)
    fn = os.getcwd() + '/'
    fname = fn + scan_type + '_id_' + str(scan_id) + '.h5'
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('uid', data = uid)
        hf.create_dataset('scan_id', data = scan_id)
        hf.create_dataset('note', data = note)
        for i in range(num_of_position):
            hf.create_dataset(f'img_pos{i+1}', data=np.squeeze(img_group[i])) 


def load_grid2D_rel(h):
    uid = h.start['uid']
    note = h.start['note']
    scan_type = 'grid2D_rel'
    scan_id = h.start['scan_id']   
    scan_time = h.start['time'] 
    x_eng = h.start['XEng']
    num1 = h.start['plan_args']['num1']
    num2 = h.start['plan_args']['num2']
    img = np.squeeze(np.array(list(h.data('Andor_image'))))
    
    fname= scan_type + '_id_' + str(scan_id)
    cwd = os.getcwd()
    try:
        os.mkdir(cwd+f'/{fname}')
    except:
        print(cwd+f'/{name} existed')
    fout= cwd+f'/{fname}' 
    for i in range(num1):
        for j in range(num2):
            fname_tif = fout + f'_({ij}).tif'
            img = Image.fromarray(img[i*num1+j])
            img.save(fname_tif)

    
def load_raster_2D(h):
    uid = h.start['uid']
    note = h.start['note']
    scan_type = 'grid2D_rel'
    scan_id = h.start['scan_id']   
    scan_time = h.start['time'] 
    num_dark = h.start['num_dark_images']
    num_bkg = h.start['num_bkg_images']
    x_eng = h.start['XEng']
    x_range = h.start['plan_args']['x_range']
    y_range = h.start['plan_args']['y_range']
    img_sizeX = h.start['plan_args']['img_sizeX']
    img_sizeY = h.start['plan_args']['img_sizeY']

    img_raw = np.squeeze(np.array(list(h.data('Andor_image'))))
    img_dark_avg = np.mean(img_raw[:num_dark], axis=0, keepdims=True)
    img_bkg_avg = np.mean(img_raw[-num_bkg:], axis=0, keppdims = True)
    img = img_raw[num_dark:-num_bkg]
    s = img.shape

    row_size = len(y_range) * s[1]
    col_size = len(x_range) * s[2]
    img_patch = np.zeros([1, row_size, col_size])
    index = 0
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            img_patch[1, j*s[1]:(j+1)*s[1], i*s[2]:(i+1)*s[2]] = img[index]
            index = index + 1
    s = img_patch.shape
    img_patch_bin = bin_ndarray(img_patch, new_shape=(s[0], s[1]/4, s[2]/4))

    fout = f'raster2D_scan_{scan_id}.tif'
    
    
    
    

    

################################################################
####################   plot scaler  ############################
################################################################


def plot_ic(scan_id=[-1], ics=[ic3, ic4]):
    '''
    plot ic reading from single/multiple scan(s),
    e.g. plot_ic([-1, -2],['ic3', 'ic4'])
    '''
    if type(scan_id) == int:
        scan_id = [scan_id]
    plt.figure()
    for sid in scan_id:
        h = db[int(sid)]
        sid = h.start['scan_id']
        num =  int(h.start['plan_args']['num'])
        try:
            st = h.start['plan_args']['start']
            en = h.start['plan_args']['stop']
        except:
            try:
                st = h.start['plan_args']['args'][1]
                en = h.start['plan_args']['args'][2]
            except:
                st = 1
                en = num
        x = np.linspace(st, en, num)
        y = []
        for ic in ics:
            y = (list(h.data(ic.name)))
            plt.plot(x, y, '.-', label=f'scan: {sid}, ic: {ic.name}')
            plt.legend()
    plt.title(f'reading of ic: {[ic.name for ic in ics]}')
    plt.show()



def plot2dsum(scan_id = -1, fn='Det_Image', save_flag=0):
    '''
    valid only if the scan using Andor or detA1 camera
    '''
    h = db[scan_id]
    if scan_id == -1:
        scan_id = h.start['scan_id']
    if 'Andor' in h.start['detectors']:
        det = 'Andor_image'
        find_areaDet = 1
    elif 'detA1' in h.start['detectors']:
        det = 'detA1_image'
        find_areaDet = 1
    else:
        find_areaDet = 0

    if find_areaDet:        
        img = np.array(list(h.data(det)))
        if len(img.shape) == 4:
            img = np.mean(img, axis=1)
        img[img<20] = 0
        img_sum = np.sum(np.sum(img, axis=1), axis=1)
        num =  int(h.start['plan_args']['steps'])
        st = h.start['plan_args']['start']
        en = h.start['plan_args']['stop']
        x = np.linspace(st, en, num)
        plt.figure()
        plt.plot(x, img_sum, 'r-.')
        plt.title('scanID: ' + str(scan_id) + ':  ' + det + '_sum')

        if save_flag:
            with h5py.File(fn, 'w') as hf:
                hf.create_dataset('data', data=img)
    else:
        print('AreaDetector is not used in the scan')
       


def plot1d(scan_id=-1, detectors=[], plot_time_stamp=0):
    h = db[scan_id]
    scan_id = h.start['scan_id']
    n = len(detectors)
    if n == 0:
        detectors = h.start['detectors']
        n = len(detectors)
    
#    y = list(h.data(detectors[0].name))
#    x = np.arange(len(y))
    pos = h.table()
    try:
        st = h.start['plan_args']['start']
        en = h.start['plan_args']['stop']
        num =  int(h.start['plan_args']['steps'])
        flag = 0
    except:
        flag = 1
    if flag or plot_time_stamp:
        mot_day, mot_hour = pos['time'].dt.day, pos['time'].dt.hour, 
        mot_min, mot_sec, mot_msec = pos['time'].dt.minute, pos['time'].dt.second, pos['time'].dt.microsecond
        mot_time = mot_day * 86400 + mot_hour * 3600 + mot_min * 60 + mot_sec + mot_msec * 1e-6
        mot_time =  np.array(mot_time)   
        x = mot_time - mot_time[0]   
    else:
        x = np.linspace(st, en, num)
    
    fig=plt.figure()
    for i in range(n):
        det_name = detectors[i].name
        if detectors[i].name == 'detA1' or detectors[i].name == 'Andor':
            det_name = det_name + '_stats1_total'
        y = list(h.data(det_name))
        ax=fig.add_subplot(n,1,i+1);ax.plot(x, y); ax.title.set_text(det_name)
    plt.xlabel('time (s)')
    fig.subplots_adjust(hspace=1)   
    plt.show()


################################################################
####################    read and save  #########################
################################################################


def readtiff(fn_pre='', num=1, x=[], bkg=0, roi=[]):
    if len(x) == 0: 
        x = np.arange(num) 
    fn = fn_pre + '_' + '{:03d}'.format(1) + '.tif'
    img = np.array(Image.open(fn))
    s = img.shape
    if len(roi) == 0:
        roi = [0, s[0], 0 , s[1]]
    img_stack = np.zeros([num, s[0], s[1]])
    img_stack[0] = img
    for i in range(1, num):
        fn = fn_pre + '_' + '{:03d}'.format(i+1) + '.tif'
        img_stack[i] = Image.open(fn)
    bkg = bkg * s[0] * s[1]
    img_stack_roi = img_stack[:, roi[0]:roi[1], roi[2]:roi[3]]
    img_sum = np.sum(np.sum(img_stack_roi, axis=1), axis=1)
    img_sum = img_sum - bkg
    plt.figure()
    plt.plot(x, img_sum, '.-')
    return img_stack, img_stack_roi

def save_hdf(img, fn='img.h5',f_dir='/home/xf18id/Documents/FXI_commision/'):
    f = f_dir + fn
    with h5py.File(f, 'w') as hf:
        hf.create_dataset('data', data = img)


################################################################
###################   tomography ###############################
################################################################

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



########################################################################
########################################################################

def print_baseline_list():
    a = list(db[-1].table('baseline'))
    with open('/home/xf18id/Documents/FXI_manual/FXI_baseline_record.txt', 'w') as tx:
         i=1
         for txt in a:
             if not i%3:
                 tx.write('\n')
             tx.write(f'\t{txt:<30}')
             i +=1 
    




def get_scan_timestamp(scan_id):
    h=db[scan_id]
    scan_id = h.start['scan_id']
    timestamp = h.start['time']
    timestamp_conv=convert_AD_timestamps(pd.Series(timestamp))
    scan_year = int(timestamp_conv.dt.year)
    scan_mon = int(timestamp_conv.dt.month)
    scan_day = int(timestamp_conv.dt.day)
    scan_day, scan_hour = int(timestamp_conv.dt.day), int(timestamp_conv.dt.hour)
    scan_min, scan_sec, scan_msec = int(timestamp_conv.dt.minute), int(timestamp_conv.dt.second), int(timestamp_conv.dt.microsecond)
    scan_time = f'scan#{scan_id}: {scan_year-20:04d}-{scan_mon:02d}-{scan_day:02d}   {scan_hour:02d}:{scan_min:02d}:{scan_sec:02d}'
    print(scan_time)




def get_scan_file_name(scan_id):
    hdr = db[scan_id]

#    print(scan_id, hdr.stop['exit_status'])  
    res_uids = list(db.get_resource_uids(hdr))
    for i, uid in enumerate(res_uids):
        res_doc = db.reg.resource_given_uid(uid)
#        print("   ", i, res_doc)
    fpath_root = res_doc['root']
    fpath_relative = res_doc['resource_path']
    fpath = fpath_root + '/' + fpath_relative

    fpath_remote = '/nsls2/xf18id1/backup/DATA/Andor/' + fpath_relative
    return print(f'local path: {fpath}\nremote path: {fpath_remote}')



read_calib_file()


