import numpy as np
import matplotlib.pylab as plt
import h5py
from operator import attrgetter
from PIL import Image
from scipy.signal import medfilt2d


def record_calib_pos1():
    global CALIBER_FLAG, CURRENT_MAG_1, CURRENT_MAG_2
    #CALIBER['th2_pos1'] = pzt_dcm_th2.pos.value
    CALIBER['chi2_pos1'] = pzt_dcm_chi2.pos.value
    CALIBER['XEng_pos1'] = XEng.position
    CALIBER['zp_x_pos1'] = zp.x.position
    CALIBER['zp_y_pos1'] = zp.y.position
    CALIBER['th2_motor_pos1'] = th2_motor.position
    
    #print(f'pzt_dcm_th2_{CALIBER["XEng_pos1"]:2.4f}\t: {CALIBER["th2_pos1"]:2.4f}')
    print(f'pzt_dcm_chi2_{CALIBER["XEng_pos1"]:2.4f}\t: {CALIBER["chi2_pos1"]:2.4f}')
    print(f'zp_x_{CALIBER["XEng_pos1"]:2.4f}\t\t: {CALIBER["zp_x_pos1"]:2.4f}')
    print(f'zp_y_{CALIBER["XEng_pos1"]:2.4f}\t\t: {CALIBER["zp_y_pos1"]:2.4f}')
    print(f'th2_motor_{CALIBER["XEng_pos1"]:2.4f}\t: {CALIBER["th2_motor_pos1"]:2.6f}')

    df = pd.DataFrame.from_dict(CALIBER, orient='index')
    df.to_csv('/home/xf18id/.ipython/profile_collection/startup/calib.csv', sep='\t')
    CURRENT_MAG_1 = (DetU.z.position / zp.z.position - 1) * GLOBAL_VLM_MAG
    if np.abs(CURRENT_MAG_1 - CURRENT_MAG_2) > 0.1:
        print('MAGNIFICATION between two calibration points does not match!!')
        print(f'MAGNIFICATION_1 = {CURRENT_MAG_1}\nMAGNIFICATION_2 = {CURRENT_MAG_2}')
        CALIBER_FLAG = 0
    else:
        
        CALIBER_FLAG = 1
    GLOBAL_MAG = np.round(CURRENT_MAG_1 * 100) / 100.
    print(f'calib_pos1 recored: current Magnification = {CURRENT_MAG_1}')


def record_calib_pos2():
    global CALIBER_FLAG, CURRENT_MAG_1, CURRENT_MAG_2
    #CALIBER['th2_pos2'] = pzt_dcm_th2.pos.value
    CALIBER['chi2_pos2'] = pzt_dcm_chi2.pos.value
    CALIBER['XEng_pos2'] = XEng.position
    CALIBER['zp_x_pos2'] = zp.x.position
    CALIBER['zp_y_pos2'] = zp.y.position
    CALIBER['th2_motor_pos2'] = th2_motor.position
    #print(f'pzt_dcm_th2_{CALIBER["XEng_pos2"]:2.4f}\t: {CALIBER["th2_pos2"]:2.4f}')
    print(f'pzt_dcm_chi2_{CALIBER["XEng_pos2"]:2.4f}\t: {CALIBER["chi2_pos2"]:2.4f}')
    print(f'zp_x_{CALIBER["XEng_pos2"]:2.4f}\t\t: {CALIBER["zp_x_pos2"]:2.4f}')
    print(f'zp_y_{CALIBER["XEng_pos2"]:2.4f}\t\t: {CALIBER["zp_y_pos2"]:2.4f}')
    print(f'th2_motor_{CALIBER["XEng_pos2"]:2.4f}\t: {CALIBER["th2_motor_pos2"]:2.6f}')
 
    df = pd.DataFrame.from_dict(CALIBER, orient='index')
    df.to_csv('/home/xf18id/.ipython/profile_collection/startup/calib.csv', sep='\t')
    CURRENT_MAG_2 = (DetU.z.position / zp.z.position - 1) * GLOBAL_VLM_MAG
    if np.abs(CURRENT_MAG_1 - CURRENT_MAG_2) > 0.1:
        print('MAGNIFICATION between two calibration points does not match!!')
        print(f'MAGNIFICATION_1 = {CURRENT_MAG_1}\nMAGNIFICATION_2 = {CURRENT_MAG_2}')
        CALIBER_FLAG = 0
    else:
        
        CALIBER_FLAG = 1

    GLOBAL_MAG = np.round(CURRENT_MAG_2 * 100) / 100.
    print(f'calib_pos2 recored: current Magnification = {CURRENT_MAG_2}')


def read_calib_file():
    fn = '/home/xf18id/.ipython/profile_collection/startup/calib.csv'
    try:
        df = pd.read_csv(fn, index_col=0, sep='\t')
        d = df.to_dict("split")
        d = dict(zip(d["index"], d["data"]))
        #CALIBER['th2_pos1'] = np.float(d['th2_pos1'][0])
        CALIBER['chi2_pos1'] = np.float(d['chi2_pos1'][0])
        CALIBER['XEng_pos1'] = np.float(d['XEng_pos1'][0])
        CALIBER['zp_x_pos1'] = np.float(d['zp_x_pos1'][0])
        CALIBER['zp_y_pos1'] = np.float(d['zp_y_pos1'][0])
        CALIBER['th2_motor_pos1'] = np.float(d['th2_motor_pos1'][0])

        #CALIBER['th2_pos2'] = np.float(d['th2_pos2'][0])
        CALIBER['chi2_pos2'] = np.float(d['chi2_pos2'][0])
        CALIBER['XEng_pos2'] = np.float(d['XEng_pos2'][0])
        CALIBER['zp_x_pos2'] = np.float(d['zp_x_pos2'][0])
        CALIBER['zp_y_pos2'] = np.float(d['zp_y_pos2'][0])
        CALIBER['th2_motor_pos2'] = np.float(d['th2_motor_pos2'][0])
    
        #print(f'pzt_dcm_th2_{CALIBER["XEng_pos1"]:2.4f}\t: {CALIBER["th2_pos1"]:2.4f}')
        print(f'pzt_dcm_chi2_{CALIBER["XEng_pos1"]:2.4f}\t: {CALIBER["chi2_pos1"]:2.4f}')
        print(f'zp_x_{CALIBER["XEng_pos1"]:2.4f}\t\t: {CALIBER["zp_x_pos1"]:2.4f}')
        print(f'zp_y_{CALIBER["XEng_pos1"]:2.4f}\t\t: {CALIBER["zp_y_pos1"]:2.4f}')
        print(f'th2_motor_{CALIBER["XEng_pos1"]:2.4f}\t: {CALIBER["th2_motor_pos1"]:2.6f}')
        print('\n')
        #print(f'pzt_dcm_th2_{CALIBER["XEng_pos2"]:2.4f}\t: {CALIBER["th2_pos2"]:2.4f}')
        print(f'pzt_dcm_chi2_{CALIBER["XEng_pos2"]:2.4f}\t: {CALIBER["chi2_pos2"]:2.4f}')
        print(f'zp_x_{CALIBER["XEng_pos2"]:2.4f}\t\t: {CALIBER["zp_x_pos2"]:2.4f}')
        print(f'zp_y_{CALIBER["XEng_pos2"]:2.4f}\t\t: {CALIBER["zp_y_pos2"]:2.4f}')
        print(f'th2_motor_{CALIBER["XEng_pos2"]:2.4f}\t: {CALIBER["th2_motor_pos2"]:2.6f}')
    except:
        print(f'\nreading calibration file: {fn} fails...\n Please optimize optics at two energy points, and using record_calib_pos1() and record_calib_pos2() after optimizing each energy points ')


def show_global_para():
    print(f'GLOBAL_MAG = {GLOBAL_MAG} X') # total magnification
    print(f'GLOBAL_VLM_MAG = {GLOBAL_VLM_MAG} X') # vlm magnification
    print(f'OUT_ZONE_WIDTH = {OUT_ZONE_WIDTH} nm') # 30 nm
    print(f'ZONE_DIAMETER = {ZONE_DIAMETER} um') # 200 um
    print(f'CURRENT_MAG_1 = {CURRENT_MAG_1} X') # calibration magnification at pos 1
    print(f'CURRENT_MAG_2 = {CURRENT_MAG_2} X') # calibration magnification at pos 1
    print(f'\nFor Andor camera, current pixel size = {6500./GLOBAL_MAG:3.1f} nm')
    print('\nChange parameters if necessary.\n\n')


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

    pre = f'/NSLS2/xf18id1/users/{year}{qut}/'
    try:        os.mkdir(pre)
    except Exception:    pass
    print('\n')

    PI_name = input('PI\'s name:')  
    PI_name = PI_name.replace(' ', '_').upper()

    if PI_name[0] == '*':
        cwd = os.getcwd()
        print(f'stay at current directory: {cwd}\n')
        return
    if PI_name[:4] == 'COMM':
        PI_name = 'FXI_commission'
        fn = pre + PI_name
        print(fn)
    else:
        proposal_id = input('Proposal ID:')
        fn = pre + PI_name + '_Proposal_' + proposal_id
        export_pdf(1)
        insert_txt('New user: {fn}\n')
        export_pdf(1)
        
    try:        
        cmd = f'mkdir -m 777 {fn}'
        os.system(cmd)
        #os.mkdir(fn)
    except Exception:
        print('Found (user, proposal) existed\nEntering folder: {}'.format(os.getcwd()))
        os.chdir(fn)       
        pass
    os.chdir(fn)
    print ('\nUser creating successful!\n\nEntering folder: {}\n'.format(os.getcwd()))


################################################################
####################   TXM paramter  ###########################
################################################################


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
        
        #pzt_dcm_th2_eng1 = CALIBER['th2_pos1']
        pzt_dcm_chi2_eng1 = CALIBER['chi2_pos1']
        zp_x_pos_eng1 = CALIBER['zp_x_pos1']
        zp_y_pos_eng1 = CALIBER['zp_y_pos1']
        th2_motor_eng1 = CALIBER['th2_motor_pos1']


        #pzt_dcm_th2_eng2 = CALIBER['th2_pos2']
        pzt_dcm_chi2_eng2 = CALIBER['chi2_pos2']
        zp_x_pos_eng2 = CALIBER['zp_x_pos2']
        zp_y_pos_eng2 = CALIBER['zp_y_pos2']
        th2_motor_eng2 = CALIBER['th2_motor_pos2']

        if np.abs(eng1 - eng2) < 1e-5: # difference less than 0.01 eV
            print(f'eng1({eng1:2.5f} eV) and eng2({eng2:2.5f} eV) in "CALIBER" are two close, will not move any motors...')
        else:
            #pzt_dcm_th2_target = (eng_new - eng2) * (pzt_dcm_th2_eng1 - pzt_dcm_th2_eng2) / (eng1-eng2) + pzt_dcm_th2_eng2
            pzt_dcm_chi2_target = (eng_new - eng2) * (pzt_dcm_chi2_eng1 - pzt_dcm_chi2_eng2) / (eng1-eng2) + pzt_dcm_chi2_eng2
            zp_x_target = (eng_new - eng2)*(zp_x_pos_eng1 - zp_x_pos_eng2)/(eng1 - eng2) + zp_x_pos_eng2
            zp_y_target = (eng_new - eng2)*(zp_y_pos_eng1 - zp_y_pos_eng2)/(eng1 - eng2) + zp_y_pos_eng2
            th2_motor_target = (eng_new - eng2) * (th2_motor_eng1 -th2_motor_eng2) / (eng1-eng2) + th2_motor_eng2

            #pzt_dcm_th2_ini = pzt_dcm_th2.pos.value
            pzt_dcm_chi2_ini = pzt_dcm_chi2.pos.value
            zp_x_ini = zp.x.position    
            zp_y_ini = zp.y.position
            th2_motor_ini = th2_motor.position
        
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
                yield from mv(zp.x, zp_x_target, zp.y, zp_y_target)                
                yield from mv(th2_feedback_enable, 0)
                yield from mv(th2_feedback, th2_motor_target)
                yield from mv(th2_feedback_enable, 1)
                yield from mv(zp.z, zp_final,det.z, det_final, XEng, eng_new)                
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
                #print ('will move pzt_dcm_th2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_th2_ini, pzt_dcm_th2_target))
                print ('will move pzt_dcm_chi2: ({0:2.4f} um --> {1:2.4f} um)'.format(pzt_dcm_chi2_ini, pzt_dcm_chi2_target))
                print ('will move th2_motor: ({0:2.6f} deg --> {1:2.6f} deg)'.format(th2_motor_ini, th2_motor_target))
    else:
        print('record_calib_pos1() or record_calib_pos2() not excuted successfully...\nWill not move anything')



#def cal_phase_ring_position(eng_new, eng_ini=0, print_flag=1):
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


#def move_phase_ring(eng_new, eng_ini, flag=1):
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



    
    

    

################################################################
####################   plot scaler  ############################
################################################################


def plot_ic(scan_id=[-1], ics=[]):
    '''
    plot ic reading from single/multiple scan(s),
    e.g. plot_ic([-1, -2],['ic3', 'ic4'])
    '''
    if type(scan_id) == int:
        scan_id = [scan_id]

    if len(ics) == 0:
        ics = [ic3, ic4]

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
        det_name = detectors[i]
        if det_name == 'detA1' or det_name == 'Andor':
            det_name = det_name + '_stats1_total'
        y = list(h.data(det_name))
        title_txt = f'scan#{scan_id}:   {det_name}'
        ax=fig.add_subplot(n,1,i+1);ax.plot(x, y); ax.title.set_text(title_txt)
    if flag:
        plt.xlabel('time (s)')
    else:
        plt.xlabel(f'{h.start["plan_args"]["motor"]} position')
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



def save_hdf_file(fn, *args):
    n = len(args)
    assert n%2 == 0, 'even number of args only'
    n = int(n/2)
    j = 0
    with h5py.File(fn, 'w') as hf:
        for i in range(n):
            j = int(2*i)
            tmp = args[j+1]
            hf.create_dataset(args[j], data=tmp)



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


def get_img(h, det='Andor', sli=[]):
    "Take in a Header and return a numpy array of detA1 image(s)."
    det_name = f'{det}_image'
    if len(sli) == 2:
        img = np.array(list(h.data('det_name'))[sli[0]:sli[1]])      
    else:
        img = np.array(list(h.data('det_name')))
    return img


def get_scan_parameter(scan_id=-1, print_flag=0):
    h=db[scan_id]
    scan_id = h.start['scan_id']
    uid = h.start['uid']
    try:
        X_eng = f'{h.start["XEng"]:2.4f}'
    except:
        X_eng = 'n/a'
    scan_type = h.start['plan_name']
    scan_time = datetime.fromtimestamp(h.start['time']).strftime('%D %H:%M')

    txt = ''
    for key, val in h.start['plan_args'].items():
        if key == 'zone_plate':
            continue
        txt += f'{key}={val}, '
    txt0 = f'#{scan_id}  (uid: {uid[:6]},  X_Eng: {X_eng} keV,  Time: {scan_time})\n'
    txt = txt0 + scan_type + f'({txt[:-2]})\n'
    try:        
        txt_tmp = ''
        for zone_plate_key in h.start['plan_args']['zone_plate'].keys():
            txt_tmp += f'{zone_plate_key}: {val[zone_plate_key]};    ' 
        txt = txt + 'Zone Plate info:  ' + txt_tmp
    except:
        pass
    if print_flag:
        print(txt)
    return txt



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



def get_scan_motor_pos(scan_id): 
    df = db[scan_id].table('baseline').T 
    mot = BlueskyMagics.positioners 
    for i in mot: 
        try:
            mot_name = i.name
            if mot_name[:3] == 'pzt':
                print(f'{mot_name:>16s}  :: {df[1][mot_name]: 14.6f}       --->  {df[2][mot_name]: 12.6f}' )
            else:
                mot_parent_name = i.parent.name
                offset_name = f'{mot_name}_user_offset'
                offset_dir = f'{mot_name}_user_offset_dir'
                offset_val = db[scan_id].config_data(mot_parent_name)["baseline"][0][offset_name]
                offset_dir_val =  db[scan_id].config_data(mot_parent_name)["baseline"][0][offset_dir]
                print(f'{mot_name:>16s}  :: {df[1][mot_name]: 14.6f} {i.motor_egu.value:>4s}  --->  {df[2][mot_name]: 14.6f} {i.motor_egu.value:>4s}      offset = {offset_val: 14.6f}    {offset_dir_val: 1d}') 
        except: 
            pass 




class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        self._indx_txt = ax.set_title(' ', loc='center')
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind, :, :], cmap='gray')
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        #self.ax.set_ylabel('slice %s' % self.ind)
        self._indx_txt.set_text(f"frame {self.ind + 1} of {self.slices}")
        self.im.axes.figure.canvas.draw()

    
def image_scrubber(data, *, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    tracker = IndexTracker(ax, data)
    # monkey patch the tracker onto the figure to keep it alive
    fig._tracker = tracker
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return tracker


read_calib_file()


