import numpy as np
import matplotlib.pylab as plt
import h5py

from operator import attrgetter


GLOBAL_MAG = 650 # total magnification
GLOBAL_VLM_MAG = 10 # vlm magnification
OUT_ZONE_WIDTH = 30 # 30 nm
ZONE_DIAMETER = 100 # 200 um


def list_fun():
    import umacro
    all_func = inspect.getmembers(umacro, inspect.isfunction)
    return all_func     


def check_eng_range(eng):
    '''
    check energy in range of 4000-12000
    Inputs:
    --------
    eng: list
        e.g. [6000,7500]
    '''
    eng = list(eng)
    high_limit = 12000
    low_limit = 4000
    for i in range(len(eng)):
        assert(eng[i] >= low_limit and eng[i] <= high_limit), 'Energy is outside the range (4000, 12000) eV'
    return 
        
def cal_parameter(eng, print_flag=1):
    '''
    Calculate parameters for given X-ray energy
    Use as: wave_length, focal_length, NA, DOF = energy_cal(Eng, print_flag=1):

    Inputs:
    -------
    eng: float
         X-ray energy, in unit of eV
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

    if eng < 4000:    eng = XEng.position * 1000 # current beam energy
    check_eng_range([eng])

    wave_length = h * c / (ec * eng) * 1e9 # nm
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
          User defined energy, in unit of eV
    eng_ini:  float
          if eng_ini < 4000 (eV), will eng_ini = current Xray energy 
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
    
    if eng_ini < 4000:    
        eng_ini = XEng.position * 1000 # current beam energy
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

    zp_final = zp_ini + zp_delta
    det_final = det_ini + det_delta

    if print_flag:    
        print ('Calculation results:')
        print ('Change energy from: {0:2.2f} eV to {1:2.2f} eV'.format(eng_ini, eng_new))
        print ('Need to move zone plate by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)'
                .format(zp_delta, zp_ini, zp_final))
        print ('Need to move CCD by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)'
                .format(det_delta, det_ini, det_final)) 
    else:
        return zp_ini, det_ini, zp_delta, det_delta, zp_final, det_final


def move_zp_ccd(eng_new, eng_ini=0, flag=1):
    '''
    move the zone_plate and ccd to the user-defined energy with constant magnification
    use the function as:
        move_zp_ccd_with_const_mag(eng_new=8000, eng_ini=9000, flag=1):

    Inputs:
    -------
    eng_new:  float 
          User defined energy, in unit of eV
    eng_ini:  float
          if eng_ini < 3000 (eV), will eng_ini = current Xray energy 
    flag: int
          0: Do calculation without moving real stages
          1: Will move stages
    '''

    det = DetU # upstream detector
    zp_ini, det_ini, zp_delta, det_delta, zp_final, det_final = cal_zp_ccd_position(eng_new, eng_ini, print_flag=0)

    assert ((det_final) > det.z.low_limit and (det_final) < det.z.high_limit), print ('Trying to move DetU to {0:2.2f}. Movement is out of travel range ({1:2.2f}, {2:2.2f})\nTry to move the bottom stage manually.'.format(det_final, det.z.low_limit, det.z.high_limit))

    if flag: # move stages
        print ('Now moving stages ....')
        # RE(mv(XEng.position, Eng) # uncomment it when doing experiment        
        RE(mvr(zp.z, zp_delta))
        print ('zone plate position: {0:2.4f} mm --> {1:2.4f} mm'.format(zp_ini, zp.z.position))
        RE(mvr(det.z, det_delta))
        print ('CCD position: {0:2.4f} mm --> {1:2.4f} mm'.format(det_ini, det.z.position))
    else:
        print ('This is calculation. No stages move') 
        print ('will move zone plate down stream by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)'.format(zp_delta, zp_ini, zp_final))
        print ('will move CCD down stream by: {0:2.4f} mm ({1:2.4f} mm --> {2:2.4f} mm)'.format(det_delta, det_ini, det_final))
    

def cal_phase_ring_position(eng_new, eng_ini=0, print_flag=1):
    '''
    calculate delta amount of phase_ring movement:
    positive means move down-stream, negative means move up-stream
    
    use as:
        cal_phase_ring_with_const_mag(eng_new=8000, eng_ini=9000)

    Inputs:
    -------
    eng_new: float
        target energy, in unit of eV
    eng_ini: float
        initial energy, in unit of eV
        it will read current Xray energy if eng_ini < 4000

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
        move_phase_ring_with_const_mag(eng_new=8000, eng_ini=9000, flag=1)

    Inputs:
    -------
    eng_new: float
        target energy, in unit of eV
    eng_ini: float
        initial energy, in unit of eV
        it will read current Xray energy if eng_ini < 4000
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
    E.g. RE(go_eng(9000)) # Note that: energy is in unit of eV
    '''
    check_eng_range([eng])
    yield from abs_set(XEng, eng/1000, wait=True)

#####################################################################
def load_image(args):
    '''
    e.g. load_image() # case 1
         load_image(120) # case 2
         load_image([120,122,125]) # case 3: load three scans
         load_image(120, 125)      # case 4: load scans from scan_120 to scan_125
         load_image(120, 125, 2)   # case 5: load scans from scan_120 to scan_125 for every 2 scans
    '''

    scan_id = list(args) # case 2 and 3
    if len(args) == 0:      scan_id = [-1] # case 1
    elif len(args) == 2:    scan_id = np.arange(args[0], args[1]+1) # case 4
    elif len(args) == 3:    scan_id = np.arange(args[0], args[1]+1, args[2]) # case 5
    else: return 'Invalid input'
    for item in scan_id:        
       load_single_image(item)    

def load_single_image(scan_id=-1):
    h = db[scan_id]
    scan_type = h.start['plan_name']
    x_eng = h.start['x_ray_energy']
     
    if scan_type == 'tomo_scan':
        load_tomo_scan(h)
    elif scan_type == 'fly_scan':
        load_fly_scan(h)
    else: 
        pass
        

def load_tomo_scan(h):
    scan_type = 'tomo_scan'
    scan_id = h.start['scan_id']
    x_eng = h.start['x_ray_energy']
    bkg_img_num = h.start['back_ground_images']
    angle_i = h.start['plan_args']['start']
    angle_e = h.start['plan_args']['stop']
    angle_n = h.start['plan_args']['num'] 
    exposure_t = h.start['plan_args']['exposure_time'] 
    img = np.array(list(h.data('detA1_image')))
    img = np.squeeze(img)

    img_bkg = img[0 : bkg_img_num]
    img_tomo = img[bkg_img_num :]
    img_angle = np.linspace(angle_i, angle_e, angle_n)
        
    fname = scan_type + '_id_' + str(scan_id) + '.h5'
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('X_eng', data = x_eng)
        hf.create_dataset('img_bkg', data = img_bkg)
        hf.create_dataset('img_tomo', data = img_tomo)
        hf.create_dataset('angle', data = img_angle)

def load_fly_scan(h): 
    scan_type = 'fly_scan'
    scan_id = h.start['scan_id']    
    x_eng = h.start['x_ray_energy']
    chunk_size = h.start['chunk_size']
    # sanity check: make sure we remembered the right stream name
    assert 'zps_sx_monitor' in h.stream_names
    pos = h.table('zps_sx_monitor')
    imgs = np.array(list(h.data('Andor_image')))
    img_dark = imgs[0]
    img_bkg = imgs[-1]

    s = imgs.shape
    imgs =imgs.reshape([s[0]*s[1], s[2], s[3]])

    with db.reg.handler_context({'AD_HDF5': AreaDetectorHDF5TimestampHandler}):
        chunked_timestamps = list(h.data('Andor_image'))

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

    mot_pos = np.array(pos['zps_sx'])
    offset = np.min([np.min(img_time), np.min(mot_time)])
    img_time -= offset
    mot_time -= offset
    mot_pos_interp = np.interp(img_time, mot_time, mot_pos)
    
    mot_pos_min = np.min(mot_pos_interp)
    mot_pos_max = np.max(mot_pos_interp)

    pos1 = len(mot_pos_interp[np.abs(mot_pos_interp - mot_pos_min) < 0.1]) 
    pos2 = len(mot_pos_interp[np.abs(mot_pos_interp - mot_pos_max) < 0.1]) 

    img_angle = mot_pos_interp[pos1:-pos2] # rotation angles
    img_tomo = imgs[pos1:-pos2]  # tomo images
    
    fname = scan_type + '_id_' + str(scan_id) + '.h5'
    with h5py.File(fname, 'w') as hf:
        hf.create_dataset('X_eng', data = x_eng)
        hf.create_dataset('img_bkg', data = img_bkg)
        hf.create_dataset('img_dark', data = img_dark)
        hf.create_dataset('img_tomo', data = img_tomo)
        hf.create_dataset('angle', data = img_angle)

################################################################

def new_user():
    now = datetime.now()
    year = np.str(now.year)
    mon  = '{:02d}'.format(now.month)

    if now.month >= 1 and now.month <=4:    qut = 'Q1'
    elif now.month >= 5 and now.month <=8:  qut = 'Q2'
    else: qut = 'Q3'

    PI_name = input('PI\'s name:')  
    PI_name = PI_name.replace(' ', '_').upper()

    proposal_id = input('Proposal ID:')

    pre = '/NSLS2/xf18id1/users/' + year + qut + '/'
    try:        os.mkdir(pre)
    except Exception:        pass
    
    fn = pre + PI_name + '_Proposal_' + proposal_id
    try:        os.mkdir(fn)
    except Exception:
        print('Found (user, proposal) existed\nEntering folder: {}'.format(os.getcwd()))
        os.chdir(fn)       
        pass
    os.chdir(fn)
    print ('\nUser created successful!\nEntering folder: {}'.format(os.getcwd()))

    
################################################

def plot1d(scan_id = -1):
    h = db[scan_id]
    scan_id = h.start['scan_id']
    num =  int(h.start['plan_args']['num'])
    st = h.start['plan_args']['start']
    en = h.start['plan_args']['stop']
    x = np.linspace(st, en, num)
    dets = h.start['detectors']
    plt.figure()    
    for det in dets:
        data = np.array(list(h.data(det)))
        plt.plot(x, data, '.-', label=det);
        plt.legend()
    plt.title('scan ID: ' + str(scan_id))
    plt.show()


def plot_ic(ics='ic3', scan_id=[-1]):
    det = ics
    plt.figure()
    for sid in scan_id:
        h = db[int(sid)]
        num =  int(h.start['plan_args']['num'])
        st = h.start['plan_args']['start']
        en = h.start['plan_args']['stop']
        x = np.linspace(st, en, num)
        data = np.array(list(h.data(det)))
        plt.plot(x, data, '-', label=str(sid))
        plt.legend()
    plt.title('reading of ic: ' + ics)
    plt.show()

    



def plot2dsum(scan_id = -1, fn='Det_Image'):
    h = db[scan_id]
    if scan_id == -1:
        scan_id = h.start['scan_id']
    det = h.start['detectors'][0]
    det = det + '_image'
    img = np.squeeze(np.array(list(h.data(det))))
    img[img<20] = 0
    img_sum = np.sum(np.sum(img, axis=1), axis=1)
    num =  int(h.start['plan_args']['num'])
    st = h.start['plan_args']['start']
    en = h.start['plan_args']['stop']
    x = np.linspace(st, en, num)
    plt.figure()
    plt.plot(x, img_sum, 'r-.')
    plt.title('scanID: ' + str(scan_id) + ':  ' + det + '_sum')

#    with h5py.File(fn, 'w') as hf:
 #       hf.create_dataset('data', data=img)

from PIL import Image
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


#######################################
'''
f1 = '/home/xf18id/Documents/FXI_commision/DCM_scan/TM_bender_off/pzt_scan_9keV_ssaV_cen_'
f2 = '_2018-02-22.csv'

x, y = list([]), list([])
vgap = np.arange(-5, 3.5, 0.5)

for gap in vgap:
    print(gap)
    fn = f1 + str(gap) + f2
    df = pd.read_csv(fn, sep='\t')
    x.append(list(df['dcm_th2 #0']))
    y.append(list(df['Vout2 #0']))
'''
'''
f = '/home/xf18id/Documents/FXI_commision/DCM_scan/pzt_scan_energy_summary.csv'
df = pd.read_csv(f)
var = np.arange(5, 13, 0.5)
data = []
data_max = []
th2 = df['dcm_th2 #0']
plt.figure()
for x in var:
    tmp = df[str(x) + ' keV']
    data.append(tmp)
    data_max.append(tmp.values.argmax())
    plt.plot(th2, tmp)
'''

#################### test image acquiring time ##############
'''
cap_img = 'XF:18IDB-BI{Det:Neo}TIFF1:WriteFile'
trg_img = 'XF:18IDB-BI{Det:Neo}cam1:Acquire'
st = time.time()
for i in range(10):    
#    print(i)
    my_set_cmd = 'caput ' + trg_img + ' ' + 'Acquire' 
    my_save_cmd = 'caput ' + cap_img + ' ' + 'Write'
#    st = time.time()
    subprocess.Popen(my_set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#    print(time.time()-st)
#    time.sleep(0.05)
    while True:
        r =subprocess.check_output(['caget', trg_img, '-t']).rstrip()
        r = str(r)[2:-1]
        if r == 'Done':
            break
    cost = time.time() - st
#    st = time.time()
    subprocess.Popen(my_save_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#    print(time.time()-st)
#    time.sleep(0.05)
    while True:
        r =subprocess.check_output(['caget', cap_img, '-t']).rstrip()
        r = str(r)[2:-1]
        if r == 'Done':
            break

cost = time.time() - st
'''
