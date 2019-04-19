#def export_scan(scan_id, binning=4):
#    '''
#    e.g. load_scan([0001, 0002]) 
#    '''
#    for item in scan_id:        
#        export_single_scan(int(item), binning)  
#        db.reg.clear_process_cache()
        
def export_scan(scan_id, scan_id_end=None, binning=4):
    '''
    e.g. load_scan([0001, 0002]) 
    '''
    if scan_id_end is None:
        for item in scan_id:        
            export_single_scan(int(item), binning)  
            db.reg.clear_process_cache()
    else:
        for i in range(scan_id, scan_id_end+1):
            try:
                export_single_scan(int(i), binning)
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                continue
            db.reg.clear_process_cache()

def export_single_scan(scan_id=-1, binning=4):
    h = db[scan_id]
    scan_id = h.start['scan_id']
    scan_type = h.start['plan_name']
#    x_eng = h.start['XEng']
     
    if scan_type == 'tomo_scan':
        print('exporting tomo scan: #{}'.format(scan_id))
        export_tomo_scan(h)
        print('tomo scan: #{} loading finished'.format(scan_id))
    elif scan_type == 'fly_scan':
        print('exporting fly scan: #{}'.format(scan_id))
        export_fly_scan(h)
        print('fly scan: #{} loading finished'.format(scan_id))
    elif scan_type == 'xanes_scan' or scan_type == 'xanes_scan2':
        print('exporting xanes scan: #{}'.format(scan_id))
        export_xanes_scan(h)
        print('xanes scan: #{} loading finished'.format(scan_id))
    elif scan_type == 'z_scan':
        print('exporting z_scan: #{}'.format(scan_id))
        export_z_scan(h)
    elif scan_type == 'test_scan':
        print('exporting test_scan: #{}'.format(scan_id))
        export_test_scan(h)    
    elif scan_type == 'multipos_count':
        print(f'exporting multipos_count: #{scan_id}')
        export_multipos_count(h)
    elif scan_type == 'grid2D_rel':
        print('exporting grid2D_rel: #{}'.format(scan_id))
        export_grid2D_rel(h)
    elif scan_type == 'raster_2D':
        print('exporting raster_2D: #{}'.format(scan_id))
        export_raster_2D(h, binning)
    elif scan_type == 'count' or scan_type == 'delay_count':
        print('exporting count: #{}'.format(scan_id))
        export_count_img(h)
    elif scan_type == 'multipos_2D_xanes_scan2':
        print('exporting multipos_2D_xanes_scan2: #{}'.format(scan_id))
        export_multipos_2D_xanes_scan2(h)
    elif scan_type == 'multipos_2D_xanes_scan3':
        print('exporting multipos_2D_xanes_scan3: #{}'.format(scan_id))
        export_multipos_2D_xanes_scan3(h)
    else:
        print('Un-recognized scan type ......')
        

def export_tomo_scan(h):
    scan_type = 'tomo_scan'
    scan_id = h.start['scan_id']
    try:
        x_eng = h.start['XEng']
    except:
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


def export_fly_scan(h): 
    uid = h.start['uid']
    note = h.start['note']
    scan_type = 'fly_scan'
    scan_id = h.start['scan_id']   
    scan_time = h.start['time'] 
    x_pos =  h.table('baseline')['zps_sx'][1]
    y_pos =  h.table('baseline')['zps_sy'][1]
    z_pos =  h.table('baseline')['zps_sz'][1]
    r_pos =  h.table('baseline')['zps_pi_r'][1]   
    
    
    try:
        x_eng = h.start['XEng']
    except:
        x_eng = h.start['x_ray_energy']
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
        hf.create_dataset('img_bkg', data = np.array(img_bkg, dtype=np.int16))
        hf.create_dataset('img_dark', data = np.array(img_dark, dtype=np.int16))
        hf.create_dataset('img_bkg_avg', data = np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset('img_dark_avg', data = np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset('img_tomo', data = np.array(img_tomo, dtype=np.int16))
        hf.create_dataset('angle', data = img_angle)
        hf.create_dataset('x_ini', data = x_pos)
        hf.create_dataset('y_ini', data = y_pos)
        hf.create_dataset('z_ini', data = z_pos)
        hf.create_dataset('r_ini', data = r_pos)
    del img_tomo
    del img_dark
    del img_bkg
    del imgs
    

def export_xanes_scan(h):
    scan_type = h.start['plan_name']   
#    scan_type = 'xanes_scan'
    uid = h.start['uid']   
    note = h.start['note']   
    scan_id = h.start['scan_id']   
    scan_time = h.start['time'] 
    try:
        x_eng = h.start['XEng']   
    except:
        x_eng = h.start['x_ray_energy']   
    chunk_size = h.start['chunk_size']   
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
        hf.create_dataset('img_bkg', data = np.array(img_bkg_avg, dtype=np.float32))
        hf.create_dataset('img_dark', data = np.array(img_dark_avg, dtype=np.float32))
        hf.create_dataset('img_xanes', data = np.array(img_xanes_norm, dtype=np.float32))
    del img_xanes, img_dark, img_bkg, img_xanes_avg, img_dark_avg
    del img_bkg_avg, imgs, img_xanes_norm


def export_z_scan(h):
    scan_type = h.start['plan_name']
    scan_id = h.start['scan_id']
    uid = h.start['uid']
    try:
        x_eng = h.start['XEng']
    except:
        x_eng = h.start['x_ray_energy']
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
        hf.create_dataset('X_eng', data = x_eng)
        hf.create_dataset('img_bkg', data = img_bkg)
        hf.create_dataset('img_dark', data = img_dark)
        hf.create_dataset('img', data = img_zscan)
        hf.create_dataset('img_norm', data=img_norm)
    del img, img_zscan, img_bkg, img_dark, img_norm

    
def export_test_scan(h):
    import tifffile
    scan_type = h.start['plan_name']
    scan_id = h.start['scan_id']
    uid = h.start['uid']   
    try:
        x_eng = h.start['XEng']
    except:
        x_eng = h.start['x_ray_energy']
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
        hf.create_dataset('X_eng', data = x_eng)
        hf.create_dataset('note', data = note)
        hf.create_dataset('img_bkg', data = img_bkg)
        hf.create_dataset('img_dark', data = img_dark)
        hf.create_dataset('img', data = np.array(img_test, dtype=np.float32))
        hf.create_dataset('img_norm', data=np.array(img_norm, dtype=np.float32))
#    tifffile.imsave(fname_tif, img_norm)
    del img, img_test, img_bkg, img_dark, img_norm



def export_count_img(h):
    '''
    load images (e.g. RE(count([Andor], 10)) ) and save to .h5 file
    '''    
    uid = h.start['uid']
    img = get_img(h)
    scan_id = h.start['scan_id']
    fn = 'count_id_' + str(scan_id) + '.h5'
    with h5py.File(fn, 'w') as hf:
        hf.create_dataset('img',data=img.astype(np.float32))
        hf.create_dataset('uid',data=uid)
        hf.create_dataset('scan_id',data=scan_id)



def export_multipos_count(h):
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


def export_grid2D_rel(h):
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

    
def export_raster_2D(h, binning=4):
    import tifffile
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
    pix = h.start['plan_args']['pxl']

    img_raw = np.squeeze(np.array(list(h.data('Andor_image'))))
    img_dark_avg = np.mean(img_raw[:num_dark], axis=0, keepdims=True)
    img_bkg_avg = np.mean(img_raw[-num_bkg:], axis=0, keepdims = True)
    img = img_raw[num_dark:-num_bkg]
    s = img.shape
    img = (img - img_dark_avg)/(img_bkg_avg-img_dark_avg)
    x_num = round((x_range[1]-x_range[0])+1)
    y_num = round((y_range[1]-y_range[0])+1)
    x_list = np.linspace(x_range[0], x_range[1], x_num) 
    y_list = np.linspace(y_range[0], y_range[1], y_num) 
    row_size = y_num * s[1]
    col_size = x_num * s[2]
    img_patch = np.zeros([1, row_size, col_size])
    index = 0
    pos_file_for_print = np.zeros([x_num*y_num, 4])
    pos_file = ['cord_x\tcord_y\tx_pos_relative\ty_pos_relative\n']
    index=0
    for i in range(int(x_num)):
        for j in range(int(y_num)):
            img_patch[0, j*s[1]:(j+1)*s[1], i*s[2]:(i+1)*s[2]] = img[index]
            pos_file_for_print[index] = [x_list[i], y_list[j], x_list[i]*pix*img_sizeX/1000, y_list[j]*pix*img_sizeY/1000]
            pos_file.append( f'{x_list[i]:3.0f}\t{y_list[j]:3.0f}\t{x_list[i]*pix*img_sizeX/1000:3.3f}\t\t{y_list[j]*pix*img_sizeY/1000:3.3f}\n') 
            index = index + 1
    s = img_patch.shape
    img_patch_bin = bin_ndarray(img_patch, new_shape=(1, int(s[1]/binning), int(s[2]/binning)))
    fout_h5 = f'raster2D_scan_{scan_id}_binning_{binning}.h5'
    fout_tiff = f'raster2D_scan_{scan_id}_binning_{binning}.tiff' 
    fout_txt = f'raster2D_scan_{scan_id}_cord.txt'     
    print(f'{pos_file_for_print}')
    with open(f'{fout_txt}','w+') as f:
        f.writelines(pos_file)
    tifffile.imsave(fout_tiff, np.array(img_patch_bin, dtype=np.float32))
    num_img = int(x_num) * int(y_num)
    cwd=os.getcwd()
    new_dir = f'{cwd}/raster_scan_{scan_id}'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir) 
    '''
    s = img.shape
    tmp = bin_ndarray(img, new_shape=(s[0], int(s[1]/binning), int(s[2]/binning)))
    for i in range(num_img):  
        fout = f'{new_dir}/img_{i:02d}_binning_{binning}.tiff'
        print(f'saving {fout}')
        tifffile.imsave(fout, np.array(tmp[i], dtype=np.float32))
    '''
    fn_h5_save = f'{new_dir}/img_{i:02d}_binning_{binning}.h5'
    with h5py.File(fn_h5_save, 'w') as hf:
        hf.create_dataset('img_patch', data = np.array(img_patch_bin, np.float32))    
        hf.create_dataset('img', data = np.array(img, np.float32))
        hf.create_dataset('img_dark', data = np.array(img_dark_avg, np.float32))       
        hf.create_dataset('img_bkg', data = np.array(img_bkg_avg, np.float32)) 


'''    
def export_multipos_2D_xanes_scan2(h):
    scan_type = h.start['plan_name']
    uid = h.start['uid']
    note = h.start['note']
    scan_id = h.start['scan_id']  
    scan_time = h.start['time']
#    x_eng = h.start['x_ray_energy']
    x_eng = h.start['XEng'][1] 61748
xf18id@xf18id-ws2:~/.ipython/profile_collection/startup$ 

    chunk_size = h.start['chunk_size']
    chunk_size = h.start['num_bkg_images']
    num_eng = h.start['num_eng']
    num_pos = h.start['num_pos']
#    repeat_num = h.start['plan_args']['repeat_num']
    imgs = np.array(list(h.data('Andor_image')))
    imgs = np.mean(imgs, axis=1)
    img_dark = imgs[0]
    eng_list = list(h.start['eng_list'])
    s = imgs.shape

    img_xanes = np.zeros([num_pos, num_eng, imgs.shape[1], imgs.shape[2]])
    img_bkg = np.zeros([num_eng, imgs.shape[1], imgs.shape[2]])
    #for repeat in range(repeat_num):
    for repeat in range(1):
        index = 1
        for i in range(num_eng):
            for j in range(num_pos):
                img_xanes[j, i] = imgs[index]
                index += 1
            img_bkg[i] = imgs[index]
            index += 1

        for i in range(num_eng):
            for j in range(num_pos):
                img_xanes[j,i] = (img_xanes[j,i] - img_dark) / (img_bkg[i] - img_dark)
        # save data
        fn = os.getcwd() + '/'
        for j in range(num_pos):
            fname = f'{fn}{scan_type}_id_{scan_id}_pos_{j}.h5'
            with h5py.File(fname, 'w') as hf:
                hf.create_dataset('uid', data = uid)
                hf.create_dataset('scan_id', data = scan_id)
                hf.create_dataset('note', data = note)
                hf.create_dataset('scan_time', data = scan_time)
                hf.create_dataset('X_eng', data = eng_list)
                hf.create_dataset('img_bkg', data = np.array(img_bkg, dtype=np.float32))
                hf.create_dataset('img_dark', data = np.array(img_dark, dtype=np.float32))
                hf.create_dataset('img_xanes', data = np.array(img_xanes[j], dtype=np.float32))
    del img_xanes
    del img_bkg
    del img_dark    
    del imgs
'''

def export_multipos_2D_xanes_scan2(h):
    scan_type = h.start['plan_name']
    uid = h.start['uid']
    note = h.start['note']
    scan_id = h.start['scan_id']  
    scan_time = h.start['time']
#    x_eng = h.start['x_ray_energy']
    x_eng = h.start['XEng']
    chunk_size = h.start['chunk_size']
    chunk_size = h.start['num_bkg_images']
    num_eng = h.start['num_eng']
    num_pos = h.start['num_pos']
    try:
        repeat_num = h.start['plan_args']['repeat_num']
    except:
        repeat_num = 1
    imgs = list(h.data('Andor_image'))

#    imgs = np.mean(imgs, axis=1)
    img_dark = np.array(imgs[0])
    img_dark = np.mean(img_dark, axis=0, keepdims=True) # revised here
    eng_list = list(h.start['eng_list'])
#    s = imgs.shape
    s = img_dark.shape # revised here e.g,. shape=(1, 2160, 2560)

#    img_xanes = np.zeros([num_pos, num_eng, imgs.shape[1], imgs.shape[2]])
    img_xanes = np.zeros([num_pos, num_eng, s[1], s[2]])
    img_bkg = np.zeros([num_eng, s[1], s[2]])
    index = 1
    for repeat in range(repeat_num):  # revised here
        try:
            print(f'repeat: {repeat}')
            for i in range(num_eng):
                for j in range(num_pos):
                    img_xanes[j, i] = np.mean(np.array(imgs[index]), axis=0)
                    index += 1
                img_bkg[i] = np.mean(np.array(imgs[index]), axis=0)
                index += 1

            for i in range(num_eng):
                for j in range(num_pos):
                    img_xanes[j,i] = (img_xanes[j,i] - img_dark) / (img_bkg[i] - img_dark)
            # save data
            fn = os.getcwd() + '/'
            for j in range(num_pos):
                fname = f'{fn}{scan_type}_id_{scan_id}_repeat_{repeat:02d}_pos_{j:02d}.h5'
                print(f'saving {fname}')
                with h5py.File(fname, 'w') as hf:
                    hf.create_dataset('uid', data = uid)
                    hf.create_dataset('scan_id', data = scan_id)
                    hf.create_dataset('note', data = note)
                    hf.create_dataset('scan_time', data = scan_time)
                    hf.create_dataset('X_eng', data = eng_list)
                    hf.create_dataset('img_bkg', data = np.array(img_bkg, dtype=np.float32))
                    hf.create_dataset('img_dark', data = np.array(img_dark, dtype=np.float32))
                    hf.create_dataset('img_xanes', data = np.array(img_xanes[j], dtype=np.float32))
        except:
            print(f'fails in export repeat# {repeat}')
    del img_xanes
    del img_bkg
    del img_dark    
    del imgs





def export_multipos_2D_xanes_scan3(h):
    scan_type = h.start['plan_name']
    uid = h.start['uid']
    note = h.start['note']
    scan_id = h.start['scan_id']  
    scan_time = h.start['time']
#    x_eng = h.start['x_ray_energy']
    x_eng = h.start['XEng']
    chunk_size = h.start['chunk_size']
    chunk_size = h.start['num_bkg_images']
    num_eng = h.start['num_eng']
    num_pos = h.start['num_pos']
#    repeat_num = h.start['plan_args']['repeat_num']
    imgs = np.array(list(h.data('Andor_image')))
    imgs = np.mean(imgs, axis=1)
    img_dark = imgs[0]
    eng_list = list(h.start['eng_list'])
    s = imgs.shape

    img_xanes = np.zeros([num_pos, num_eng, imgs.shape[1], imgs.shape[2]])
    img_bkg = np.zeros([num_eng, imgs.shape[1], imgs.shape[2]])

    index = 1
    for i in range(num_eng):
        for j in range(num_pos):
            img_xanes[j, i] = imgs[index]
            index += 1

    img_bkg = imgs[-num_eng:]

    for i in range(num_eng):
        for j in range(num_pos):
            img_xanes[j,i] = (img_xanes[j,i] - img_dark) / (img_bkg[i] - img_dark)
    # save data
    fn = os.getcwd() + '/'
    for j in range(num_pos):
        fname = f'{fn}{scan_type}_id_{scan_id}_pos_{j}.h5'
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('uid', data = uid)
            hf.create_dataset('scan_id', data = scan_id)
            hf.create_dataset('note', data = note)
            hf.create_dataset('scan_time', data = scan_time)
            hf.create_dataset('X_eng', data = eng_list)
            hf.create_dataset('img_bkg', data = np.array(img_bkg, dtype=np.float32))
            hf.create_dataset('img_dark', data = np.array(img_dark, dtype=np.float32))
            hf.create_dataset('img_xanes', data = np.array(img_xanes[j], dtype=np.float32))
    del img_xanes
    del img_bkg
    del img_dark    
    del imgs


    
