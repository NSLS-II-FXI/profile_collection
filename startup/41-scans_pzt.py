def pzt_scan(pzt_motor, start, stop, steps, detectors=[Vout2], sleep_time=1, md=None):
    """
    scan the pzt_motor (e.g., pzt_dcm_th2), detectors can be any signal or motor (e.g., Andor, dcm.th2)

    Inputs:
    ---------
    pzt_motor: choose from pzt_dcm_th2, pzt_dcm_chi2

    start: float, start position

    stop: float, stop position

    steps: int, number of steps

    detectors: list of detectors, e.g., [Vout2, Andor, ic3]

    sleep time: float, in unit of sec
    
    """
    if Andor in detectors:
        exposure_time = (yield from bps.rd(Andor.cam.acquire_time))
        yield from mv(Andor.cam.acquire, 0)
        yield from mv(Andor.cam.image_mode, 0)
        yield from mv(Andor.cam.num_images, 1)
        Andor.cam.acquire_period.put(exposure_time)

    motor = pzt_motor.setpos
    motor_readback = pzt_motor.pos
    motor_ini_pos = motor_readback.get()
    detector_set_read = [motor, motor_readback]
    detector_all = detector_set_read + detectors

    _md = {
        "detectors": [det.name for det in detector_all],
        "detector_set_read": [det.name for det in detector_set_read],
        "motors": [motor.name],
        "XEng": XEng.position,
        "plan_args": {
            "pzt_motor": pzt_motor.name,
            "start": start,
            "stop": stop,
            "steps": steps,
            "detectors": "detectors",
            "sleep_time": sleep_time,
        },
        "plan_name": "pzt_scan",
        "hints": {},
        "motor_pos": wh_pos(print_on_screen=0),
        "operator": "FXI",
    }
    _md.update(md or {})
    try:
        dimensions = [(pzt_motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detector_all))
    @run_decorator(md=_md)
    def pzt_inner_scan():
        my_var = np.linspace(start, stop, steps)
        print(my_var)
        for x in my_var:
            print(x)
            yield from mv(motor, x)
            yield from bps.sleep(sleep_time)
            yield from trigger_and_read(list(detector_all))
        yield from mv(motor, motor_ini_pos)

    uid = yield from pzt_inner_scan()

    h = db[-1]
    scan_id = h.start["scan_id"]
    det = [det.name for det in detectors]
    det_name = ""
    for i in range(len(det)):
        det_name += det[i]
        det_name += ", "
    det_name = "[" + det_name[:-2] + "]"
    txt1 = get_scan_parameter()
    txt2 = f"detectors = {det_name}"
    txt = txt1 + "\n" + txt2
    insert_text(txt)
    print(txt)
    return uid

    # def pzt_scan(moving_pzt, start, stop, steps, read_back_dev, record_dev, delay_time=5, print_flag=1, overlay_flag=0):
    """
    Input:
    -------
    moving_pzt: pv name of the pzt device, e.g. 'XF:18IDA-OP{Mir:DCM-Ax:Th2Fine}SET_POSITION.A'

    read_back_dev: device (encoder) that changes with moving_pzt, e.g., dcm.th2

    record_dev: signal you want to record, e.g. Vout2

    delay_time: waiting time for device to response
    """


#    current_pos = moving_pzt.pos

#    my_set_cmd = 'caput ' + moving_pzt.setting_pv + ' ' + str(start)
#    subprocess.Popen(my_set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#    time.sleep(delay_time)

#    my_var = np.linspace(start, stop, steps)
#    pzt_readout = []
#    motor_readout = []
#    signal_readout = []
#    for x in my_var:
#        my_set_cmd = 'caput ' + moving_pzt.setting_pv + ' ' + str(x)
#        subprocess.Popen(my_set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

#        time_start = time.time()
#        while True:
#            pos = subprocess.check_output(['caget', pzt_dcm_th2.getting_pv, '-t']).rstrip()
#            pos = np.float(str_convert(pos))
#            if np.abs(pos-x) < 1e-2 or (time.time()-time_start > delay_time):
#                break
#        time.sleep(1)

#        pzt_readout.append(pos)
#        y = read_back_dev.position

#        motor_readout.append(y)
#        z = record_dev.value
#        signal_readout.append(z)

#        prt1 = moving_pzt.name + ': {:2.4f}'.format(x)
#        prt2 = 'pzt read_back: {:2.4f}'.format(pos)
#        prt3 = read_back_dev.name + ': {:2.4f}'.format(y)
#        prt4 = record_dev.name + ': {:2.4f}'.format(z)

#        print(prt1 + '   ' + prt2 + '   ' + prt3 + '   ' + prt4)

#    my_set_cmd = 'caput ' + moving_pzt.setting_pv + ' ' + str(current_pos)
#    subprocess.Popen(my_set_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

#    print('\nmoving {0} back to original position: {1:2.4f}'.format(moving_pzt.name, current_pos))

#    if not overlay_flag:
#        plt.figure()
#    if print_flag:

#        plt.subplot(221);plt.plot(my_var, pzt_readout, '.-')
#        plt.xlabel(moving_pzt.name+' set_point');plt.ylabel(read_back_dev.name+ ' read_out')

#        plt.subplot(222); plt.plot(pzt_readout, motor_readout, '.-')
#        plt.xlabel(moving_pzt.name);plt.ylabel(read_back_dev.name)

#        plt.subplot(223); plt.plot(pzt_readout, signal_readout, '.-')
#        plt.xlabel(moving_pzt.name);plt.ylabel(record_dev.name)

#        plt.subplot(224); plt.plot(motor_readout, signal_readout, '.-')
#        plt.xlabel(read_back_dev.name);plt.ylabel(record_dev.name)

#    return pzt_readout, motor_readout, signal_readout


def pzt_scan_multiple(
    moving_pzt,
    start,
    stop,
    steps,
    detectors=[Vout2],
    repeat_num=2,
    sleep_time=1,
    fn="/home/xf18id/Documents/FXI_commision/DCM_scan/",
):
    """
    Repeat scanning the pzt (e.g. pzt_dcm_ch2, pzt_dcm_th2), and read the detector outputs. Images and .csv data file will be saved
    Use as:
    e.g., pzt_scan_multiple(pzt_dcm_th2,-4, -2, 3, [Vout2, Andor,ic1, dcm.th2], repeat_num=2, save_file_dir='/home/xf18id/Documents/FXI_commision/DCM_scan/')

    Inputs:
    ---------

    moving_pzt: pzt device
        e.g., pzt_dcm_th2, pzt_dcm_chi
    start: float
        start position of pzt stage
    stop: float
        stop position of pzt stage
    steps:  int
        steps of scanning
    detectors: list of detector device or signals
        e.g., [dcm.th2, Andor, ic3, Vout2]
    repeat_num: int
        repeat scanning for "repeat_num" times
    save_file_dir: str
        directory to save files and images

    """

    det = [det.name for det in detectors]
    det_name = ""
    for i in range(len(det)):
        det_name += det[i]
        det_name += ", "
    det_name = "[" + det_name[:-2] + "]"
    txt = f"pzt_scan_multiple(moving_pzt={moving_pzt.name}, start={start}, stop={stop}, steps={steps}, detectors={det_name}, repeat_num={repeat_num}, sleep_time={sleep_time}, fn={fn})\n  Consisting of:\n"
    insert_text(txt)

    current_eng = XEng.position
    df = pd.DataFrame(data=[])

    for num in range(repeat_num):
        yield from pzt_scan(
            moving_pzt, start, stop, steps, detectors=detectors, sleep_time=sleep_time
        )
    yield from abs_set(XEng, current_eng, wait=True)
    print("\nscan finished, ploting and saving data...")
    fig = plt.figure()
    for num in reversed(range(repeat_num)):
        h = db[-1 - num]
        scan_id = h.start["scan_id"]
        detector_set_read = h.start["detector_set_read"]
        col_x_prefix = detector_set_read[1]
        col_x = col_x_prefix + " #" + "{}".format(scan_id)

        motor_readout = np.array(list(h.data(col_x_prefix)))
        df[col_x] = pd.Series(motor_readout)

        detector_signal = h.start["detectors"]

        for i in range(len(detector_signal)):
            det = detector_signal[i]

            if (det == "Andor") or (det == "detA1"):
                det = det + "_stats1_total"
            det_readout = np.array(list(h.data(det)))
            col_y_prefix = det
            col_y = col_y_prefix + " #" + "{}".format(scan_id)
            df[col_y] = pd.Series(det_readout)
            plt.subplot(len(detector_signal), 1, i + 1)
            plt.plot(df[col_x], df[col_y])
            plt.ylabel(det)

    plt.subplot(len(detector_signal), 1, len(detector_signal))
    plt.xlabel(col_x_prefix)
    plt.subplot(len(detector_signal), 1, 1)
    plt.title("X-ray Energy: {:2.1f}keV".format(current_eng))

    now = datetime.now()
    year = np.str(now.year)
    mon = "{:02d}".format(now.month)
    day = "{:02d}".format(now.day)
    hour = "{:02d}".format(now.hour)
    minu = "{:02d}".format(now.minute)
    current_date = year + "-" + mon + "-" + day
    fn = (
        save_file_dir
        + "pzt_scan_"
        + "{:2.1f}keV_".format(current_eng)
        + current_date
        + "_"
        + hour
        + "-"
        + minu
    )
    fn_fig = fn + ".tiff"
    fn_file = fn + ".csv"
    df.to_csv(fn_file, sep="\t")
    fig.savefig(fn_fig)
    print("save to: " + fn_file)
    txt_finish = '## "pzt_scan_multiple()" finished'
    insert_text(txt_finish)


######################


def pzt_energy_scan(
    moving_pzt,
    start,
    stop,
    steps,
    eng_list,
    detectors=[dcm.th2, Vout2],
    repeat_num=1,
    sleep_time=1,
    fn="/home/xf18id/Documents/FXI_commision/DCM_scan/",
):
    """
    With given energy list, scan the pzt multiple times and record the signal from various detectors, file will be saved to local folder.

    Inputs:
    ---------
    moving_pzt: pzt device
        e.g., pzt_dcm_th2, pzt_dcm_chi
    start: float
        start position of pzt stage
    stop: float
        stop position of pzt stage
    steps:  int
        steps of scanning
    eng_list: list or array(float)
        e.g. [8.9, 9.0]
    detectors: list of detector device or signals
        e.g., [dcm.th2, Andor, ic3, Vout2]
    repeat_num: int
        repeat scanning for "repeat_num" times
    save_file_dir: str
        directory to save files and images

    """
    det = [det.name for det in detectors]
    det_name = ""
    for i in range(len(det)):
        det_name += det[i]
        det_name += ", "
    det_name = "[" + det_name[:-2] + "]"
    txt = f"pzt_energy_scan(moving_pzt={moving_pzt.name}, start={start}, stop={stop}, steps={steps}, eng_list, detectors={det_name}, repeat_num={repeat_num}, sleep_time={sleep_time}, fn={fn})\neng+list={eng_list}\n  Consisting of:\n"
    insert_text(txt)
    eng_ini = XEng.position
    yield from abs_set(shutter_open, 1)
    yield from bps.sleep(1)
    yield from abs_set(shutter_open, 1)
    print("shutter open")
    for eng in eng_list:
        yield from abs_set(XEng, eng, wait=True)
        current_eng = XEng.position
        yield from bps.sleep(1)
        print("current X-ray Energy: {:2.1f}keV".format(current_eng))
        yield from pzt_scan_multiple(
            pzt_dcm_th2,
            start,
            stop,
            steps,
            detectors,
            repeat_num=repeat_num,
            sleep_time=sleep_time,
            fn=fn,
        )
    yield from abs_set(XEng, eng_ini, wait=True)
    yield from abs_set(shutter_close, 1)
    yield from bps.sleep(1)
    yield from abs_set(shutter_close, 1)
    txt_finish = '## "pzt_energy_scan()" finished'
    insert_text(txt_finish)


def pzt_overnight_scan(
    moving_pzt,
    start,
    stop,
    steps,
    detectors=[dcm.th2, Vout2],
    repeat_num=10,
    sleep_time=1,
    night_sleep_time=3600,
    scan_num=12,
    fn="/home/xf18id/Documents/FXI_commision/DCM_scan/",
):
    """
    At current energy, repeating scan the pzt multiple times and record the signal from various detectors, file will be saved to local folder.

    Inputs:
    ---------
    moving_pzt: pzt device
        e.g., pzt_dcm_th2, pzt_dcm_chi
    start: float
        start position of pzt stage
    stop: float
        stop position of pzt stage
    steps:  int
        steps of scanning
    eng_list: list or array(float)
        e.g. [8.9, 9.0]
    detectors: list of detector device or signals
        e.g., [dcm.th2, Andor, ic3, Vout2]
    repeat_num: int
        single step: repeat scanning for "repeat_num" times
    sleep_time: float
        time interval (seconds) for successive scans, e.g. 3600 seconds
    scan_num: int
        repeat multiple step scanning, e.g. 12 times, at 1 hour interval (set sleep_time=3600)
    save_file_dir: str
        directory to save files and images

    """

    eng_ini = XEng.position
    print("current X-ray Energy: {:2.1f}keV".format(current_def))
    print("run {0:d} times at {1:d} seconds interval".format(repeat_num, scan_num))
    for i in range(scan_num):
        print("scan num: {:d}".format(i))
        yield from pzt_scan_multiple(
            pzt_dcm_th2,
            start,
            stop,
            steps,
            detectors,
            repeat_num=repeat_num,
            sleep_time=sleep_time,
            fn=save_file_dir,
        )
        yield from bps.sleep(night_sleep_time)


#######
