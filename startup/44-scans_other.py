# try:
#     print(detA1)
# except:
#     detA1 = None


def test_test():
    yield from count([Andor], 2)
    h = db[-1]
    print(h.start["scan_id"])


def test_scan(
    exposure_time=0.1,
    out_x=-100,
    out_y=-100,
    out_z=0,
    out_r=0,
    num_img=10,
    num_bkg=10,
    note="",
    period=0.1,
    simu=False,
    md=None,
):
    """
    Take multiple images (Andor camera)

    Input:
    ------------
    exposure_time: float, exposure time for each image

    out_x: float(int), relative sample out position for zps.sx

    out_y: float(int), relative sampel out position for zps.sy

    out_z: float(int), relative sampel out position for zps.sz

    out_r: float(int), relative sampel out position for zps.pi_r

    num_img: int, number of images to take

    num_bkg: int, number of backgroud image to take
    """

    yield from _set_andor_param(exposure_time, period, 1)

    detectors = [Andor]
    y_ini = zps.sy.position
    y_out = y_ini + out_y if not (out_y is None) else y_ini
    x_ini = zps.sx.position
    x_out = x_ini + out_x if not (out_x is None) else x_ini
    z_ini = zps.sz.position
    z_out = z_ini + out_z if not (out_z is None) else z_ini
    r_ini = zps.pi_r.position
    r_out = r_ini + out_r if not (out_r is None) else r_ini
    _md = {
        "detectors": ["Andor"],
        "XEng": XEng.position,
        "plan_args": {
            "exposure_time": exposure_time,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "num_img": num_img,
            "num_bkg": num_bkg,
            "note": note if note else "None",
        },
        "plan_name": "test_scan",
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        # "motor_pos": wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    _md["hints"].setdefault("dimensions", [(("time",), "primary")])

    @stage_decorator(list(detectors))
    @run_decorator(md=_md)
    def inner_scan():
        yield from _open_shutter(simu=simu)
        """
        yield from abs_set(shutter_open, 1, wait=True)
        yield from bps.sleep(2)
        yield from abs_set(shutter_open, 1, wait=True)
        yield from bps.sleep(2)
        """
        for i in range(num_img):
            yield from trigger_and_read(list(detectors))
        # taking out sample and take background image
        yield from mv(zps.pi_r, r_out)
        yield from mv(zps.sz, z_out)
        yield from mv(zps.sx, x_out, zps.sy, y_out)
        for i in range(num_bkg):
            yield from trigger_and_read(list(detectors))
        # close shutter, taking dark image
        yield from _close_shutter(simu=simu)
        """
        yield from abs_set(shutter_close, 1, wait=True)
        yield from bps.sleep(1)
        yield from abs_set(shutter_close, 1, wait=True)
        yield from bps.sleep(1)
        """
        for i in range(num_bkg):
            yield from trigger_and_read(list(detectors))
        yield from mv(zps.sz, z_ini)
        yield from mv(zps.pi_r, r_ini)

        yield from mv(zps.sx, x_ini, zps.sy, y_ini)
        # yield from abs_set(shutter_open, 1, wait=True)

    uid = yield from inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    yield from _close_shutter(simu=simu)
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)
    #    print('loading test_scan and save file to current directory')
    #    load_test_scan(db[-1])
    return uid

def test_scan2(
    exposure_time=0.1,
    period_time=0.1,
    out_x=-100,
    out_y=-100,
    out_z=0,
    out_r=0,
    num_img=10,
    take_dark_img=True,
    relative_move_flag=1,
    rot_first_flag=1, 
    note="",
    simu=False,
    md=None,
):
    """
    Take multiple images (Andor camera)

    Input:
    ------------
    exposure_time: float, exposure time for each image

    out_x: float(int), relative sample out position for zps.sx

    out_y: float(int), relative sampel out position for zps.sy

    out_z: float(int), relative sampel out position for zps.sz

    out_r: float(int), relative sampel out position for zps.pi_r

    num_img: int, number of images to take
    """

    yield from _set_andor_param(exposure_time, period_time, 1)

    detectors = [Andor]
    motor_x_ini = zps.sx.position
    motor_y_ini = zps.sy.position
    motor_z_ini = zps.sz.position
    motor_r_ini = zps.pi_r.position

    if relative_move_flag:
        motor_x_out = motor_x_ini + out_x if not (out_x is None) else motor_x_ini
        motor_y_out = motor_y_ini + out_y if not (out_y is None) else motor_y_ini
        motor_z_out = motor_z_ini + out_z if not (out_z is None) else motor_z_ini
        motor_r_out = motor_r_ini + out_r if not (out_r is None) else motor_r_ini
    else:
        motor_x_out = out_x if not (out_x is None) else motor_x_ini
        motor_y_out = out_y if not (out_y is None) else motor_y_ini
        motor_z_out = out_z if not (out_z is None) else motor_z_ini
        motor_r_out = out_r if not (out_r is None) else motor_r_ini


    motors = [zps.sx, zps.sy, zps.sz, zps.pi_r]

    _md = {
        "detectors": ["Andor"],
        "motors": [mot.name for mot in motors],
        "XEng": XEng.position,
        "plan_args": {
            "exposure_time": exposure_time,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "out_r": out_r,
            "num_img": num_img,
            "num_bkg": 20,
            "note": note if note else "None",
        },
        "plan_name": "test_scan2",
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "note": note if note else "None",
        # "motor_pos": wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    _md["hints"].setdefault("dimensions", [(("time",), "primary")])

    @stage_decorator(list(detectors) + motors)
    @run_decorator(md=_md)
    def inner_scan():

        # close shutter, dark images: numer=chunk_size (e.g.20)
        if take_dark_img:
            print("\nshutter closed, taking dark images...")
            yield from _take_dark_image(detectors, motors, num=1, chunk_size=20, stream_name="dark", simu=simu)

        yield from _open_shutter(simu=simu)
        yield from _set_Andor_chunk_size(detectors, chunk_size=num_img)
        yield from _take_image(detectors, motors, num=1, stream_name="primary")


        # taking out sample and take background image
        print("\nTaking background images...")
        yield from _take_bkg_image(
            motor_x_out,
            motor_y_out,
            motor_z_out,
            motor_r_out,
            detectors,
            [],
            num=1,
            chunk_size=20,
            rot_first_flag=rot_first_flag,
            stream_name="flat",
            simu=simu,
        )
        if take_dark_img:
            yield from _close_shutter(simu=simu)
        yield from _move_sample_in(
            motor_x_ini,
            motor_y_ini,
            motor_z_ini,
            motor_r_ini,
            trans_first_flag=rot_first_flag,
            repeat=3,
        )

    uid = yield from inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    #yield from _close_shutter(simu=simu)
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)
    return uid


def z_scan(
    start=-0.03,
    stop=0.03,
    steps=5,
    out_x=-100,
    out_y=-100,
    chunk_size=10,
    exposure_time=0.1,
    note="",
    md=None,
    simu=False,
    cam=Andor
):
    """
    scan the zone-plate to find best focus
    use as:
    z_scan(start=-0.03, stop=0.03, steps=5, out_x=-100, out_y=-100, chunk_size=10, exposure_time=0.1, fn='/home/xf18id/Documents/tmp/z_scan.h5', note='', md=None)

    Input:
    ---------
    start: float, relative starting position of zp_z

    stop: float, relative stop position of zp_z

    steps: int, number of steps between [start, stop]

    out_x: float, relative amount to move sample out for zps.sx

    out_y: float, relative amount to move sample out for zps.sy

    chunk_size: int, number of images per each subscan (for Andor camera)

    exposure_time: float, exposure time for each image

    note: str, experiment notes

    """

    detectors = [cam]
    motor = zp.z
    z_ini = motor.position  # zp.z intial position
    z_start = z_ini + start
    z_stop = z_ini + stop
    #    detectors = [cam]
    y_ini = zps.sy.position  # sample y position (initial)
    y_out = (
        y_ini + out_y if not (out_y is None) else y_ini
    )  # sample y position (out-position)
    x_ini = zps.sx.position
    x_out = x_ini + out_x if not (out_x is None) else x_ini
    yield from mv(cam.cam.acquire, 0)
    yield from mv(cam.cam.image_mode, 0)
    yield from mv(cam.cam.num_images, chunk_size)
    yield from mv(cam.cam.acquire_time, exposure_time)
    period_cor = max(exposure_time + 0.01, 0.05)
    yield from mv(cam.cam.acquire_period, period_cor)

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [motor.name],
        "XEng": XEng.position,
        "plan_args": {
            "start": start,
            "stop": stop,
            "steps": steps,
            "out_x": out_x,
            "out_y": out_y,
            "chunk_size": chunk_size,
            "exposure_time": exposure_time,
            "note": note if note else "None",
        },
        "plan_name": "z_scan",
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        "motor_pos": wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    my_var = np.linspace(z_start, z_stop, steps)
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + [motor])
    @run_decorator(md=_md)
    def inner_scan():
        #        yield from abs_set(shutter_open, 1, wait=True)
        #        yield from bps.sleep(1)
        #        yield from abs_set(shutter_open, 1)
        #        yield from bps.sleep(1)
        yield from _open_shutter(simu=simu)
        for x in my_var:
            yield from mv(motor, x)
            yield from trigger_and_read(list(detectors) + [motor])
        # backgroud images
        yield from mv(zps.sx, x_out)
        yield from mv(zps.sy, y_out)
        yield from mv(zps.sx, x_out, zps.sy, y_out)
        yield from bps.sleep(0.5)
        yield from trigger_and_read(list(detectors) + [motor])
        yield from _close_shutter(simu=simu)
        yield from bps.sleep(0.5)
        yield from trigger_and_read(list(detectors) + [motor])
        # dark images
        #        yield from abs_set(shutter_close, 1, wait=True)
        #        yield from bps.sleep(1)
        #        yield from abs_set(shutter_close, 1)
        #        yield from bps.sleep(1)
        # move back zone_plate and sample y
        yield from mv(zps.sx, x_ini)
        yield from mv(zps.sy, y_ini)
        yield from mv(zp.z, z_ini)
        # yield from abs_set(shutter_open, 1, wait=True)
        yield from mv(cam.cam.image_mode, 1)

    uid = yield from inner_scan()
    yield from mv(cam.cam.image_mode, 1)
    yield from _close_shutter(simu=simu)
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)

    return uid


def z_scan2(
    start=-0.03,
    stop=0.03,
    steps=5,
    out_x=-100,
    out_y=-100,
    out_z=0,
    chunk_size=10,
    exposure_time=0.1,
    note="",
    md=None,
):
    """
    scan the zone-plate to find best focus
    use as:
    z_scan(start=-0.03, stop=0.03, steps=5, out_x=-100, out_y=-100, chunk_size=10, exposure_time=0.1, fn='/home/xf18id/Documents/tmp/z_scan.h5', note='', md=None)

    Input:
    ---------
    start: float, relative starting position of zp_z

    stop: float, relative stop position of zp_z

    steps: int, number of steps between [start, stop]

    out_x: float, relative amount to move sample out for zps.sx

    out_y: float, relative amount to move sample out for zps.sy

    chunk_size: int, number of images per each subscan (for Andor camera)

    exposure_time: float, exposure time for each image

    note: str, experiment notes

    """

    detectors = [Andor]
    motor = [zps.sx, zps.sy, zps.sz, zps.sz, zp.z]
    zp_ini = zp.z.position  # zp.z intial position
    zp_start = zp_ini + start
    zp_stop = zp_ini + stop
    #    detectors = [Andor]
    y_ini = zps.sy.position  # sample y position (initial)
    y_out = (
        y_ini + out_y if not (out_y is None) else y_ini
    )  # sample y position (out-position)
    x_ini = zps.sx.position
    x_out = x_ini + out_x if not (out_x is None) else x_ini
    z_ini = zps.sz.position
    z_out = z_ini if not (out_z is None) else z_ini
    period = max(exposure_time + 0.01, 0.05)

    yield from _set_andor_param(
        exposure_time=exposure_time, period=period, chunk_size=20
    )

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motor],
        "XEng": XEng.position,
        "plan_args": {
            "start": start,
            "stop": stop,
            "steps": steps,
            "out_x": out_x,
            "out_y": out_y,
            "out_z": out_z,
            "chunk_size": chunk_size,
            "exposure_time": exposure_time,
            "note": note if note else "None",
        },
        "plan_name": "z_scan2",
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        #'motor_pos': wh_pos(print_on_screen=0),
    }
    _md.update(md or {})
    my_var = np.linspace(zp_start, zp_stop, steps)
    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def z_inner_scan():

        # take dark image
        yield from _take_dark_image(detectors, motor)
        yield from _open_shutter()
        for z_pos in my_var:
            yield from mv(zps.sx, x_ini, zps.sy, y_ini, zp.z, z_pos)
            yield from bps.sleep(0.1)
            yield from mv(zps.sx, x_ini, zps.sy, y_ini, zp.z, z_pos)
            yield from bps.sleep(0.1)
            yield from _take_image(detectors, motor=motor, num=1)
            yield from _take_bkg_image(
                out_x=x_out,
                out_y=y_out,
                out_z=z_out,
                out_r=0,
                detectors=detectors,
                motor=motor,
                chunk_size=chunk_size,
            )

        # move back zone_plate and sample y
        yield from mv(zps.sx, x_ini, zps.sy, y_ini, zps.sz, z_ini, zp.z, zp_ini)
        # yield from abs_set(shutter_open, 1, wait=True)

    yield from z_inner_scan()
    yield from mv(Andor.cam.image_mode, 1)
    yield from _close_shutter(simu=False)
    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)


def z_scan3(
    start=-0.03,
    stop=0.03,
    steps=5,
    out_x=-100,
    out_y=-100,
    chunk_size=10,
    exposure_time=0.1,
    note="",
    md=None,
):
    """
    scan the sample z to find best focus
    use as:
    z_scan(start=-0.03, stop=0.03, steps=5, out_x=-100, out_y=-100, chunk_size=10, exposure_time=0.1, fn='/home/xf18id/Documents/tmp/z_scan.h5', note='', md=None)

    Input:
    ---------
    start: float, relative starting position of zp_z

    stop: float, relative stop position of zp_z

    steps: int, number of steps between [start, stop]

    out_x: float, relative amount to move sample out for zps.sx

    out_y: float, relative amount to move sample out for zps.sy

    chunk_size: int, number of images per each subscan (for Andor camera)

    exposure_time: float, exposure time for each image

    note: str, experiment notes

    """

    detectors = [Andor]
    motor = [zps.sx, zps.sy, zps.sz, zps.sz, zp.z]

    #    detectors = [Andor]
    y_ini = zps.sy.position  # sample y position (initial)
    y_out = (
        y_ini + out_y if not (out_y is None) else y_ini
    )  # sample y position (out-position)
    x_ini = zps.sx.position
    x_out = x_ini + out_x if not (out_x is None) else x_ini
    z_ini = zps.sz.position
    z_out = z_ini if not (out_z is None) else z_ini

    z_start = z_ini + start
    z_stop = z_ini + stop

    yield from _set_andor_param(
        exposure_time=exposure_time, period=period, chunk_size=20
    )

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [mot.name for mot in motor],
        "XEng": XEng.position,
        "plan_args": {
            "start": start,
            "stop": stop,
            "steps": steps,
            "out_x": out_x,
            "out_y": out_y,
            "chunk_size": chunk_size,
            "exposure_time": exposure_time,
            "note": note if note else "None",
        },
        "plan_name": "z_scan3",
        "plan_pattern": "linspace",
        "plan_pattern_module": "numpy",
        "hints": {},
        "operator": "FXI",
        #'motor_pos': wh_pos(print_on_screen=0),
    }
    _md.update(md or {})

    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def z_inner_scan():
        my_var = np.linspace(z_start, z_stop, steps)
        # take dark image
        yield from _take_dark_image(detectors, motor)
        yield from _open_shutter()
        for z_pos in my_var:
            yield from mv(zps.sx, x_ini, zps.sy, y_ini, zps.sz, z_pos)
            yield from bps.sleep(0.1)
            yield from mv(zps.sx, x_ini, zps.sy, y_ini, zps.sz, z_pos)
            yield from bps.sleep(0.1)
            yield from _take_image(detectors, motor=motor, num=1)
            yield from _take_bkg_image(
                x_out,
                y_out,
                z_out,
                None,
                detectors=detectors,
                motor=motor,
            )

        # move back zone_plate and sample y
        yield from mv(zps.sx, x_ini, zps.sy, y_ini, zps.sz, z_ini)
        # yield from abs_set(shutter_open, 1, wait=True)

    yield from z_inner_scan()

    txt = get_scan_parameter()
    insert_text(txt)
    print(txt)


#####################


@parameter_annotation_decorator(
    {
        "parameters": {
            "detectors": {
                "annotation": "typing.List[DetectorType1]",
                "devices": {"DetectorType1": ["detA1"]},
                "default": ["detA1"],
            }
        }
    }
)
def cond_scan(detectors=[detA1], *, md=None):
    motor = clens.x

    _md = {
        "detectors": [det.name for det in detectors],
        "motors": [clens.x.name],
        "plan_args": {
            "detectors": list(map(repr, detectors)),
            "motor": repr(motor),
        },
        "plan_name": "cond_scan",
        "hints": {},
        "operator": "FXI",
        "motor_pos": wh_pos(),
    }
    _md.update(md or {})

    try:
        dimensions = [(motor.hints["fields"], "primary")]
    except (AttributeError, KeyError):
        pass
    else:
        _md["hints"].setdefault("dimensions", dimensions)

    @stage_decorator(list(detectors) + [clens.x, clens.y1, clens.y2])
    @run_decorator(md=_md)
    def cond_inner_scan():
        cnt = 1
        #        for z1 in range(0, 1500, 100):
        #            yield from mv(clens.z1, z1, clens.z2, -z1)
        #            for p in range(-1000, 1000, 100):
        #                yield from mv(clens.p, p)
        #                print(f'cnt={cnt}, z1={z1}, z2={-z1}, pitch={p}\n')
        #                for x in range(-3200, -200, 300):
        #                    yield from mv(clens.x, x)
        #                    for y in range(2500, 4500, 200):
        #                        yield from mv(clens.y1, y, clens.y2, y)
        #                        yield from trigger_and_read(list(detectors)+ [clens.x, clens.y1, clens.y2])
        #                        cnt +=1
        for x in range(-1850, -550, 50):
            yield from mv(clens.x, x)
            for y in range(2400, 3200, 50):
                yield from mv(clens.y1, y, clens.y2, y)
                yield from trigger_and_read(
                    list(detectors) + [clens.x, clens.y1, clens.y2]
                )

    return (yield from cond_inner_scan())


#from bluesky.callbacks.mpl_plotting import QtAwareCallback
#from bluesky.preprocessors import subs_wrapper


class LoadCellScanPlot(QtAwareCallback):
    """
    Class for plotting data from 'load_cell_scan' plan.
    """

    def __init__(self, **kwargs):
        super().__init__(use_teleporter=kwargs.pop("use_teleporter", None))

        self._start_new_figure = False
        self._show_axes_titles = False

        self._scan_id_start = 0
        self._scan_id_end = 0

        self._fig = None
        self._ax1 = None
        self._ax2 = None

        # Parameters used in 'ax2' title
        self._load_cell_force = None
        self._bender_pos = None

    def start_new_figure(self):
        """
        Create new figure when the next 'stop' document is received. The next plot
        will be placed in the new figure. Call before the first scan in the series.
        """
        self._start_new_figure = True

    def show_axes_titles(self, *, load_cell_force, bender_pos):
        """
        Add subtitles to the current plot once the next stop document is received.
        Call before the last scan in the series.
        """
        self._show_axes_titles = True
        self._load_cell_force = load_cell_force
        self._bender_pos = bender_pos

    def stop(self, doc):
        # Scan UID
        uid = doc["run_start"]

        h = db[uid]
        # Scan ID
        scan_id = h.start["scan_id"]

        plan_args = h.start["plan_args"]
        # Get values for some parameters used for plotting
        eng_start = plan_args["start"]
        eng_end = plan_args["stop"]
        if h.start["plan_name"] == "delay_scan":
            steps = plan_args["steps"]
        else:
            steps = plan_args["num"]

        if self._start_new_figure:
            self._fig = plt.figure()
            self._ax1 = self._fig.add_subplot(211)
            self._ax2 = self._fig.add_subplot(212)

            # Save ID of the first scan
            self._scan_id_start = scan_id
            self._scan_id_end = scan_id

            self._start_new_figure = False

        y0 = np.abs(np.array(list(h.data(ic3.name))))
        y1 = np.abs(np.array(list(h.data(ic4.name))))
        x = np.linspace(eng_start, eng_end, steps)
        if h.start["plan_name"] == "delay_scan":
            y0 = np.abs(np.array(list(h.data(Vout1.name))))
            y1 = np.abs(np.array(list(h.data(ic3.name))))
            self._ax1.plot(x, y0, ".-")
            self._ax2.plot(x, y1, ".-")
        else:
            r = np.log(y0 / y1)
            self._ax1.plot(x, r, ".-", label=str(np.round(pbsl.y_ctr.position, 2)))
            r_dif = np.array([0] + list(np.diff(r)))
            self._ax2.plot(x, r_dif, ".-", label=str(np.round(pbsl.y_ctr.position, 2)))
            self._ax1.legend()
            self._ax2.legend()

        if self._show_axes_titles:
            self._scan_id_end = scan_id
            if h.start["plan_name"] == "delay_scan":
                self._ax1.title.set_text(
                    "scan_id: {}-{}, {}".format(
                        self._scan_id_start,
                        self._scan_id_end,
                        Vout1.name
                    )
                )
                self._ax2.title.set_text(
                    "scan_id: {}-{}, {}".format(
                        self._scan_id_start,
                        self._scan_id_end,
                        ic3.name
                    )
                )
                self._fig.subplots_adjust(hspace=0.5)
            else:
                self._ax1.title.set_text(
                    "scan_id: {}-{}, ratio of: {}/{}".format(
                        self._scan_id_start,
                        self._scan_id_end,
                        ic3.name,
                        ic4.name,
                    )
                )
                self._ax2.title.set_text(
                    "load_cell: {}, bender_pos: {}".format(
                        self._load_cell_force
                        if self._load_cell_force is not None
                        else "NOT SET",
                        self._bender_pos if self._bender_pos is not None else "NOT SET",
                    )
                )
                self._fig.subplots_adjust(hspace=0.5)

            self._load_cell_force = None
            self._bender_pos = None

            self._show_axes_titles = False

        super().stop(doc)


lcs_plot = LoadCellScanPlot()


def load_cell_scan(
    pzt_cm_bender_pos_list,
    pbsl_y_pos_list,
    num,
    eng_start,
    eng_end,
    steps,
    delay_time=0.5,
):
    """
    At every position in the pzt_cm_bender_pos_list, scan the pbsl.y_ctr under diffenent energies
    Use as:
    load_cell_scan(pzt_cm_bender_pos_list, pbsl_y_pos_list, num, eng_start, eng_end, steps, delay_time=0.5)
    note: energies are in unit if keV

    Inputs:
    --------
    pzt_cm_bender_pos_list: list of "CM_Bender Set Position"
        PV:    XF:18IDA-OP{Mir:CM-Ax:Bender}SET_POSITION

    pbsl_y_pos_list: list of PBSL_y Center Position
        PV:    XF:18IDA-OP{PBSL:1-Ax:YCtr}Mtr

    num: number of repeating scans (engergy scan) at each pzt_cm_bender position and each pbsl_y center position

    eng_start: float, start energy in unit of keV

    eng_end: float, end of energy in unit of keV

    steps:  num of steps from eng_start to eng_end

    delay_time: delay_time between each energy step, in unit of sec
    """

    txt1 = f"load_cell_scan(pzt_cm_bender_pos_list, pbsl_y_pos_list, num={num}, eng_start={eng_start}, eng_end={eng_end}, steps={steps}, delay_time={delay_time})"
    txt2 = f"pzt_cm_bender_pos_list = {pzt_cm_bender_pos_list}"
    txt3 = f"pbsl_y_pos_list = {pbsl_y_pos_list}"
    txt = "##" + txt1 + "\n" + txt2 + "\n" + txt3 + "\n  Consisting of:\n"
    insert_text(txt)

    pbsl_y_ctr_ini = pbsl.y_ctr.position

    check_eng_range([eng_start, eng_end])
    num_pbsl_pos = len(pbsl_y_pos_list)
    yield from _open_shutter(simu=False)

    idx = 0
    for bender_pos in pzt_cm_bender_pos_list:
        yield from mv(pzt_cm.setpos, bender_pos)
        yield from bps.sleep(5)
        load_cell_force = yield from bps.rd(pzt_cm_loadcell)

        # Initiate creating of a new figure
        lcs_plot.start_new_figure()
        if idx % 2:
            pos_list = pbsl_y_pos_list[::-1]
        else:
            pos_list = pbsl_y_pos_list.copy()
        idx += 1 
        for pbsl_pos in pos_list:
            yield from mv(pbsl.y_ctr, pbsl_pos)
            for i in range(num):

                # If the scan is the last in the series, display axes titles and align the plot
                if (pbsl_pos == pbsl_y_pos_list[-1]) and (i == num - 1):
                    lcs_plot.show_axes_titles(
                        load_cell_force=load_cell_force, bender_pos=bender_pos
                    )

                eng_scan_with_plot = subs_wrapper(
                    eng_scan_basic(
                        eng_start,
                        stop=eng_end,
                        num=steps,
                        detectors=[ic3, ic4],
                        delay_time=delay_time,
                    ),
                    [lcs_plot],
                )

                yield from eng_scan_with_plot

    yield from _close_shutter(simu=False)
    yield from mv(pbsl.y_ctr, pbsl_y_ctr_ini)
    print(f"moving pbsl.y_ctr back to initial position: {pbsl.y_ctr.position} mm")
    txt_finish = '## "load_cell_scan()" finished'
    insert_text(txt_finish)


# ===============================================================================================
# The following is the original version of the plan code before the update.
# DELETE THIS CODE AFTER IT IS VERIFIED THAT THE NEW VERSION OF 'load_cell_scan' works properly
# ===============================================================================================
def load_cell_scan_original(
    pzt_cm_bender_pos_list,
    pbsl_y_pos_list,
    num,
    eng_start,
    eng_end,
    steps,
    delay_time=0.5,
):
    """
    At every position in the pzt_cm_bender_pos_list, scan the pbsl.y_ctr under diffenent energies
    Use as:
    load_cell_scan(pzt_cm_bender_pos_list, pbsl_y_pos_list, num, eng_start, eng_end, steps, delay_time=0.5)
    note: energies are in unit if keV

    Inputs:
    --------
    pzt_cm_bender_pos_list: list of "CM_Bender Set Position"
        PV:    XF:18IDA-OP{Mir:CM-Ax:Bender}SET_POSITION

    pbsl_y_pos_list: list of PBSL_y Center Position
        PV:    XF:18IDA-OP{PBSL:1-Ax:YCtr}Mtr

    num: number of repeating scans (engergy scan) at each pzt_cm_bender position and each pbsl_y center position

    eng_start: float, start energy in unit of keV

    eng_end: float, end of energy in unit of keV

    steps:  num of steps from eng_start to eng_end

    delay_time: delay_time between each energy step, in unit of sec
    """

    txt1 = f"load_cell_scan(pzt_cm_bender_pos_list, pbsl_y_pos_list, num={num}, eng_start={eng_start}, eng_end={eng_end}, steps={steps}, delay_time={delay_time})"
    txt2 = f"pzt_cm_bender_pos_list = {pzt_cm_bender_pos_list}"
    txt3 = f"pbsl_y_pos_list = {pbsl_y_pos_list}"
    txt = "##" + txt1 + "\n" + txt2 + "\n" + txt3 + "\n  Consisting of:\n"
    insert_text(txt)

    pbsl_y_ctr_ini = pbsl.y_ctr.position

    check_eng_range([eng_start, eng_end])
    num_pbsl_pos = len(pbsl_y_pos_list)
    yield from _open_shutter(simu=False)

    for bender_pos in pzt_cm_bender_pos_list:
        yield from mv(pzt_cm.setpos, bender_pos)
        yield from bps.sleep(5)
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        load_cell_force = yield from bps.rd(pzt_cm_loadcell)

        for pbsl_pos in pbsl_y_pos_list:
            yield from mv(pbsl.y_ctr, pbsl_pos)
            for i in range(num):
                yield from eng_scan(
                    eng_start,
                    stop=eng_end,
                    num=steps,
                    detectors=[ic3, ic4],
                    delay_time=delay_time,
                )
                h = db[-1]
                y0 = np.array(list(h.data(ic3.name)))
                y1 = np.array(list(h.data(ic4.name)))
                r = np.log(y0 / y1)
                x = np.linspace(eng_start, eng_end, steps)
                ax1.plot(x, r, ".-")
                r_dif = np.array([0] + list(np.diff(r)))
                ax2.plot(x, r_dif, ".-")
        ax1.title.set_text(
            "scan_id: {}-{}, ratio of: {}/{}".format(
                h.start["scan_id"] - num * num_pbsl_pos + 1,
                h.start["scan_id"],
                ic3.name,
                ic4.name,
            )
        )
        ax2.title.set_text(
            "load_cell: {}, bender_pos: {}".format(load_cell_force, bender_pos)
        )
        fig.subplots_adjust(hspace=0.5)
        plt.show()
    yield from _close_shutter(simu=False)
    yield from mv(pbsl.y_ctr, pbsl_y_ctr_ini)
    print(f"moving pbsl.y_ctr back to initial position: {pbsl.y_ctr.position} mm")
    txt_finish = '## "load_cell_scan()" finished'
    insert_text(txt_finish)




def beam_profile_scan(
    dir, start, end, steps, delay_time=0.1, mv_back=False
):
    """
    At every position in the pzt_cm_bender_pos_list, scan the pbsl.y_ctr under diffenent energies
    Use as:
    load_cell_scan(pzt_cm_bender_pos_list, pbsl_y_pos_list, num, eng_start, eng_end, steps, delay_time=0.5)
    note: energies are in unit if keV

    Inputs:
    --------
    pzt_cm_bender_pos_list: list of "CM_Bender Set Position"
        PV:    XF:18IDA-OP{Mir:CM-Ax:Bender}SET_POSITION

    pbsl_y_pos_list: list of PBSL_y Center Position
        PV:    XF:18IDA-OP{PBSL:1-Ax:YCtr}Mtr

    num: number of repeating scans (engergy scan) at each pzt_cm_bender position and each pbsl_y center position

    eng_start: float, start energy in unit of keV

    eng_end: float, end of energy in unit of keV

    steps:  num of steps from eng_start to eng_end

    delay_time: delay_time between each energy step, in unit of sec
    """

    txt = f"## beam profile scan, dir={dir}, start={start}, end={end}, steps={steps}, delay_time={delay_time})"
    insert_text(txt)

    if dir == 'y':
        pbsl_ctr_ini = pbsl.y_ctr.position
        mot = pbsl.y_ctr
    elif dir == 'x':
        pbsl_ctr_ini = pbsl.x_ctr.position
        mot = pbsl.x_ctr   

    pbsl_pos_list = np.linspace(start, end, steps, endpoint=True)
    num_pbsl_pos = steps
    yield from _open_shutter(simu=False)

    # Initiate creating of a new figure
    lcs_plot.start_new_figure()

    # If the scan is the last in the series, display axes titles and align the plot
    lcs_plot.show_axes_titles(
        load_cell_force=f"beam profile scan on {dir}", bender_pos=''
    )

    profile_scan_with_plot = subs_wrapper(
        delay_scan(
        [ic3, ic4, Vout1],
        mot,
        start,
        end,
        steps,
        sleep_time=delay_time,
        md=None,
        mv_back=mv_back
    ),
        [lcs_plot],
    )

    yield from profile_scan_with_plot

    yield from _close_shutter(simu=False)
    #yield from mv(mot, pbsl_ctr_ini)
    txt_finish = '## "beam_profile_scan()" finished'
    insert_text(txt_finish)


def tm_pitch_scan(tm_pitch_list, ssa_h_start, ssa_h_end, steps, delay_time=0.5):
    """
    At every position in the tm_pitch_list, scan the ssa_h_ctr_list
    Use as:


    Inputs:
    --------
    tm_pitch_list: tm incident angle list

    ssa_h_ctr_list: ssa h ceter list

    ssa_start: float, start energy in unit of keV

    ssa_end: float, end of energy in unit of keV

    steps:  num of steps from eng_start to eng_end

    delay_time: delay_time between each energy step, in unit of sec
    """
    from matplotlib.pyplot import legend

    txt1 = f"tm_pitch_scan(tm_pitch_list, ssa_start={ssa_h_start}, ssa_end={ssa_h_end}, steps={steps}, delay_time={delay_time})"
    txt2 = f"tm_pitch_list = {tm_pitch_list}"

    txt = "##" + txt1 + "\n" + txt2 + "\n" + " Consisting of:\n"
    insert_text(txt)

    tm_pitch_ini = tm.p.position

    num_tm_pitch_pos = len(tm_pitch_list)

    lines = []
    labels = []
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for tm_pos in tm_pitch_list:
        yield from mv(tm.p, tm_pos)
        yield from bps.sleep(1)
        yield from delay_scan(
            [ic3], ssa.h_ctr, ssa_h_start, ssa_h_end, steps, sleep_time=delay_time
        )
        h = db[-1]
        y = np.array(list(h.data(ic3.name)))
        x = np.linspace(ssa_h_start, ssa_h_end, steps)
        (line,) = ax1.plot(x, y, ".-")
        label = str(tm_pos)
        lines.append(line)
        labels.append(label)
        plt.show()

    legend(lines, labels)
    plt.show()
    ax1.title.set_text(
        "scan_id: {}-{}, ic3".format(
            h.start["scan_id"] - num_tm_pitch_pos + 1, h.start["scan_id"]
        )
    )

    yield from mv(tm.p, tm_pitch_ini)
    print(f"moving tm_pitch back to initial position: {tm.p.position} mrad")
    txt_finish = '## "tm_pitch_scan()" finished'
    insert_text(txt_finish)


###########################
def ssa_scan_tm_bender(bender_pos_list, ssa_motor, ssa_start, ssa_end, ssa_steps, mv_back=False):
    """
    scanning ssa, with different pzt_tm_bender position

    Inputs:
    --------
    bender_pos_list: list of pzt_tm position
        PV:     XF:18IDA-OP{Mir:TM-Ax:Bender}

    ssa_motor: choose from ssa.v_gap, ssa.v_ctr, ssa.h_gap, ssa.h_ctr

    ssa_start: float, start position of ssa_motor

    ssa_end: float, end position of ssa_motor

    ssa_steps: int, number of ssa_motor movement

    """
    txt1 = f"ssa_scan_tm_bender(bender_pos_list=bender_pos_list, ssa_motor={ssa_motor.name}, ssa_start={ssa_start}, ssa_end={ssa_end}, ssa_steps={ssa_steps})"
    txt2 = f"bender_pos_list = {bender_pos_list}"
    txt = "## " + txt1 + "\n" + txt2 + "\n  Consisting of:\n"
    insert_text(txt)

    pzt_motor = pzt_tm.setpos
    x = np.linspace(ssa_start, ssa_end, ssa_steps)
    for bender_pos in bender_pos_list:
        yield from mv(pzt_motor, bender_pos)
        yield from bps.sleep(2)
        load_cell_force = yield from bps.rd(pzt_tm_loadcell)
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        #        yield from scan([ic3, ic4, Vout2], ssa_motor, ssa_start, ssa_end, ssa_steps)
        yield from delay_scan(
            [ic3, ic4, Vout2],
            ssa_motor,
            ssa_start,
            ssa_end,
            ssa_steps,
            sleep_time=0.2,
            md=None,
            mv_back=mv_back
        )
        h = db[-1]
        y0 = np.abs(np.array(list(h.data(ic3.name))))
        y1 = np.abs(np.array(list(h.data(ic4.name))))
        y2 = np.array(list(h.data(Vout2.name)))
        ax1.plot(x, y0, ".-")
        #            r_dif = np.array([0] + list(np.diff(r)))
        ax2.plot(x, y1, ".-")
        ax3.plot(x, y2, ".-")
        ax1.title.set_text("scan_id: {}, ic3".format(h.start["scan_id"]))
        ax2.title.set_text(
            "ic4, load_cell: {}, bender_pos: {}".format(load_cell_force, bender_pos)
        )
        ax3.title.set_text("Vout2")
        fig.subplots_adjust(hspace=0.5)
        #plt.show()
    txt_finish = '## "ssa_scan_tm_bender()" finished'
    insert_text(txt_finish)


def ssa_scan_tm_yaw(tm_yaw_pos_list, ssa_motor, ssa_start, ssa_end, ssa_steps):
    """
    scanning ssa, with different tm.yaw position

    Inputs:
    --------
    tm_yaw_pos_list: list of tm_yaw position
        PV:     XF:18IDA-OP{Mir:TM-Ax:Yaw}Mtr

    ssa_motor: choose from ssa.v_gap, ssa.v_ctr, ssa.h_gap, ssa.h_ctr

    ssa_start: float, start position of ssa_motor

    ssa_end: float, end position of ssa_motor

    ssa_steps: int, number of ssa_motor movement
    """
    txt1 = f"ssa_scan_tm_yaw(tm_yaw_pos_list=tm_yaw_pos_list, ssa_motor={ssa_motor.name}, ssa_start={ssa-start}, ssa_end={ssa_end}, ssa_steps={ssa_steps})"
    txt2 = f"tm_yaw_pos_list = {tm_yaw_pos_list}"
    txt = "## " + txt1 + "\n" + txt2 + "\n  Consisting of:\n"
    insert_text(txt)
    motor = tm.yaw
    tm_yaw_ini = tm.yaw.position
    x = np.linspace(ssa_start, ssa_end, ssa_steps)
    for tm_yaw_pos in tm_yaw_pos_list:
        yield from mv(motor, tm_yaw_pos)
        yield from bps.sleep(2)
        load_cell_force = yield from bps.rd(pzt_tm_loadcell)
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        #        yield from scan([ic3, ic4, Vout2], ssa_motor, ssa_start, ssa_end, ssa_steps)
        yield from delay_scan(
            [ic3, ic4, Vout2],
            ssa_motor,
            ssa_start,
            ssa_end,
            ssa_steps,
            sleep_time=1.2,
            md=None,
        )
        h = db[-1]
        y0 = np.array(list(h.data(ic3.name)))
        y1 = np.array(list(h.data(ic4.name)))
        y2 = np.array(list(h.data(Vout2.name)))
        ax1.plot(x, y0, ".-")
        #            r_dif = np.array([0] + list(np.diff(r)))
        ax2.plot(x, y1, ".-")
        ax3.plot(x, y2, ".-")
        ax1.title.set_text("scan_id: {}, ic3".format(h.start["scan_id"]))
        ax2.title.set_text("ic4, load_cell: {}".format(load_cell_force))
        ax3.title.set_text("Vout2, tm_yaw = {}".format(tm_yaw_pos))
        fig.subplots_adjust(hspace=0.5)
        plt.show()
    yield from mv(tm.yaw, tm_yaw_ini)
    print(f"moving tm.yaw to initial position: {tm.yaw.position}")
    txt_finish = '## "ssa_scan_tm_yaw()" finished'
    insert_text(txt_finish)


def ssa_scan_pbsl_x_gap(pbsl_x_gap_list, ssa_motor, ssa_start, ssa_end, ssa_steps):
    """
    scanning ssa, with different pbsl.x_gap position
    """

    txt1 = f"ssa_scan_pbsl_x_gap(pbsl_x_gap_list=pbsl_x_gap_list, ssa_motor={ssa_motor.name}, ssa_start={ssa-start}, ssa_end={ssa_end}, ssa_steps={ssa_steps})"
    txt2 = f"pbsl_x_gap_list = {pbsl_x_gap_list}"
    txt = "## " + txt1 + "\n" + txt2 + "\n  Consisting of:\n"
    insert_text(txt)

    motor = pbsl.x_gap
    pbsl_x_gap_ini = pbsl.x_gap.position
    x = np.linspace(ssa_start, ssa_end, ssa_steps)
    for pbsl_x_gap in pbsl_x_gap_list:
        yield from mv(motor, pbsl_x_gap)
        yield from bps.sleep(2)
        load_cell_force = yield from bps.rd(pzt_tm_loadcell)
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        #        yield from scan([ic3, ic4, Vout2], ssa_motor, ssa_start, ssa_end, ssa_steps)
        yield from delay_scan(
            [ic3, ic4, Vout2],
            ssa_motor,
            ssa_start,
            ssa_end,
            ssa_steps,
            sleep_time=1.2,
            md=None,
        )
        h = db[-1]
        y0 = np.array(list(h.data(ic3.name)))
        y1 = np.array(list(h.data(ic4.name)))
        y2 = np.array(list(h.data(Vout2.name)))
        ax1.plot(x, y0, ".-")
        #            r_dif = np.array([0] + list(np.diff(r)))
        ax2.plot(x, y1, ".-")
        ax3.plot(x, y2, ".-")
        ax1.title.set_text("scan_id: {}, ic3".format(h.start["scan_id"]))
        ax2.title.set_text("ic4, load_cell: {}".format(load_cell_force))
        ax3.title.set_text("Vout2, pbsl_x_gap = {}".format(pbsl_x_gap))
        fig.subplots_adjust(hspace=0.5)
        plt.show()
    yield from mv(pbsl.x_gap, pbsl_x_gap_ini)
    print(f"moving pbsl.x_gap to initial position: {pbsl.x_gap.position}")
    txt_finish = '## "ssa_scan_pbsl_x_gap()" finished'
    insert_text(txt_finish)


def ssa_scan_pbsl_y_gap(pbsl_y_gap_list, ssa_motor, ssa_start, ssa_end, ssa_steps):
    """
    scanning ssa, with different pbsl.y_gap position
    """
    txt1 = f"ssa_scan_pbsl_y_gap(pbsl_y_gap_list=pbsl_y_gap_list, ssa_motor={ssa_motor.name}, ssa_start={ssa-start}, ssa_end={ssa_end}, ssa_steps={ssa_steps})"
    txt2 = f"pbsl_y_gap_list = {pbsl_y_gap_list}"
    txt = "## " + txt1 + "\n" + txt2 + "\n  Consisting of:\n"
    insert_text(txt)

    motor = pbsl.y_gap
    pbsl_y_gap_ini = pbsl.y_gap.position
    x = np.linspace(ssa_start, ssa_end, ssa_steps)
    for pbsl_y_gap in pbsl_y_gap_list:
        yield from mv(motor, pbsl_y_gap)
        yield from bps.sleep(2)
        load_cell_force = yield from bps.rd(pzt_tm_loadcell)
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        #        yield from scan([ic3, ic4, Vout2], ssa_motor, ssa_start, ssa_end, ssa_steps)
        yield from delay_scan(
            [ic3, ic4, Vout2],
            ssa_motor,
            ssa_start,
            ssa_end,
            ssa_steps,
            sleep_time=1.2,
            md=None,
        )
        h = db[-1]
        y0 = np.array(list(h.data(ic3.name)))
        y1 = np.array(list(h.data(ic4.name)))
        y2 = np.array(list(h.data(Vout2.name)))
        ax1.plot(x, y0, ".-")
        #            r_dif = np.array([0] + list(np.diff(r)))
        ax2.plot(x, y1, ".-")
        ax3.plot(x, y2, ".-")
        ax1.title.set_text("scan_id: {}, ic3".format(h.start["scan_id"]))
        ax2.title.set_text("ic4, load_cell: {}".format(load_cell_force))
        ax3.title.set_text("Vout2, pbsl_y_gap = {}".format(pbsl_y_gap))
        fig.subplots_adjust(hspace=0.5)
        plt.show()
    yield from mv(pbsl.y_gap, pbsl_y_gap_ini)
    print(f"moving pbsl.y_gap to initial position: {pbsl.y_gap.position}")
    txt_finish = '## "ssa_scan_pbsl_y_gap()" finished'
    insert_text(txt_finish)


def repeat_scan(detectors, motor, start, stop, steps, num=1, sleep_time=1.2):
    det = [det.name for det in detectors]
    det_name = ""
    for i in range(len(det)):
        det_name += det[i]
        det_name += ", "
    det_name = "[" + det_name[:-2] + " ]"
    txt1 = "repeat_scan(detectors=detectors, motor={motor.name}, start={start}, stop={stop}, steps={steps}, num={num}, sleep_time={sleep_time})"
    txt2 = "detectors={det_name}"
    txt = txt1 + "\n" + txt2 + "\n  Consisting of:\n"
    print(txt)
    for i in range(num):
        yield from delay_scan(detectors, motor, start, stop, steps, sleep_time=1.2)


###############


def overnight_count(detectors, num=1, delay=None, *, md=None):
    """
    same function as the default "count",
    re_write it in order to add auto-logging
    """
    if num is None:
        num_intervals = None
    else:
        num_intervals = num - 1
    _md = {
        "detectors": [det.name for det in detectors],
        "num_points": num,
        "XEng": XEng.position,
        "num_intervals": num_intervals,
        "plan_args": {"detectors": "detectors", "num": num, "delay": delay},
        "plan_name": "overnight_count",
        "hints": {},
    }
    _md.update(md or {})
    _md["hints"].setdefault("dimensions", [(("time",), "primary")])

    @bpp.stage_decorator(detectors)
    @bpp.run_decorator(md=_md)
    def inner_count():
        for i in range(num):
            yield from abs_set(shutter_open, 1)
            yield from bps.sleep(1)
            yield from abs_set(shutter_open, 1)
            yield from bps.sleep(1)
            yield from trigger_and_read(list(detectors))
            yield from abs_set(shutter_close, 1)
            yield from bps.sleep(1)
            yield from abs_set(shutter_close, 1)
            yield from bps.sleep(1)
            print("sleep for 60 sec")
            yield from bps.sleep(60)
        yield from mvr(zps.sy, -3000)
        yield from abs_set(shutter_open, 1)
        yield from bps.sleep(1)
        yield from abs_set(shutter_open, 1)
        yield from bps.sleep(1)
        print("take sample out, and take background image")
        for i in range(10):
            yield from trigger_and_read(list(detectors))
        print("close shutter, taking dark image")
        yield from abs_set(shutter_close, 1)
        yield from bps.sleep(1)
        yield from abs_set(shutter_close, 1)
        yield from bps.sleep(1)
        for i in range(10):
            yield from trigger_and_read(list(detectors))
        yield from mvr(zps.sy, 3000)

    #        return (yield from bps.repeat(partial(bps.trigger_and_read, detectors),
    #                                      num=num, delay=delay))
    uid = yield from inner_count()
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


@parameter_annotation_decorator(
    {
        "parameters": {
            "det": {
                "annotation": "typing.List[DetectorType1]",
                "devices": {"DetectorType1": ["detA1"]},
                "default": ["detA1"],
            },
            "mot1": {
                "annotation": "MotorType1",
                "devices": {"MotorType1": ["zps_sz"]},
                "default": "zps_sz",
            },
            "mot2": {
                "annotation": "MotorType2",
                "devices": {"MotorType2": ["zps_sz"]},
                "default": "zps_sy",
            },
        }
    }
)
def knife_edge_scan_for_condensor(
    det=[detA1],
    mot1=zps.sz,
    mot1_start=-1000,
    mot1_end=1000,
    mot1_points=11,
    mot2=zps.sy,
    mot2_start=-50,
    mot2_end=50,
    mot2_points=11,
    mot2_snake=False,
):

    import h5py
    import matplotlib.pylab as plt
    import numpy as np
    from numpy import polyfit, poly1d
    from scipy.optimize import curve_fit, least_squares
    from scipy.special import erf
    from scipy.signal import gaussian

    yield from rel_grid_scan(
        det,
        mot1,
        mot1_start,
        mot1_end,
        mot1_points,
        mot2,
        mot2_start,
        mot2_end,
        mot2_points,
        mot2_snake,
    )

    hdr = db[-1]
    res_uids = list(db.get_resource_uids(hdr))
    for i, uid in enumerate(res_uids):
        res_doc = db.reg.resource_given_uid(uid)
    fpath_root = res_doc["root"]
    fpath_relative = res_doc["resource_path"]
    fn = fpath_root + "/" + fpath_relative

    def erfunc(x, mFL, a, b):
        return mFL * erf((x - a) / (b * np.sqrt(2)))

    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

    #    def fit_gauss()

    f = h5py.File(fn, "r")
    img = f["/entry/data/data"][:]
    f.close()

    zpt = mot1_points
    ypt = mot2_points
    cur = np.sum(img, axis=(1, 2))

    fig0, ax0 = plt.subplots()
    line0 = []
    for ii in range(zpt):
        (l,) = ax0.plot(
            cur[ii * ypt : (ii + 1) * ypt],
            #            color=(0.1 * ii, np.mod(0.2 * ii, 1), np.mod(0.3 * ii, 1)),
        )
        line0.append(l)
        line0[ii].set_label("{0}".format(ii))
        ax0.legend()

    fig1, ax1 = plt.subplots()
    line1 = []
    for ii in range(zpt):
        (l,) = ax1.plot(
            np.gradient(np.log10(cur[ii * ypt : (ii + 1) * ypt])),
            #            color=(0.1 * ii, np.mod(0.2 * ii, 1), np.mod(0.3 * ii, 1)),
        )
        line1.append(l)
        line1[ii].set_label("{0}".format(ii))
        ax1.legend()

    fig2, ax2 = plt.subplots()
    line2 = []
    wz = []
    for ii in range(zpt):
        p0 = [-0.15, 6, 1.2]
        y = np.linspace(0, ypt - 1, num=ypt)
        yf = np.linspace(0, ypt - 1, num=(ypt - 1) * 10 + 1)
        params, extras = curve_fit(
            gauss, y, np.gradient(np.log10(cur[ii * ypt : (ii + 1) * ypt])), p0
        )
        #        params, extras = least_squares(
        #            gauss, y, jac='3-points', np.gradient(np.log10(cur[ii * ypt : (ii + 1) * ypt])), p0
        #        )
        wz.append(params[2])
        (l,) = ax2.plot(
            yf,
            gauss(yf, *params),
            #            color=(0.1 * ii, np.mod(0.2 * ii, 1), np.mod(0.3 * ii, 1)),
        )

        line2.append(l)
        line2[ii].set_label("{0}".format(ii))
        ax2.legend()

    plt.figure(100)
    plt.plot(wz)

    zpt_list = np.linspace(mot1_start, mot1_end, num=zpt)
    for ii in range(zpt):
        print(ii, zpt_list[ii])

    plt.show()
