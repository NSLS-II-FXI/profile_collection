import itertools

from ophyd import Component as Cpt, Signal
from ophyd.areadetector.filestore_mixins import (
    FileStoreHDF5IterativeWrite,
    resource_factory,
)
from pathlib import PurePath
from ophyd import EpicsSignal, AreaDetector
from ophyd import (
    ImagePlugin,
    TransformPlugin,
    ROIPlugin,
    HDF5Plugin,
    ProcessPlugin,
)

from ophyd.areadetector.trigger_mixins import SingleTrigger

from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.detectors import DetectorBase
from nslsii.ad33 import StatsPluginV33, CamV33Mixin
#from nslsii.ad33 import SingleTriggerV33
from ophyd.areadetector.trigger_mixins import TriggerBase, ADTriggerStatus
from ophyd.device import Staged
from ophyd.status import SubscriptionStatus

global TimeStampRecord
TimeStampRecord = []

class ExternalFileReference(Signal):
    """
    A pure software signal where a Device can stash a datum_id
    """

    def __init__(self, *args, shape, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape

    def describe(self):
        res = super().describe()
        res[self.name].update(
            dict(external="FILESTORE:", dtype="array", shape=self.shape)
        )
        return res


class SingleTriggerV33(TriggerBase):
    _status_type = ADTriggerStatus

    def __init__(self, *args, image_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        if image_name is None:
            image_name = '_'.join([self.name, 'image'])
        self._image_name = image_name

    def trigger(self):
        "Trigger one acquisition."
        if self._staged != Staged.yes:
            raise RuntimeError("This detector is not ready to trigger."
                               "Call the stage() method before triggering.")

        def acquire_complete(*args, old_value, value, **kwargs):
            return old_value != 0 and value == 0

        status = SubscriptionStatus(self.cam.detector_state, run=False, callback=acquire_complete)
        self._acquisition_signal.set(1)
        self.dispatch(self._image_name, ttime.time())
        return status


class AndorCam(CamV33Mixin, AreaDetectorCam):
    def __init__(self, *args, **kwargs):
        AreaDetectorCam.__init__(self, *args, **kwargs)
        self.stage_sigs["wait_for_plugins"] = "Yes"

    def ensure_nonblocking(self):
        self.stage_sigs["wait_for_plugins"] = "Yes"
        for c in self.parent.component_names:
            cpt = getattr(self.parent, c)
            if cpt is self:
                continue
            if hasattr(cpt, "ensure_nonblocking"):
                cpt.ensure_nonblocking()


class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    # AD v2.2.0 (at least) does not have this. It is present in v1.9.1.
    file_number_sync = None
    time_stamp = Cpt(ExternalFileReference, value="", kind="normal", shape=[])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ts_datum_factory = None
        self._ts_resource_uid = ""
        self._ts_counter = None

    def stage(self):
        self._ts_counter = itertools.count()
        return super().stage()

    def get_frames_per_point(self):
        return self.parent.cam.num_images.get()

    def make_filename(self):
        # stash this so that it is available on resume
        self._ret = super().make_filename()
        return self._ret

    def _generate_resource(self, resource_kwargs):
        # don't re-write the "normal" code path .... yet
        super()._generate_resource(resource_kwargs)
        fn = PurePath(self._fn).relative_to(self.reg_root)

        # Update the shape that describe() will report.
        self.time_stamp.shape = [self.get_frames_per_point()]

        resource, self._ts_datum_factory = resource_factory(
            spec="AD_HDF5_TS",
            root=str(self.reg_root),
            resource_path=str(fn),
            resource_kwargs=resource_kwargs,
            path_semantics=self.path_semantics,
        )

        self._ts_resource_uid = resource["uid"]
        self._asset_docs_cache.append(("resource", resource))

    def generate_datum(self, key, timestamp, datum_kwargs):
        # again, don't re-work the normal code path... yet
        ret = super().generate_datum(key, timestamp, datum_kwargs)
        datum_kwargs = datum_kwargs or {}
        datum_kwargs.update({"point_number": next(self._ts_counter)})
        # make the timestamp datum, in this case we know they match
        datum = self._ts_datum_factory(datum_kwargs)
        datum_id = datum["datum_id"]

        # stash so that we can collect later
        self._asset_docs_cache.append(("datum", datum))
        # put in the soft-signal so it get auto-read later
        self.time_stamp.put(datum_id)
        # yes, just return one to avoid breaking the API
        return ret



def timing(f):
    from functools import wraps
    from time import asctime, localtime, time
    global TimeStampRecord
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        TimeStampRecord.append(te-ts)
        print(f"{asctime(localtime())} {f.__name__} {(te-ts):2.4f} sec")
        return result
    return wrap

#
class AndorKlass(SingleTriggerV33, DetectorBase):
    cam = Cpt(AndorCam, "cam1:")
    image = Cpt(ImagePlugin, "image1:")
    #    stats1 = Cpt(StatsPluginV33, "Stats1:")
    #    stats2 = Cpt(StatsPluginV33, 'Stats2:')
    #    stats3 = Cpt(StatsPluginV33, 'Stats3:')
    #    stats4 = Cpt(StatsPluginV33, 'Stats4:')
    #    stats5 = Cpt(StatsPluginV33, 'Stats5:')
    trans1 = Cpt(TransformPlugin, "Trans1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    roi2 = Cpt(ROIPlugin, "ROI2:")
    roi3 = Cpt(ROIPlugin, "ROI3:")
    roi4 = Cpt(ROIPlugin, "ROI4:")
    proc1 = Cpt(ProcessPlugin, "Proc1:")

    hdf5 = Cpt(
        HDF5PluginWithFileStore,
        suffix="HDF1:",
        write_path_template="/nsls2/data/fxi-new/legacy/Andor/%Y/%m/%d/",
        # write_path_template='/tmp/test_2022/%Y/%m/%d/' ,
        # write_path_template="/nsls2/data/fxi-new/assets/default/%Y/%m/%d/",
        # write_path_template="/nsls2/data/fxi-new/legacy/Andor//%Y/%m/%d/",
        # root="/nsls2/data/fxi-new/assets/default",
        root="/nsls2/data/fxi-new/legacy/Andor",
        # root = "/tmp/test_2022",
        # root="/nsls2/data/fxi-new/legacy/Andor/",
        # write_path_template='/tmp/',
        # root='/dev/shm',
    )

    ac_period = Cpt(EpicsSignal, "cam1:AcquirePeriod")
    binning = Cpt(EpicsSignal, "cam1:A3Binning")

    def stop(self):
        self.hdf5.capture.put(0)
        return super().stop()

    def pause(self):
        self.hdf5.capture.put(0)
        return super().pause()

    def resume(self):
        self.hdf5.capture.put(1)
        # The AD HDF5 plugin bumps its file_number and starts writing into a
        # *new file* because we toggled capturing off and on again.
        # Generate a new Resource document for the new file.

        # grab the stashed result from make_filename
        filename, read_path, write_path = self.hdf5._ret
        self.hdf5._fn = self.hdf5.file_template.get() % (
            read_path,
            filename,
            self.hdf5.file_number.get() - 1,
        )
        # file_number is *next*
        # iteration
        res_kwargs = {"frame_per_point": self.hdf5.get_frames_per_point()}
        self.hdf5._generate_resource(res_kwargs)
        return super().resume()

    @timing
    def stage(self):
        import itertools
        if self.cam.detector_state.get() != 0:
            raise RuntimeError("Andor must be in the Idle state to stage.")
        
        for j in itertools.count():
            try:
                print(f"stage attempt {j}")
                return super().stage()
            except TimeoutError:
                N_try = 20
                if j < N_try:
                    print(f"failed to stage on try{j}/{N_try}, may try again")
                    continue
                else:
                    raise

    @timing
    def unstage(self, *args, **kwargs):
        import itertools
        #self._acquisition_signal.put(0, wait=True)
        for j in itertools.count():
            try:
                print(f"unstage attempt {j}")
                ret = super().unstage()
            except TimeoutError:
                N_try = 20
                if j < N_try:
                    print(f"failed to unstage on attempt {j}/{N_try}, may try again")
                    continue
                else:
                    raise
            else:
                break
        return ret


class Manta(SingleTrigger, AreaDetector):
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPluginV33, "Stats1:")
    #    stats2 = Cpt(StatsPluginV33, 'Stats2:')
    #    stats3 = Cpt(StatsPluginV33, 'Stats3:')
    #    stats4 = Cpt(StatsPluginV33, 'Stats4:')
    #    stats5 = Cpt(StatsPluginV33, 'Stats5:')
    trans1 = Cpt(TransformPlugin, "Trans1:")
    roi1 = Cpt(ROIPlugin, "ROI1:")
    #    roi2 = Cpt(ROIPlugin, 'ROI2:')
    #    roi3 = Cpt(ROIPlugin, 'ROI3:')
    #    roi4 = Cpt(ROIPlugin, 'ROI4:')
    proc1 = Cpt(ProcessPlugin, "Proc1:")

    hdf5 = Cpt(
        HDF5PluginWithFileStore,
        suffix="HDF1:",
        # write_path_template="/nsls2/data/fxi-new/legacy/Andor/%Y/%m/%d/",
        write_path_template="/nsls2/data/fxi-new/legacy/Andor//%Y/%m/%d/",
        # write_path_template = '/dev/shm/',
        root="/nsls2/data/fxi-new/legacy/Andor/",
        # write_path_template='/tmp/',
        # root='/',
    )

    ac_period = Cpt(EpicsSignal, "cam1:AcquirePeriod")

    def stop(self):
        self.hdf5.capture.put(0)
        return super().stop()

    def pause(self):
        self.hdf5.capture.put(0)
        return super().pause()

    def resume(self):
        self.hdf5.capture.put(1)
        # The AD HDF5 plugin bumps its file_number and starts writing into a
        # *new file* because we toggled capturing off and on again.
        # Generate a new Resource document for the new file.

        # grab the stashed result from make_filename
        filename, read_path, write_path = self.hdf5._ret
        self.hdf5._fn = self.hdf5.file_template.get() % (
            read_path,
            filename,
            self.hdf5.file_number.get() - 1,
        )
        # file_number is *next*
        # iteration
        res_kwargs = {"frame_per_point": self.hdf5.get_frames_per_point()}
        self.hdf5._generate_resource(res_kwargs)
        return super().resume()


WPFS = Manta("XF:18IDA-BI{WPFS:1}", name="WPFS")
WPFS.read_attrs = ["hdf5", "stats1"]
WPFS.stats1.read_attrs = ["total"]
WPFS.hdf5.read_attrs = []

PMFS = Manta("XF:18IDA-BI{PMFS:1}", name="PMFS")
PMFS.read_attrs = ["hdf5", "stats1"]
PMFS.stats1.read_attrs = ["total"]
PMFS.hdf5.read_attrs = []

MFS = Manta("XF:18IDA-BI{MFS:1}", name="MFS")
MFS.read_attrs = ["hdf5", "stats1"]
MFS.stats1.read_attrs = ["total"]
MFS.hdf5.read_attrs = []

detA1 = Manta("XF:18IDB-BI{Det:A1}", name="detA1")
detA1.read_attrs = ["hdf5", "stats1"]
#detA1.read_attrs = ['hdf5']
detA1.read_attrs = ["hdf5", "stats1"]
detA1.stats1.read_attrs = ["total"]
# detA1.stats5.read_attrs = ['total']
detA1.hdf5.read_attrs = []

"""
# return to old version of Andor
Andor = Manta('XF:18IDB-BI{Det:Neo}', name='Andor')
#Andor.read_attrs = ['hdf5', 'stats1', 'stats5']
#Andor.read_attrs = ['hdf5']
Andor.read_attrs = ['hdf5', 'stats1']
Andor.stats1.read_attrs = ['total']
#Andor.stats5.read_attrs = ['total']
Andor.hdf5.read_attrs = []
"""


Andor = AndorKlass("XF:18IDB-BI{Det:Neo2}", name="Andor")
Andor.cam.ensure_nonblocking()
# Andor.read_attrs = ['hdf5', 'stats1', 'stats5']
Andor.read_attrs = ['hdf5']
#Andor.read_attrs = ["hdf5", "stats1"]
#Andor.stats1.read_attrs = ["total"]
# Andor.stats5.read_attrs = ['total']
Andor.hdf5.read_attrs = ["time_stamp"]
Andor.stage_sigs["cam.image_mode"] = 0
#for k in ("image", "stats1", "trans1", "roi1", "proc1"):
#    getattr(Andor, k).ensure_nonblocking()
for k in ("image", "trans1", "roi1", "proc1"):
    getattr(Andor, k).ensure_nonblocking()
Andor.hdf5.time_stamp.name = "Andor_timestamps"


# vlm = Manta("XF:18IDB-BI{VLM:1}", name="vlm")
# detA1.read_attrs = ['hdf5', 'stats1', 'stats5']
# detA1.read_attrs = ['hdf5']
# vlm.read_attrs = ["hdf5", "stats1"]
# vlm.stats1.read_attrs = ["total"]
# detA1.stats5.read_attrs = ['total']
# vlm.hdf5.read_attrs = []


#for det in [detA1, Andor]:
for det in [detA1]:
    det.stats1.total.kind = "hinted"
    # It does not work since it's not defined in the class, commenting out:
    # det.stats5.total.kind = 'hinted'

