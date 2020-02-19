from ophyd import Component as Cpt
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd import EpicsSignal, AreaDetector
from ophyd import (
    ImagePlugin,
    StatsPlugin,
    TransformPlugin,
    ROIPlugin,
    HDF5Plugin,
    ProcessPlugin,
)

from ophyd.areadetector.plugins import HDF5Plugin, ProcessPlugin
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd.areadetector.trigger_mixins import (
    TriggerBase,
    ADTriggerStatus,
    SingleTrigger,
)
from ophyd import Component as Cpt, Device, EpicsSignal, EpicsSignalRO
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.detectors import DetectorBase
from ophyd.device import Staged
import time as ttime
from nslsii.ad33 import SingleTriggerV33, StatsPluginV33, CamV33Mixin


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

    def get_frames_per_point(self):
        return self.parent.cam.num_images.get()

    def make_filename(self):
        # stash this so that it is available on resume
        self._ret = super().make_filename()
        return self._ret


class AndorKlass(SingleTriggerV33, DetectorBase):
    cam = Cpt(AndorCam, "cam1:")
    image = Cpt(ImagePlugin, "image1:")
    stats1 = Cpt(StatsPluginV33, "Stats1:")
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
        write_path_template="/NSLS2/xf18id1/DATA/Andor/%Y/%m/%d/",
        # write_path_template='/dev/shm/%Y/%m/%d/' ,
        root="/NSLS2/xf18id1/DATA/Andor",
        # write_path_template='/tmp/',
        # root='/dev/shm',
        reg=None,
    )  # placeholder to be set on instance as obj.hdf5.reg

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

    def stage(self):
        import itertools

        for j in itertools.count():
            try:
                return super().stage()
            except TimeoutError:
                N_try = 20
                if j < 20:
                    print(f"failed to stage on try{j}/{N_try}, may try again")
                    continue
                else:
                    raise

    def unstage(self, *args, **kwargs):
        import itertools

        for j in itertools.count():
            try:
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
        write_path_template="/NSLS2/xf18id1/DATA/Andor/%Y/%m/%d/",
        # write_path_template = '/dev/shm/',
        root="/NSLS2/xf18id1/DATA/Andor",
        # write_path_template='/tmp/',
        # root='/',
        reg=None,
    )  # placeholder to be set on instance as obj.hdf5.reg

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
WPFS.hdf5.reg = db.reg
WPFS.hdf5._reg = db.reg
WPFS.read_attrs = ["hdf5", "stats1"]
WPFS.stats1.read_attrs = ["total"]
WPFS.hdf5.read_attrs = []

PMFS = Manta("XF:18IDA-BI{PMFS:1}", name="PMFS")
PMFS.hdf5.reg = db.reg
PMFS.hdf5._reg = db.reg
PMFS.read_attrs = ["hdf5", "stats1"]
PMFS.stats1.read_attrs = ["total"]
PMFS.hdf5.read_attrs = []

MFS = Manta("XF:18IDA-BI{MFS:1}", name="MFS")
MFS.hdf5.reg = db.reg
MFS.hdf5._reg = db.reg
MFS.read_attrs = ["hdf5", "stats1"]
MFS.stats1.read_attrs = ["total"]
MFS.hdf5.read_attrs = []

detA1 = Manta("XF:18IDB-BI{Det:A1}", name="detA1")
detA1.hdf5.reg = db.reg
detA1.hdf5._reg = db.reg
# detA1.read_attrs = ['hdf5', 'stats1', 'stats5']
# detA1.read_attrs = ['hdf5']
detA1.read_attrs = ["hdf5", "stats1"]
detA1.stats1.read_attrs = ["total"]
# detA1.stats5.read_attrs = ['total']
detA1.hdf5.read_attrs = []

"""
# return to old version of Andor
Andor = Manta('XF:18IDB-BI{Det:Neo}', name='Andor')
Andor.hdf5.reg = db.reg
Andor.hdf5._reg = db.reg
#Andor.read_attrs = ['hdf5', 'stats1', 'stats5']
#Andor.read_attrs = ['hdf5']
Andor.read_attrs = ['hdf5', 'stats1']
Andor.stats1.read_attrs = ['total']
#Andor.stats5.read_attrs = ['total']
Andor.hdf5.read_attrs = []
"""


Andor = AndorKlass("XF:18IDB-BI{Det:Neo}", name="Andor")
Andor.cam.ensure_nonblocking()
Andor.hdf5.reg = db.reg
Andor.hdf5._reg = db.reg
# Andor.read_attrs = ['hdf5', 'stats1', 'stats5']
# Andor.read_attrs = ['hdf5']
Andor.read_attrs = ["hdf5", "stats1"]
Andor.stats1.read_attrs = ["total"]
# Andor.stats5.read_attrs = ['total']
Andor.hdf5.read_attrs = []
Andor.stage_sigs["cam.image_mode"] = 0
for k in ("image", "stats1", "trans1", "roi1", "proc1"):
    getattr(Andor, k).ensure_nonblocking()


vlm = Manta("XF:18IDB-BI{VLM:1}", name="vlm")
vlm.hdf5.reg = db.reg
vlm.hdf5._reg = db.reg
# detA1.read_attrs = ['hdf5', 'stats1', 'stats5']
# detA1.read_attrs = ['hdf5']
vlm.read_attrs = ["hdf5", "stats1"]
vlm.stats1.read_attrs = ["total"]
# detA1.stats5.read_attrs = ['total']
vlm.hdf5.read_attrs = []


for det in [detA1, Andor]:
    det.stats1.total.kind = "hinted"
    # It does not work since it's not defined in the class, commenting out:
    # det.stats5.total.kind = 'hinted'
