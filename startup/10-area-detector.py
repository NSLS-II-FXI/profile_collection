from ophyd import Component as Cpt
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd import EpicsSignal, AreaDetector
from ophyd import (ImagePlugin,
                   StatsPlugin,
                   TransformPlugin,
                   ROIPlugin,
                   HDF5Plugin,
                   ProcessPlugin)

from ophyd.areadetector.plugins import HDF5Plugin, ProcessPlugin
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd.areadetector.trigger_mixins import TriggerBase, ADTriggerStatus, SingleTrigger
from ophyd import Component as Cpt, Device, EpicsSignal, EpicsSignalRO
from ophyd.areadetector.cam import AreaDetectorCam
from ophyd.areadetector.detectors import DetectorBase
from ophyd.device import Staged
import time as ttime

class v33_plugin_mixin(Device):
    adcore_version = Cpt(EpicsSignalRO, 'ADCoreVersion_RBV', string=True, kind='config')
    driver_version = Cpt(EpicsSignalRO, 'DriverVersion_RBV', string=True, kind='config')

    
class v33_cam_mixin(v33_plugin_mixin):
    wait_for_plugins = Cpt(EpicsSignal, 'WaitForPlugins', string=True, kind='config')

    
class v33_file_mixin(v33_plugin_mixin):
    create_directories = Cpt(EpicsSignal, 'CreateDirectory', kind='config')

    
class ProcessPlugin33(ProcessPlugin, v33_plugin_mixin):
    ...

class AndorCam(v33_cam_mixin, AreaDetectorCam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_sigs['wait_for_plugins'] = 'Yes'

class SingleTrigger33(TriggerBase):
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

        self._status = self._status_type(self)

        def _acq_done(*args, **kwargs):
            # TODO sort out if anything useful in here
            self._status._finished()

        self._acquisition_signal.put(1, use_complete=True, callback=_acq_done)
        self.dispatch(self._image_name, ttime.time())
        return self._status

class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    # AD v2.2.0 (at least) does not have this. It is present in v1.9.1.
    file_number_sync = None

    def get_frames_per_point(self):
        return self.parent.cam.num_images.get()


class AndorKlass(SingleTrigger33, DetectorBase):
    cam = Cpt(AndorCam, 'cam1:')
    image = Cpt(ImagePlugin, 'image1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')
#    stats2 = Cpt(StatsPlugin, 'Stats2:')
#    stats3 = Cpt(StatsPlugin, 'Stats3:')
#    stats4 = Cpt(StatsPlugin, 'Stats4:')
#    stats5 = Cpt(StatsPlugin, 'Stats5:')
    trans1 = Cpt(TransformPlugin, 'Trans1:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
#    roi2 = Cpt(ROIPlugin, 'ROI2:')
#    roi3 = Cpt(ROIPlugin, 'ROI3:')
#    roi4 = Cpt(ROIPlugin, 'ROI4:')
    proc1 = Cpt(ProcessPlugin33, 'Proc1:')
    
    hdf5 = Cpt(HDF5PluginWithFileStore,
               suffix='HDF1:',
               write_path_template='/NSLS2/xf18id1/DATA/Andor/%Y/%m/%d/',
               root='/NSLS2/xf18id1/DATA/Andor',
               # write_path_template='/tmp/',
               # root='/',
               reg=None)  # placeholder to be set on instance as obj.hdf5.reg
    
    ac_period = Cpt(EpicsSignal, 'cam1:AcquirePeriod')    
    def stop(self):
        self.hdf5.capture.put(0)
        return super().stop()

    def pause(self):
        self.hdf5.capture.put(0)
        return super().pause()

    def resume(self):
        self.hdf5.capture.put(1)
        return super().resume()

class Manta(SingleTrigger, AreaDetector):
    image = Cpt(ImagePlugin, 'image1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')
#    stats2 = Cpt(StatsPlugin, 'Stats2:')
#    stats3 = Cpt(StatsPlugin, 'Stats3:')
#    stats4 = Cpt(StatsPlugin, 'Stats4:')
#    stats5 = Cpt(StatsPlugin, 'Stats5:')
    trans1 = Cpt(TransformPlugin, 'Trans1:')
    roi1 = Cpt(ROIPlugin, 'ROI1:')
#    roi2 = Cpt(ROIPlugin, 'ROI2:')
#    roi3 = Cpt(ROIPlugin, 'ROI3:')
#    roi4 = Cpt(ROIPlugin, 'ROI4:')
    proc1 = Cpt(ProcessPlugin, 'Proc1:')
    
    hdf5 = Cpt(HDF5PluginWithFileStore,
               suffix='HDF1:',
               write_path_template='/NSLS2/xf18id1/DATA/Andor/%Y/%m/%d/',
               root='/NSLS2/xf18id1/DATA/Andor',
               # write_path_template='/tmp/',
               # root='/',
               reg=None)  # placeholder to be set on instance as obj.hdf5.reg
    
    ac_period = Cpt(EpicsSignal, 'cam1:AcquirePeriod')    
    def stop(self):
        self.hdf5.capture.put(0)
        return super().stop()

    def pause(self):
        self.hdf5.capture.put(0)
        return super().pause()

    def resume(self):
        self.hdf5.capture.put(1)
        return super().resume()


WPFS = Manta('XF:18IDA-BI{WPFS:1}', name='WPFS')
WPFS.hdf5.reg = db.reg
WPFS.hdf5._reg = db.reg
WPFS.read_attrs = ['hdf5', 'stats1']
WPFS.stats1.read_attrs = ['total']
WPFS.hdf5.read_attrs = []

PMFS = Manta('XF:18IDA-BI{PMFS:1}', name='PMFS')
PMFS.hdf5.reg = db.reg
PMFS.hdf5._reg = db.reg
PMFS.read_attrs = ['hdf5', 'stats1']
PMFS.stats1.read_attrs = ['total']
PMFS.hdf5.read_attrs = []

MFS = Manta('XF:18IDA-BI{MFS:1}', name='MFS')
MFS.hdf5.reg = db.reg
MFS.hdf5._reg = db.reg
MFS.read_attrs = ['hdf5', 'stats1']
MFS.stats1.read_attrs = ['total']
MFS.hdf5.read_attrs = []

detA1 = Manta('XF:18IDB-BI{Det:A1}', name='detA1')
detA1.hdf5.reg = db.reg
detA1.hdf5._reg = db.reg
#detA1.read_attrs = ['hdf5', 'stats1', 'stats5']
#detA1.read_attrs = ['hdf5']
detA1.read_attrs = ['hdf5', 'stats1']
detA1.stats1.read_attrs = ['total']
#detA1.stats5.read_attrs = ['total']
detA1.hdf5.read_attrs = []

'''
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
'''


Andor = AndorKlass('XF:18IDB-BI{Det:Neo}', name = 'Andor')
Andor.hdf5.reg = db.reg
Andor.hdf5._reg = db.reg
#Andor.read_attrs = ['hdf5', 'stats1', 'stats5']
#Andor.read_attrs = ['hdf5']
Andor.read_attrs = ['hdf5', 'stats1']
Andor.stats1.read_attrs = ['total']
#Andor.stats5.read_attrs = ['total']
Andor.hdf5.read_attrs = []
Andor.stage_sigs['cam.image_mode'] = 0
for k in ('image', 'stats1', 'trans1', 'roi1', 'proc1'):
    getattr(Andor, k).ensure_nonblocking()

for det in [detA1, Andor]:
    det.stats1.total.kind = 'hinted'
    # It does not work since it's not defined in the class, commenting out:
    # det.stats5.total.kind = 'hinted'

