from ophyd import AreaDetector, Component as Cpt
from ophyd.areadetector.trigger_mixins import SingleTrigger
from ophyd.areadetector.filestore_mixins import FileStoreHDF5IterativeWrite
from ophyd import (ImagePlugin,
                   StatsPlugin,
                   TransformPlugin,
                   ROIPlugin,
                   HDF5Plugin,
                   ProcessPlugin)


class HDF5PluginWithFileStore(HDF5Plugin, FileStoreHDF5IterativeWrite):
    # AD v2.2.0 (at least) does not have this. It is present in v1.9.1.
    file_number_sync = None

    def get_frames_per_point(self):
        return self.parent.cam.num_images.get()


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
               write_path_template='/NSLS2/xf18id1/DATA/detA1/commissioning',
               root='/NSLS2/xf18id1/DATA/detA1/',
               reg=None)  # placeholder to be set on instance as obj.hdf5.reg

    def stop(self):
        self.hdf5.capture.put(0)
        return super().stop()

    def pause(self):
        self.hdf5.capture.put(0)
        return super().pause()

    def resume(self):
        self.hdf5.capture.put(1)
        return super().resume()

    @property
    def hints(self):
        return {'fields': [self.stats1.total.name,
                           self.stats5.total.name]}


detA1 = Manta('XF:18IDB-BI{Det:A1}', name='detA1')
detA1.hdf5.reg = db.reg
detA1.hdf5._reg = db.reg
#detA1.read_attrs = ['hdf5', 'stats1', 'stats5']
#detA1.read_attrs = ['hdf5']
detA1.read_attrs = ['hdf5', 'stats1']
detA1.stats1.read_attrs = ['total']
#detA1.stats5.read_attrs = ['total']
detA1.hdf5.read_attrs = []



Andor = Manta('XF:18IDB-BI{Det:Neo}', name = 'Andor')
Andor.hdf5.reg = db.reg
Andor.hdf5._reg = db.reg
#Andor.read_attrs = ['hdf5', 'stats1', 'stats5']
#Andor.read_attrs = ['hdf5']
Andor.read_attrs = ['hdf5', 'stats1']
Andor.stats1.read_attrs = ['total']
#Andor.stats5.read_attrs = ['total']
Andor.hdf5.read_attrs = []
Andor.stage_sigs['cam.image_mode'] = 0
