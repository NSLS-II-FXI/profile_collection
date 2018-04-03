
class pzt:
    def __init__(self, pzt_prefix, name, flag=0):
        self.name = name
        self.pos = EpicsSignal(str(pzt_prefix) + 'GET_POSITION', name = self.name + '_pos')
        self.p_gain = EpicsSignal(str(pzt_prefix) + 'GET_SERVO_PGAIN', name = self.name + '_p_gain')
        self.i_gain = EpicsSignal(str(pzt_prefix) + 'GET_SERVO_IGAIN', name = self.name + '_i_gain')
        self.d_gain = EpicsSignal(str(pzt_prefix) + 'GET_SERVO_DGAIN', name = self.name + '_d_gain')
        self.setting_pv = str(pzt_prefix) + 'SET_POSITION.A'
        self.setpos = EpicsSignal(self.setting_pv, name= 'setpos' )
        
        self.bender = self.pzt_bender(pzt_prefix, flag)
        self.stat = self.stat(pzt_prefix)

    def pzt_bender(self, pzt_prefix, flag):
        return EpicsSignal(str(pzt_prefix) + 'W-I').value if flag else 'None'

    def stat(self, pzt_prefix):
        return 'Enabled' if EpicsSignal(str(pzt_prefix) + 'GET_SERVO_STATE').value else 'Disabled'

pzt_dcm_chi2 = pzt('XF:18IDA-OP{Mir:DCM-Ax:Chi2Fine}', name='pzt_dcm_chi2')
pzt_dcm_th2  = pzt('XF:18IDA-OP{Mir:DCM-Ax:Th2Fine}', name='pzt_dcm_th2')
pzt_tm = pzt('XF:18IDA-OP{Mir:TM-Ax:Bender}', name='pzt_tm', flag=1)
pzt_cm = pzt('XF:18IDA-OP{Mir:CM-Ax:Bender}', name='pzt_cm', flag=1)

pzt_cm_loadcell = EpicsSignal('XF:18IDA-OP{Mir:CM-Ax:Bender}W-I', name='pzt_cm_loadcell')
pzt_tm_loadcell = EpicsSignal('XF:18IDA-OP{Mir:TM-Ax:Bender}W-I', name='pzt_tm_loadcell')

'''
pzt_motors = [pzt_dcm_chi2, pzt_dcm_th2, pzt_tm, pzt_cm]
'''



