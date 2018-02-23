
class pzt:
    def __init__(self, pzt_prefix, name, flag=0):
        self.name = name
        self.pos = np.round(EpicsSignalRO(str(pzt_prefix) + 'GET_POSITION').value, decimals=4)
        self.p_gain = np.round(EpicsSignalRO(str(pzt_prefix) + 'GET_SERVO_PGAIN').value, decimals=1)
        self.i_gain = np.round(EpicsSignalRO(str(pzt_prefix) + 'GET_SERVO_IGAIN').value, decimals=1)
        self.d_gain = np.round(EpicsSignalRO(str(pzt_prefix) + 'GET_SERVO_DGAIN').value, decimals=1)
        self.setting_pv = str(pzt_prefix) + 'SET_POSITION.A'
        self.getting_pv = str(pzt_prefix) + 'GET_POSITION'

        self.bender = self.pzt_bender(pzt_prefix, flag)
        self.stat = self.stat(pzt_prefix)

    def pzt_bender(self, pzt_prefix, flag):
        return EpicsSignalRO(str(pzt_prefix) + 'W-I').value if flag else 'None'

    def stat(self, pzt_prefix):
        return 'Enabled' if EpicsSignalRO(str(pzt_prefix) + 'GET_SERVO_STATE').value else 'Disabled'

pzt_dcm_chi2 = pzt('XF:18IDA-OP{Mir:DCM-Ax:Chi2Fine}', name='pzt_dcm_chi2',)
pzt_dcm_th2  = pzt('XF:18IDA-OP{Mir:DCM-Ax:Th2Fine}', name='pzt_dcm_th2')
pzt_tm = pzt('XF:18IDA-OP{Mir:TM-Ax:Bender}', name='pzt_tm', flag=1)
pzt_cm = pzt('XF:18IDA-OP{Mir:CM-Ax:Bender}', name='pzt_cm', flag=1)
'''
pzt_motors = [pzt_dcm_chi2, pzt_dcm_th2, pzt_tm, pzt_cm]
'''
