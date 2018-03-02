import os
import sys
import shutil
import inspect
import time
import subprocess
import threading
from datetime import datetime


class Auto_Log_Save(object):
    '''
    Auto save the motor position into logfile (/NSLS2/xf18id1/DATA/Motor_position_log/) at 11pm everyday.
    '''
    def __init__(self, interval=1):
        self.interval = interval
        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def run(self):
        while True:
            now = datetime.now()
            if now.hour == 23: 
                save_pos(print_flag=0, comment='routine record')
                time.sleep(80000)

       

def save_pos(print_flag = 0, comment=''):
    '''
    Get motor positions and save to file /NSLS2/xf18id1/DATA/Motor_position_log/
    To print it out, set print_flag=1 
    '''
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush() # If you want the output to be visible immediately
        def flush(self) :
            for f in self.files:
                f.flush()

    now = datetime.now()
    year = np.str(now.year)
    mon  = '{:02d}'.format(now.month)
    day  = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minu = '{:02d}'.format(now.minute)
    current_date = year + '-' + mon + '-' + day
    
    fn = '/NSLS2/xf18id1/DATA/Motor_position_log/log-' + current_date + '_' + hour + '-' + minu + '.log' 
    f = open(fn,'w')
    '''
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    print('\n{0} {1}:{2}'.format(current_date, np.str(now.hour), np.str(now.minute)))
    wh_pos(comment)
    print('\nsaved to file: {}'.format(fn))
    sys.stdout = original
    '''
    f.write('\n{0} {1}:{2}\n'.format(current_date, hour, minu))
    lines = wh_pos(comment, 0)
    f.write('\n'.join(lines))
    f.write('\n\nsaved to file: {}'.format(fn))
    f.close()

    if print_flag:
        shutil.copyfile(fn,'/NSLS2/xf18id1/DATA/Motor_position_log/tmp.log')
        os.system("lp -o cpi=20 -o lpi=10 -o media='letter' -d HP_Color_LaserJet_M553 /NSLS2/xf18id1/DATA/Motor_position_log/tmp.log")


def wh_pos(comment='', print_on_screen=1):

    positioners = BlueskyMagics.positioners
    values = []
    for p in positioners:
        try:
            values.append(p.position)
        except Exception as exc:
            values.append(exc)

    headers = ['Positioner', 'Value', 'Unit', 'Low Limit', 'High Limit', 'Offset', 'Offset_dir', 
                    'Encoder Dial', 'Encoder Cnt.', 'Encoder Res', 'Motor Res', 'Motor Status']
    LINE_FMT = '{: <16} {: <12} {: <6} {: <12} {: <12} {: <12} {: <12} {: <14} {: <14} {: <14} {: <14} {: <12}'
    lines = []
    lines.append(str(comment) + '\n')
    
    lines.append(LINE_FMT.format(*headers))
    for p, v in zip(positioners, values):
        if not isinstance(v, Exception):
            try:
                prec = p.precision
            except Exception:
                prec = self.FMT_PREC
            value = np.round(v, decimals=prec)

            try:
                low_limit, high_limit = p.limits
            except Exception as exc:
                low_limit = high_limit = exc.__class__.__name__
            else:
                low_limit = np.round(low_limit, decimals=prec)
                high_limit = np.round(high_limit, decimals=prec)

            try:
                offset = p.user_offset.get()
            except Exception as exc:
                offset = exc.__class__.__name__
            else:
                offset = np.round(offset, decimals=prec)

            try:
                encoder = p.dial_readback.value
                counts = p.dial_counts.value
                encoder_res = p.encoder_res.value
                motor_res = p.motor_res.value               
                motor_velocity = p.velocity.value
                motor_stat = p.motor_stat.value              
                offset_dir = p.user_offset_dir.value
                motor_unit = p.motor_egu.value
                
            except Exception as exc:
                encoder = counts = motor_res = encoder_res = motor_velocity = motor_stat = motor_stat = offset_dir = motor_unit = exc.__class__.__name__
            else:
                encoder = np.round(encoder, decimals=prec)
                counts = np.round(counts, decimals=prec)     
                motor_stat = 'Alarmed' if motor_stat else 'Normal'
                motor_res = format(motor_res, '.5e')
                encoder_res = format(encoder_res, '.5e')
                motor_velocity = np.round(motor_velocity, decimals=prec)
                
        else:
            value = v.__class__.__name__  # e.g. 'DisconnectedError'
            low_limit = high_limit = offset = encoder = counts = motor_res = encoder_res = motor_velocity = ''


#       encoder, counts = get_encoder(p.prefix)
        lines.append(LINE_FMT.format(p.name, value, motor_unit, low_limit, high_limit, offset, offset_dir,
                    encoder, counts, encoder_res, motor_res, motor_stat))
    lines.append('\n##########\nPZT STATUS:\n')
    
    LINE_FMT = '{: <30} {: <11} {: <11} {: <11} {: <11} {: <11} {: <11}'
    PZT_header = ['Positioner', 'status', 'position', 'P_gain', 'I_gain', 'D_gain', 'Bender_force']
    lines.append(LINE_FMT.format(*PZT_header))
	
    pzt_dcm_chi2 = pzt('XF:18IDA-OP{Mir:DCM-Ax:Chi2Fine}', name='pzt_dcm_chi2',)
    pzt_dcm_th2  = pzt('XF:18IDA-OP{Mir:DCM-Ax:Th2Fine}', name='pzt_dcm_th2')
    pzt_tm = pzt('XF:18IDA-OP{Mir:TM-Ax:Bender}', name='pzt_tm', flag=1)
    pzt_cm = pzt('XF:18IDA-OP{Mir:CM-Ax:Bender}', name='pzt_cm', flag=1)

    pzt_motors = [pzt_dcm_chi2, pzt_dcm_th2, pzt_tm, pzt_cm]

    for p in pzt_motors: # pzt_motors is defined in 13-pzt.py
        lines.append(LINE_FMT.format(p.name, p.stat, p.pos, p.p_gain, p.i_gain, p.d_gain, p.bender))

#    if print_on_screen:   
#        print('\n'.join(lines))
#        return 
#    else:
#        return lines
    return lines

'''    
def get_encoder(motor_prefix):
    ENCODER = str(motor_prefix) + '.DRBV'
    COUNTS = str(motor_prefix) + '.RRBV'

    encoder = subprocess.check_output(['caget', ENCODER, '-t']).rstrip()
    encoder = str_convert(encoder)

    counts = subprocess.check_output(['caget', COUNTS, '-t']).rstrip()
    counts = str_convert(counts)

    return encoder, round(float(counts))


def get_pzt_position(pzt_prefix, flag=''):
    POS = str(pzt_prefix) + 'GET_POSITION'
    STAT = str(pzt_prefix) + 'GET_SERVO_STATE'
    PGAIN = str(pzt_prefix) + 'GET_SERVO_PGAIN'
    IGAIN = str(pzt_prefix) + 'GET_SERVO_IGAIN'
    DGAIN = str(pzt_prefix) + 'GET_SERVO_DGAIN'
    BENDER = str(pzt_prefix) + 'W-I'

    pos = subprocess.check_output(['caget', POS, '-t']).rstrip()
    pos = str_convert(pos)

    stat = subprocess.check_output(['caget', STAT, '-t']).rstrip()
    stat = str_convert(stat, 0)

    P_gain = subprocess.check_output(['caget', PGAIN, '-t']).rstrip()
    P_gain = str_convert(P_gain)

    I_gain = subprocess.check_output(['caget', IGAIN, '-t']).rstrip()
    I_gain = str_convert(I_gain)

    D_gain = subprocess.check_output(['caget', DGAIN, '-t']).rstrip()
    D_gain = str_convert(D_gain)

    if flag:
        Bender_force = subprocess.check_output(['caget', BENDER, '-t']).rstrip()
        Bender_force = str_convert(Bender_force)
    else:
        Bender_force = 'N/A'
    
    return stat, pos, P_gain, I_gain, D_gain, Bender_force

'''
def str_convert(my_string, flag=1):
    tmp = str(my_string)
    fmt = '{:3.4f}'
    output = tmp[2:len(tmp)-1]
    if flag:        return fmt.format(float(output))
    else: return output

