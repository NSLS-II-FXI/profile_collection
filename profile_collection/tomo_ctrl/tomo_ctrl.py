import sys 
import os
import re
import json
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QRadioButton, QApplication,
                             QLineEdit, QWidget, QPushButton, QLabel, QGroupBox,
                             QVBoxLayout, QHBoxLayout, QGridLayout)

from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot

from datetime import datetime
import numpy as np

#from temp import tomo_test

global tomo

class App(QWidget):
    
    def __init__(self):
        super().__init__()
        self.title =  'Tomo Control'
        self.left = 400
        self.top = 400
        self.width = 800
        self.height = 480
        self.initUI()
        self.scan_mode = '3D'
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.font1 = QtGui.QFont('Arial', 11, QtGui.QFont.Bold)
        self.font2 = QtGui.QFont('Arial', 11, QtGui.QFont.Normal)
        self.fpath = os.getcwd()
        self.xeng = 8000 # need modify
        
#        lb_empty = QLabel()
        grid = QGridLayout()
        gpbox1 = self.GP_prepare()
        gpbox2 = self.GP_tomo_param()
        
        self.pb_run = QPushButton('Start scan')
        self.pb_run.setToolTip('Run tomo scan')
        self.pb_run.setFont(self.font1)
        self.pb_run.clicked.connect(run_tomo)
        self.pb_run.setFixedWidth(150)
        
        grid.addWidget(gpbox1,0,1)
        grid.addWidget(gpbox2,1,1)
        
        layout = QVBoxLayout()

        layout.addLayout(grid)
        layout.addWidget(QLabel())
        layout.addWidget(self.pb_run)

        self.setLayout(layout)
#        self.resize(640,200)
        
    def GP_prepare(self):
        
        lb_empty = QLabel()
        
        gpbox = QGroupBox('Preparation')
        gpbox.setFont(self.font1)

        lb_pid = QLabel(self)
        lb_pid.setText('Proposal ID:')
        lb_pid.setFont(self.font2)

        lb_or = QLabel(self)
        lb_or.setText(' or ')
        lb_or.setFont(self.font2)         
        
        self.tx_pid = QLineEdit(self) # proposal ID
        self.tx_pid.setValidator(QIntValidator())
        
        lb_pi = QLabel(self)
        lb_pi.setText('PI name:')
        lb_pi.setFont(self.font2)
            
        self.tx_pi = QLineEdit(self) # PI name

        self.pb_godir = QPushButton('Go to directory')
        self.pb_godir.setToolTip('create directory if not exist')
        self.pb_godir.setFont(self.font2)
        self.pb_godir.clicked.connect(go_to_directory)
        self.pb_godir.setFixedWidth(150)
        
        self.pb_new = QPushButton('New user')  
        self.pb_new.setToolTip('create a new user')
        self.pb_new.setFont(self.font2)
        self.pb_new.clicked.connect(new_user)    
        self.pb_new.setFixedWidth(150)
    
        self.lb_path = QLabel()
        self.lb_path.setFont(self.font2)
        self.lb_path.setText('Current directory:   {}'.format(self.fpath))
        
        pb_conf = QPushButton('Load config file')
        pb_conf.setFont(self.font2)
        pb_conf.setFixedWidth(150)
        pb_conf.clicked.connect(load_config_file)
        
        self.lb_conf = QLabel()
        self.lb_conf.setFont(self.font2)
#        self.lb_conf.setText('No configuration file selected ... ')
        
    # configure file info    
        lb_zd = QLabel()
        lb_zd.setFont(self.font2)
        lb_zd.setText('ZP Dia (um):')
        lb_zd.setFixedWidth(100)
        
        lb_zw = QLabel()
        lb_zw.setFont(self.font2)
        lb_zw.setText('ZP Width (nm):')
        lb_zw.setFixedWidth(100)
        
        lb_vlm = QLabel()
        lb_vlm.setFont(self.font2)
        lb_vlm.setText('VLM Mag.:')
        lb_vlm.setFixedWidth(100)
        
        lb_mag = QLabel()
        lb_mag.setFont(self.font2)
        lb_mag.setText('GLB Mag.:')
        lb_mag.setFixedWidth(100)
        
        lb_elow = QLabel()
        lb_elow.setFont(self.font2)
        lb_elow.setText('Low limit (ev).:')
        lb_elow.setFixedWidth(100)
        
        lb_ehigh = QLabel()
        lb_ehigh.setFont(self.font2)
        lb_ehigh.setText('High limit (ev).:')
        lb_ehigh.setFixedWidth(100)
        
        self.tx_zd = QLineEdit(self)
        self.tx_zd.setValidator(QIntValidator())
        self.tx_zd.setDisabled(True)
        
        self.tx_zw = QLineEdit(self)  
        self.tx_zw.setValidator(QIntValidator())
        self.tx_zw.setDisabled(True)
        
        self.tx_vlm = QLineEdit(self) 
        self.tx_vlm.setValidator(QIntValidator())
        self.tx_vlm.setDisabled(True)
        
        self.tx_mag = QLineEdit(self) 
        self.tx_mag.setValidator(QIntValidator())
        self.tx_mag.setDisabled(True)
        
        self.tx_elow = QLineEdit(self) 
        self.tx_elow.setValidator(QDoubleValidator())
        self.tx_elow.setDisabled(True)
        
        self.tx_ehigh = QLineEdit(self) 
        self.tx_ehigh.setValidator(QDoubleValidator())
        self.tx_ehigh.setDisabled(True)
        
        self.load_config_default()
    
        hbox1 = QHBoxLayout()
        hbox1.addWidget(lb_pid)
        hbox1.addWidget(self.tx_pid)
        hbox1.addWidget(lb_pi)
        hbox1.addWidget(self.tx_pi)
        
        hbox2 = QHBoxLayout()  
        hbox2.addWidget(self.pb_new)
        hbox2.addWidget(lb_or)
        hbox2.addWidget(self.pb_godir)
        hbox2.addWidget(self.lb_path)
        hbox2.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox3 = QHBoxLayout()  
        hbox3.addWidget(pb_conf)
        hbox3.addWidget(self.lb_conf)
        hbox3.setAlignment(QtCore.Qt.AlignLeft)
        
        hbox4 = QHBoxLayout()  
        hbox4.addWidget(lb_zd)
        hbox4.addWidget(self.tx_zd)
        hbox4.addWidget(lb_zw)
        hbox4.addWidget(self.tx_zw)
        hbox4.addWidget(lb_mag)
        hbox4.addWidget(self.tx_mag)
        
        hbox5 = QHBoxLayout()  
        hbox5.addWidget(lb_vlm)
        hbox5.addWidget(self.tx_vlm)
        hbox5.addWidget(lb_elow)
        hbox5.addWidget(self.tx_elow)
        hbox5.addWidget(lb_ehigh)
        hbox5.addWidget(self.tx_ehigh)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addWidget(lb_empty)
        
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)
        gpbox.setLayout(vbox)
        
        return gpbox


    def GP_tomo_param(self):
        gpbox = QGroupBox('Tomo parameters')
        gpbox.setFont(self.font1)
        
        lb_empty = QLabel()
        
        lb_eng = QLabel()
        lb_eng.setFont(self.font2)
        lb_eng.setText('X-ray Energy:')
        lb_eng.setFixedWidth(150)
        
        
# X-ray energy related
        self.tx_eng = QLineEdit(self)
        self.tx_eng.setFont(self.font2)
#        self.tx_eng.setValidator(QDoubleValidator())
        
        self.pb_eng = QPushButton('Gen. Energ. list')
        self.pb_eng.setToolTip('Energy is in unit of eV')
        self.pb_eng.setFont(self.font2)
        self.pb_eng.setFixedWidth(150)
        self.pb_eng.clicked.connect(gen_eng_list)
        
        self.pb_goeng = QPushButton('Goto 1st energy')
        self.pb_goeng.setFont(self.font2)
        self.pb_goeng.setFixedWidth(150)
        self.pb_goeng.clicked.connect(go_to_energy)
        
        
        self.lb_c_eng = QLabel()
        self.lb_c_eng.setFont(self.font1)
        self.lb_c_eng.setText('Current Energy:   {0}  ({1} - {2} eV)'.format(self.xeng, (self.tx_elow.text()), (self.tx_ehigh.text())))
        self.lb_c_eng.setFixedWidth(400)
    
        lb_nt = QLabel()
        lb_nt.setFont(self.font2)
        lb_nt.setText('e.g., single/multiple energy: [8000,8100,8200]; with 1eV interval: [8000:1:8500]; combination: [8000, 8100];[8032:1:8060]')
    
        lb_eng_list = QLabel()
        lb_eng_list.setFont(self.font1)
        lb_eng_list.setText('Energy list:')        

        self.lb_eng_list2 = QLabel()
        self.lb_eng_list2.setFont(self.font2)
        self.lb_eng_list2.setText('No energy entered...')  
        self.lb_eng_list2.setFixedWidth(600)
    
        lb_md = QLabel()
        lb_md.setFont(self.font1)
        lb_md.setText('Scan mode:')
        
        self.rd_md_2d = QRadioButton('2D')
        self.rd_md_2d.toggled.connect(self.check_scan_mode)
        self.rd_md_3d = QRadioButton('3D')
        self.rd_md_3d.setChecked(True)
        
        
        hbox1 = QHBoxLayout()
        hbox1.addWidget(lb_eng)
        hbox1.addWidget(self.tx_eng)
        hbox1.addWidget(self.pb_eng)
        hbox1.addWidget(self.pb_goeng)
        
        hbox2 = QHBoxLayout()
        hbox2.addWidget(lb_eng_list)
        hbox2.addWidget(self.lb_eng_list2)
        hbox2.setAlignment(QtCore.Qt.AlignLeft)
#        hbox2.addStretch(1)
        

        vbox = QVBoxLayout()
        vbox.addWidget(self.lb_c_eng)
        vbox.addLayout(hbox1)
        vbox.addWidget(lb_nt) 
#        vbox.addWidget(lb_empty)
        vbox.addLayout(hbox2)
        vbox.addWidget(lb_empty)

        
        
# other control parameters
    # Angle infor
        lb_as = QLabel()
        lb_as.setFont(self.font2)
        lb_as.setText('Angle start:')
        lb_as.setFixedWidth(130)
        
        self.tx_as  =QLineEdit(self)
        self.tx_as.setFont(self.font2)
        self.tx_as.setValidator(QDoubleValidator())
        
        lb_ae = QLabel()
        lb_ae.setFont(self.font2)
        lb_ae.setText('end:')
        lb_ae.setFixedWidth(80)
        
        self.tx_ae  =QLineEdit(self)
        self.tx_ae.setFont(self.font2)
        self.tx_ae.setValidator(QDoubleValidator())
              
        lb_ai = QLabel()
        lb_ai.setFont(self.font2)
        lb_ai.setText('interval:')
        lb_ai.setFixedWidth(80)
        
        self.tx_ai  =QLineEdit(self)
        self.tx_ai.setFont(self.font2)
        self.tx_ai.setValidator(QDoubleValidator())
        
        hbox3 = QHBoxLayout()
        hbox3.addWidget(lb_as)
        hbox3.addWidget(self.tx_as)
        hbox3.addWidget(lb_ae)
        hbox3.addWidget(self.tx_ae)
        hbox3.addWidget(lb_ai)
        hbox3.addWidget(self.tx_ai)
        
        
    # others    
        lb_t = QLabel()
        lb_t.setFont(self.font2)
        lb_t.setText('Exposure time (s):')
        lb_t.setFixedWidth(130)
        
        lb_bkg = QLabel()
        lb_bkg.setFont(self.font2)
        lb_bkg.setText('# of Bkg:')    
        lb_bkg.setFixedWidth(80)
               
        lb_dk = QLabel()
        lb_dk.setFont(self.font2)
        lb_dk.setText('# of Dark:')  
        lb_dk.setFixedWidth(80)

        self.tx_t = QLineEdit(self)
        self.tx_t.setFont(self.font2)
        self.tx_t.setValidator(QDoubleValidator())
        self.tx_bkg = QLineEdit(self)
        self.tx_bkg.setFont(self.font2)
        self.tx_bkg.setValidator(QIntValidator())
        self.tx_dk = QLineEdit(self)
        self.tx_dk.setFont(self.font2)
        self.tx_dk.setValidator(QIntValidator())
        
        lb_splx = QLabel()
        lb_splx.setFont(self.font2)
        lb_splx.setText('Sample out x (um):')
        lb_splx.setFixedWidth(130)
        
        lb_sply = QLabel()
        lb_sply.setFont(self.font2)
        lb_sply.setText('y (um):')
        lb_sply.setFixedWidth(80)
        
        lb_splz = QLabel()
        lb_splz.setFont(self.font2)
        lb_splz.setText('z (um):')
        lb_splz.setFixedWidth(80)

        self.tx_splx = QLineEdit(self)
        self.tx_splx.setFont(self.font2)
        self.tx_splx.setValidator(QDoubleValidator())
        self.tx_sply = QLineEdit(self)
        self.tx_sply.setFont(self.font2)
        self.tx_sply.setValidator(QDoubleValidator())
        self.tx_splz = QLineEdit(self)
        self.tx_splz.setFont(self.font2)
        self.tx_splz.setValidator(QDoubleValidator())       
        

        hbox4 = QHBoxLayout()
        hbox4.addWidget(lb_t)
        hbox4.addWidget(self.tx_t)       
        hbox4.addWidget(lb_bkg)
        hbox4.addWidget(self.tx_bkg)
        hbox4.addWidget(lb_dk)
        hbox4.addWidget(self.tx_dk)
        
        hbox5 = QHBoxLayout()
        hbox5.addWidget(lb_splx)
        hbox5.addWidget(self.tx_splx) 
        hbox5.addWidget(lb_sply)
        hbox5.addWidget(self.tx_sply) 
        hbox5.addWidget(lb_splz)
        hbox5.addWidget(self.tx_splz) 
        
        hbox_md = QHBoxLayout()
        hbox_md.addWidget(lb_md)
        hbox_md.addWidget(self.rd_md_2d)
        hbox_md.addWidget(self.rd_md_3d)
        hbox_md.setAlignment(QtCore.Qt.AlignLeft)
        
        vbox.addLayout(hbox_md)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
#        vbox.addWidget(lb_empty)
        vbox.addLayout(hbox5)
        
        gpbox.setLayout(vbox)
        
        return gpbox

    @pyqtSlot()
    def on_click(self):
        print(self.title)
        
    def load_config_default(self):
        fn_pre = '/home/xf18id/.ipython/profile_collection/startup/tomo_ctrl/'
        fn_f = 'zp_D100_30_vlm10_gMag650_(7000_8000).json' 
        fn = fn_pre + fn_f
        self.lb_conf.setText('{} loaded'.format(fn_f))
        with open(fn) as f:
            conf = json.load(f)
            self.tx_zd.setText(str(conf['ZONE_DIAMETER']))           
            self.tx_zd.setStyleSheet('color: rgb(80, 80, 80);')
            self.tx_zw.setText(str(conf['OUT_ZONE_WIDTH']))           
            self.tx_zw.setStyleSheet('color: rgb(80, 80, 80);')
            self.tx_mag.setText(str(conf['GLOBAL_MAG']))           
            self.tx_mag.setStyleSheet('color: rgb(80, 80, 80);')
            self.tx_vlm.setText(str(conf['GLOBAL_VLM_MAG']))           
            self.tx_vlm.setStyleSheet('color: rgb(80, 80, 80);')
            self.tx_elow.setText(str(conf['Eng_L']))           
            self.tx_elow.setStyleSheet('color: rgb(80, 80, 80);')
            self.tx_ehigh.setText(str(conf['Eng_H']))           
            self.tx_ehigh.setStyleSheet('color: rgb(80, 80, 80);')
        f.close()
        
    def check_scan_mode(self):
        if self.rd_md_2d.isChecked() == True:
            self.tx_as.setDisabled(True)
            self.tx_ae.setDisabled(True)
            self.tx_ai.setDisabled(True)
            self.tx_bkg.setDisabled(True)
            self.tx_dk.setDisabled(True)
            self.tx_splx.setDisabled(True)
            self.tx_sply.setDisabled(True)
            self.tx_splz.setDisabled(True)
            self.scan_mode = '2D'
        else:
            self.tx_as.setEnabled(True)
            self.tx_ae.setEnabled(True)
            self.tx_ai.setEnabled(True)
            self.tx_bkg.setEnabled(True)
            self.tx_dk.setEnabled(True)
            self.tx_splx.setEnabled(True)
            self.tx_sply.setEnabled(True)
            self.tx_splz.setEnabled(True)
            self.scan_mode = '3D'            
            
            
        
def new_user():
    now = datetime.now()
    year = np.str(now.year)

    if now.month >= 1 and now.month <=4:    qut = 'Q1'
    elif now.month >= 5 and now.month <=8:  qut = 'Q2'
    else: qut = 'Q3'

    PI_name = str(tomo.tx_pi.text())
    PI_name = PI_name.replace(' ', '_').upper()
    proposal_id = str(tomo.tx_pid.text())
    
    if len(PI_name) == 0 or len(proposal_id) == 0:
        tomo.lb_path.setText('Please input proposal ID and PI name')
        return 0
    pre = '/home/mingyuan/test/' + year + qut + '/' # need modify
    fn = pre + PI_name + '_Proposal_' + proposal_id
    try:        os.makedirs(fn)
    except Exception:
        print('(user, proposal) existed\nEntering folder: {}'.format(os.getcwd()))
        os.chdir(fn)
        pass
    os.chdir(fn)
    tomo.lb_path.setText('current directory:   {}'.format(fn))
    
    
        
def go_to_directory():
    fn = str(QFileDialog.getExistingDirectory(tomo, "Select Directory"))
    tomo.lb_path.setText('Current directory:   {}'.format(fn))
    print('go_to_directory: {}'.format(fn))
    print('{}'.format(tomo.tx_pid.text()))
    
def go_to_energy():
    print('go_to_energy')
    print(clens.x.position)


    tomo.lb_c_eng.setText('Current Energy:   {} eV'.format(tomo.xeng[0]))
    
def run_tomo():
    print('start tomo scan')
    

def load_config_file():
    options = QFileDialog.Option()
    options |= QFileDialog.DontUseNativeDialog
    fn_conf, _ = QFileDialog.getOpenFileName(tomo, "QFileDialog.getOpenFileName()", "", "json files (*.json)", options=options)
    if fn_conf:
        print(fn_conf)
        fn_relative = fn_conf.split('/')[-1]
        tomo.lb_conf.setText('{} loaded'.format(fn_relative))
        
        with open(fn_conf) as f:
            conf = json.load(f)
            tomo.tx_zd.setText(str(conf['ZONE_DIAMETER']))           
            tomo.tx_zd.setStyleSheet('color: rgb(80, 80, 80);')
            tomo.tx_zw.setText(str(conf['OUT_ZONE_WIDTH']))           
            tomo.tx_zw.setStyleSheet('color: rgb(80, 80, 80);')
            tomo.tx_mag.setText(str(conf['GLOBAL_MAG']))           
            tomo.tx_mag.setStyleSheet('color: rgb(80, 80, 80);')
            tomo.tx_vlm.setText(str(conf['GLOBAL_VLM_MAG']))           
            tomo.tx_vlm.setStyleSheet('color: rgb(80, 80, 80);')
            tomo.tx_elow.setText(str(conf['Eng_L']))           
            tomo.tx_elow.setStyleSheet('color: rgb(80, 80, 80);')
            tomo.tx_ehigh.setText(str(conf['Eng_H']))           
            tomo.tx_ehigh.setStyleSheet('color: rgb(80, 80, 80);')
        f.close()
    
    

def gen_eng_list():
    txt = str(tomo.tx_eng.text())
    txt = txt.replace(' ','')
    if txt[0] != '[':
        invalid_flag = 1
    else:
        txt = txt.split(';')
        invalid_flag = 0
        eng_list = []
        for i in range(len(txt)):
            tmp = txt[i]
            if tmp[0] != '[' or tmp[-1] != ']':
                invalid_flag = 1
                print('{}'.format(i))
                continue
            num_l = len([a.start() for a in re.finditer('(?=\[)', tmp)])
            num_r = len([a.start() for a in re.finditer('(?=\])', tmp)])
            if num_l != num_r: 
                invalid_flag = 2
                continue
            tmp = tmp.replace('[', '')
            tmp = tmp.replace(']', '')
            num_colon = len([a.start() for a in re.finditer('(?=:)', tmp)])  
            if num_colon == 2:
                tmp_eng = tmp.split(':')
                st = float(tmp_eng[0])
                it = float(tmp_eng[1])
                ed = float(tmp_eng[2]) + it
                tmp_eng_list = list(np.arange(st, ed, it))
            elif num_colon == 0:
                tmp_eng = tmp.split(',')
                tmp_eng_list = [float(t) for t in tmp_eng]
            else:
                invalid_flag = 3
                continue
            
            eng_list = eng_list + tmp_eng_list
        
        eng_list = np.sort(np.unique(np.array(eng_list)))
    
    if invalid_flag:
        tomo.xeng = [0]
        eng_list = [0]
        tomo.lb_eng_list2.setText('Invalid energy!')
    else:
        tomo.xeng = eng_list
        tomo.lb_eng_list2.setText(str(eng_list))
#        tomo.setFixedSize(tomo.layout.sizeHint())
        
def gen_json():        
    fn_conf = 'zp_D100_30_vlm10_gMag650_(7000_8000).json'        
    conf={'ZONE_DIAMETER': 100,
         'OUT_ZONE_WIDTH': 30,
         'GLOBAL_MAG': 650,
         'GLOBAL_VLM_MAG': 10,
         'Eng_L': 7000,
         'Eng_H': 8000,
            }  
    with open(fn_conf, 'w') as f:
        json.dump(conf, f)
        
        
    

    
if __name__ == '__main__':

    app = QApplication(sys.argv)
    tomo = App()

    tomo.show()
    
    sys.exit(app.exec_())
