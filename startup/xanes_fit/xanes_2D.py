#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:39:15 2018

@author: mingyuan
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:07:42 2018

@author: mingyuan
"""

import sys
import os


from PyQt5 import QtGui
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QRadioButton, QApplication,
                             QLineEdit, QWidget, QPushButton, QLabel, QGroupBox,
                             QScrollBar, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QListWidget, QListWidgetItem, QAbstractItemView, QScrollArea,
                             QComboBox, QButtonGroup, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5 import QtCore
import numpy as np
from skimage import io
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from image_util import align_img
from scipy.ndimage.interpolation import shift
from scipy.signal import medfilt2d
from scipy.interpolate import interp1d
from xanes_util import fit_2D_xanes_non_iter, fit_2D_xanes_iter, normalize_2D_xanes, normalize_1D_xanes
import tifffile
import pandas as pd
import psutil
import os
import threading
import time

global xanes
#
#
# class bkg_memory_check(object):
#
#     def __init__(self, interval = 1):
#         self.interval = interval
#         thread = threading.Thread(target = self.run, args=())
#         thread.daemon = True
#         thread.start()
#
#     def run(self):
#         global xanes
#         while True:
#             PID = os.getpid()
#             py = psutil.Process(PID)
#             MEM = list(psutil.virtual_memory())
#             prog_used_mem = py.memory_info()[0]
#             MEM.append(prog_used_mem)
#             prog_used = '{:4.1f}'.format(MEM[-1] / 2. ** 30)  # unit in Gb
#             tot_mem = '{:4.1f}'.format(MEM[0] / 2. ** 30)
#             mem_pecent = '{:2.1f}'.format(MEM[2])
#             xanes.lb_pid_display.setText(str(PID))
#             xanes.lb_mem_prog_used.setText(prog_used + ' / ' + tot_mem + 'Gb')
#             xanes.lb_mem_avail.setText(mem_pecent)
#             # MEM = [total, available, percent, used, free, active, inactive, buffers, cached, shared, current_program_used]
#             time.sleep(self.interval)


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'XANES Control'
        screen_resolution = QApplication.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
        self.width = 1020
        self.height = 800
        self.left = (width - self.width) // 2
        self.top = (height - self.height) // 2
        self.initUI()
        self.bkg_memory_check()
        self.default_layout()


    def bkg_memory_check(self):
        thread = threading.Thread(target=self.bkg_memory_check_run, args=())
        thread.daemon = True
        thread.start()

    def bkg_memory_check_run(self):
        while True:
            PID = os.getpid()
            py = psutil.Process(PID)
            MEM = list(psutil.virtual_memory())
            prog_used_mem = py.memory_info()[0]
            MEM.append(prog_used_mem)
            prog_used = '{:4.1f}'.format(MEM[-1] / 2. ** 30)  # unit in Gb
            tot_mem = '{:4.1f}'.format(MEM[0] / 2. ** 30)
            mem_pecent = '{:2.1f}'.format(100 - MEM[2])
            self.lb_pid_display.setText(str(PID))
            self.lb_mem_prog_used.setText(prog_used + ' / ' + tot_mem + 'Gb')
            self.lb_mem_avail.setText(mem_pecent + ' %')
            if float(mem_pecent) < 10:
                self.lb_mem_avail.setStyleSheet('color: rgb(200, 50, 50);')
            else:
                self.lb_mem_avail.setStyleSheet('color: rgb(0, 0, 0);')
            # MEM = [total, available, percent, used, free, active, inactive, buffers, cached, shared, current_program_used]
            time.sleep(1)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.font1 = QtGui.QFont('Arial', 11, QtGui.QFont.Bold)
        self.font2 = QtGui.QFont('Arial', 11, QtGui.QFont.Normal)
        self.fpath = os.getcwd()
        self.roi_file_id = 0

        grid = QGridLayout()
        gpbox_prep = self.layout_GP_prepare()
        gpbox_msg = self.layout_msg()
        gpbox_xanes = self.layout_xanes()

        grid.addWidget(gpbox_prep, 0, 1)
        grid.addLayout(gpbox_msg, 1, 1)
        grid.addWidget(gpbox_xanes, 2, 1)

        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addWidget(QLabel())
        self.setLayout(layout)



    def default_layout(self):
        try:
            del self.img_xanes, self.img_update, self.xanes_2d_fit, self.xanes_fit_cost # self.img_bkg, self.img_bkg_removed, self.img_bkg_update

        except:
            pass
        default_img = np.zeros([1,500, 500])
        self.save_version = 0
        self.xanes_eng = np.array([0])
        self.img_xanes = deepcopy(default_img)
        self.img_update = deepcopy(default_img)
        self.current_img = deepcopy(default_img)
        self.mask1 = np.array([1])
        self.mask2 = np.array([1])
        self.roi_spec = np.array([0])
        self.msg = ''
        self.shift_list = []
        self.lst_roi.clear()
        self.lb_ang1.setText('No energy data ...')
        # self.lb_ang2.setVisible(False)
        # self.tx_ang.setVisible(False)
        # self.pb_ang.setVisible(False)
        self.num_ref = 0
        self.spectrum_ref = {}
        self.xanes_2d_fit = None
        self.xanes_fit_cost = 0
        self.elem_label = []

        self.pb_save_fit_img.setEnabled(False)
        self.pb_plot_roi.setEnabled(False)
        self.pb_export_roi_fit.setEnabled(False)
        self.pb_colormix.setEnabled(False)
        self.pb_save.setEnabled(False)

        t = self.cb1.findText('Image updated')
        if t:  self.cb1.removeItem(t)
        # t = self.cb1.findText('Aligned image')
        # if t:  self.cb1.removeItem(t)
        t = self.cb1.findText('Raw image')
        if t:  self.cb1.removeItem(t)

    def layout_msg(self):

        self.lb_ip = QLabel()
        self.lb_ip.setFont(self.font2)
        self.lb_ip.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_ip.setText('File loaded:')
        #        self.lb_ip.setFixedWidth(300)
        self.lb_msg = QLabel()
        self.lb_msg.setFont(self.font1)
        self.lb_msg.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_msg.setText('Message:')

        vbox_msg = QVBoxLayout()
        vbox_msg.addWidget(self.lb_ip)
        #        vbox_msg.addWidget(lb_empty1)
        vbox_msg.addWidget(self.lb_msg)
        #        vbox_msg.addWidget(lb_empty)
        vbox_msg.setAlignment(QtCore.Qt.AlignLeft)
        return vbox_msg


    def gpbox_system_info(self):
        lb_empty1 = QLabel()
        lb_empty1.setFixedWidth(80)
        lb_mem = QLabel()
        lb_mem.setFont(self.font1)
        lb_mem.setText('Memory:')
        lb_mem.setFixedWidth(80)

        lb_mem_prog_used = QLabel()
        lb_mem_prog_used.setFont(self.font2)
        lb_mem_prog_used.setText('Prog. used:')
        lb_mem_prog_used.setFixedWidth(80)

        lb_mem_avail = QLabel()
        lb_mem_avail.setFont(self.font2)
        lb_mem_avail.setText('Available:')
        lb_mem_avail.setFixedWidth(80)

        self.lb_mem_avail = QLabel()
        self.lb_mem_avail.setFont(self.font2)
        self.lb_mem_avail.setFixedWidth(60)

        self.lb_mem_prog_used = QLabel()
        self.lb_mem_prog_used.setFont(self.font2)
        self.lb_mem_prog_used.setFixedWidth(100)

        lb_pid = QLabel()
        lb_pid.setFont(self.font1)
        lb_pid.setText('PID:')
        lb_pid.setFixedWidth(80)

        self.lb_pid_display = QLabel()
        self.lb_pid_display.setFont(self.font2)
        self.lb_pid_display.setFixedWidth(80)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(lb_pid)
        hbox1.addWidget(self.lb_pid_display)
        hbox1.setAlignment(QtCore.Qt.AlignLeft)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(lb_mem)
        hbox2.addWidget(lb_mem_prog_used)
        hbox2.addWidget(self.lb_mem_prog_used)

        hbox2.setAlignment(QtCore.Qt.AlignLeft)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(lb_empty1)
        hbox3.addWidget(lb_mem_avail)
        hbox3.addWidget(self.lb_mem_avail)

        hbox3.setAlignment(QtCore.Qt.AlignLeft)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.setAlignment(QtCore.Qt.AlignTop)

        return vbox

    def layout_GP_prepare(self):
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(10)
        lb_empty3 = QLabel()
        lb_empty3.setFixedWidth(100)

        gpbox = QGroupBox('Load image')
        gpbox.setFont(self.font1)

        lb_ld = QLabel()
        lb_ld.setFont(self.font2)
        lb_ld.setText('Image type:')
        lb_ld.setFixedWidth(100)

        self.pb_ld = QPushButton('Load image stack')
        self.pb_ld.setToolTip('image type: .hdf, .tiff')
        self.pb_ld.setFont(self.font2)
        self.pb_ld.clicked.connect(self.load_image)
        self.pb_ld.setFixedWidth(150)

        lb_ang = QLabel()
        lb_ang.setFont(self.font2)
        lb_ang.setText('XANES energy:')
        lb_ang.setFixedWidth(120)

        self.lb_ang1 = QLabel()
        self.lb_ang1.setFont(self.font2)
        self.lb_ang1.setText('No energy data ...')
        self.lb_ang1.setFixedWidth(350)

        self.lb_ang2 = QLabel()
        self.lb_ang2.setFont(self.font2)
        self.lb_ang2.setText('Manual input  (python command):')
        self.lb_ang2.setFixedWidth(250)
        # self.lb_ang2.setVisible(False)

        self.tx_ang = QLineEdit()
        self.tx_ang.setFixedWidth(280)
        self.tx_ang.setFont(self.font2)
        # self.tx_ang.setVisible(False)

        self.pb_ang = QPushButton('Execute')
        self.pb_ang.setFont(self.font2)
        self.pb_ang.clicked.connect(self.manu_energy_input)
        self.pb_ang.setFixedWidth(85)
        # self.pb_ang.setVisible(False)

        lb_mod = QLabel()
        lb_mod.setFont(self.font2)
        lb_mod.setText('Image mode:')
        lb_mod.setFixedWidth(100)

        self.file_group = QButtonGroup()
        self.file_group.setExclusive(True)
        self.rd_hdf = QRadioButton('hdf')
        self.rd_hdf.setFixedWidth(60)
        self.rd_hdf.setChecked(True)
        self.rd_hdf.toggled.connect(self.select_file)

        self.rd_tif = QRadioButton('tif')
        self.rd_tif.setFixedWidth(60)
        self.rd_tif.toggled.connect(self.select_file)
        
        self.rd_db = QRadioButton('Databroker')
        self.rd_db.setFixedWidth(100)
        self.rd_db.toggled.connect(self.select_file)
        self.file_group.addButton(self.rd_hdf)
        self.file_group.addButton(self.rd_tif)
        self.file_group.addButton(self.rd_db)

        lb_hdf_xanes = QLabel()
        lb_hdf_xanes.setFont(self.font2)
        lb_hdf_xanes.setText('Dataset for XANES:')
        lb_hdf_xanes.setFixedWidth(140)

        lb_hdf_eng = QLabel()
        lb_hdf_eng.setFont(self.font2)
        lb_hdf_eng.setText('  energy:')
        lb_hdf_eng.setFixedWidth(60)
        
        lb_db_xanes = QLabel()
        lb_db_xanes.setFont(self.font2)
        lb_db_xanes.setText('Scan id:')
        lb_db_xanes.setFixedWidth(60)
        
        self.tx_db_xanes = QLineEdit()
        self.tx_db_xanes.setText('-1')
        self.tx_db_xanes.setFixedWidth(85)
        self.tx_db_xanes.setFont(self.font2)
        self.tx_db_xanes.setVisible(True)

        self.tx_hdf_xanes = QLineEdit()
        self.tx_hdf_xanes.setText('img_xanes')
        self.tx_hdf_xanes.setFixedWidth(85)
        self.tx_hdf_xanes.setFont(self.font2)
        self.tx_hdf_xanes.setVisible(True)

        self.tx_hdf_eng = QLineEdit()
        self.tx_hdf_eng.setText('X_eng')
        self.tx_hdf_eng.setFixedWidth(85)
        self.tx_hdf_eng.setFont(self.font2)
        self.tx_hdf_eng.setVisible(True)



        self.type_group = QButtonGroup()
        self.type_group.setExclusive(True)
        self.rd_absp = QRadioButton('Absorption')
        self.rd_absp.setFont(self.font2)
        self.rd_absp.setFixedWidth(100)
        self.rd_absp.setChecked(True)
        self.rd_flrc = QRadioButton('Fluorescence')
        self.rd_flrc.setFont(self.font2)
        self.rd_flrc.setFixedWidth(120)
        self.rd_flrc.setChecked(False)
        self.type_group.addButton(self.rd_absp)
        self.type_group.addButton(self.rd_flrc)

        lb_fp = QLabel()
        lb_fp.setFont(self.font2)
        # lb_fp.setText('Image loaded: ')
        lb_fp.setText('')
        lb_fp.setFixedWidth(100)


        gpbox_sys = self.gpbox_system_info()

        hbox1 = QHBoxLayout()
        hbox1.addWidget(lb_ld)
        hbox1.addWidget(self.rd_tif)
        hbox1.addWidget(self.rd_hdf)
        hbox1.addWidget(lb_hdf_xanes)
        hbox1.addWidget(self.tx_hdf_xanes)
        hbox1.addWidget(lb_hdf_eng)
        hbox1.addWidget(self.tx_hdf_eng)
        hbox1.addWidget(lb_empty)
        hbox1.setAlignment(QtCore.Qt.AlignLeft)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(lb_empty3)
        hbox2.addWidget(self.rd_db)
        hbox2.addWidget(lb_db_xanes)
        hbox2.addWidget(self.tx_db_xanes)
        hbox2.addWidget(lb_empty2)
        hbox2.addWidget(self.pb_ld)
        hbox2.addWidget(lb_empty)
        hbox2.setAlignment(QtCore.Qt.AlignLeft)

        hbox_ang = QHBoxLayout()
        hbox_ang.addWidget(lb_ang)
        hbox_ang.addWidget(self.lb_ang1)
        hbox_ang.setAlignment(QtCore.Qt.AlignLeft)

        hbox_manul_input = QHBoxLayout()
        hbox_manul_input.addWidget(self.lb_ang2)
        hbox_manul_input.addWidget(self.tx_ang)
        hbox_manul_input.addWidget(self.pb_ang)
        hbox_manul_input.setAlignment(QtCore.Qt.AlignLeft)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox_ang)
        vbox.addLayout(hbox_manul_input)
        vbox.setAlignment(QtCore.Qt.AlignLeft)


        hbox_tot = QHBoxLayout()
        hbox_tot.addLayout(vbox)
        hbox_tot.addLayout(gpbox_sys)
        hbox_tot.addWidget(lb_empty)
        hbox_tot.setAlignment(QtCore.Qt.AlignLeft)

        gpbox.setLayout(hbox_tot)

        return gpbox

    def layout_xanes(self):
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(5)

        gpbox = QGroupBox('XANES fitting')
        gpbox.setFont(self.font1)

        gpbox_left = QGroupBox('')
        scroll = QScrollArea()


        xanes_prep_layout = self.layout_xanes_prep()
        xanes_roi_layout = self.layout_plot_spec
        xanes_roi_norm_layout = self.layout_roi_normalization()
        xanes_fit2d_layout = self.layout_fit2d()
        canvas_layout = self.layout_canvas()

        vbox_xanes_fit = QVBoxLayout()
        vbox_xanes_fit.addLayout(xanes_prep_layout)
        vbox_xanes_fit.addLayout(xanes_roi_layout)
        vbox_xanes_fit.addLayout(xanes_roi_norm_layout)
        vbox_xanes_fit.addLayout(xanes_fit2d_layout)
        vbox_xanes_fit.addWidget(lb_empty)
        vbox_xanes_fit.setAlignment(QtCore.Qt.AlignTop)

        gpbox_left.setLayout(vbox_xanes_fit)
        scroll.setWidget(gpbox_left)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(600)


        hbox = QHBoxLayout()
        # hbox.addLayout(vbox_xanes_fit)
        hbox.addWidget(scroll)
        hbox.addLayout(canvas_layout)
#        hbox.addWidget(lb_empty2)
        hbox.setAlignment(QtCore.Qt.AlignLeft)
        # return hbox
        gpbox.setLayout(hbox)
        return gpbox

    @property
    def layout_plot_spec(self):
        lb_empty = QLabel()
        lb_roi = QLabel()
        lb_roi.setFont(self.font1)
        lb_roi.setText('ROI for Spec.')
        lb_roi.setFixedWidth(100)

        lb_info = QLabel()
        lb_info.setFont(self.font2)
        lb_info.setStyleSheet('color: rgb(200, 50, 50);')
        lb_info.setText('Spectrum calc. based on current image stack')

        lb_roi_x1 = QLabel()
        lb_roi_x1.setText('Top-left  x:')
        lb_roi_x1.setFont(self.font2)
        lb_roi_x1.setFixedWidth(80)

        lb_roi_y1 = QLabel()
        lb_roi_y1.setText('y:')
        lb_roi_y1.setFont(self.font2)
        lb_roi_y1.setFixedWidth(20)

        lb_roi_x2 = QLabel()
        lb_roi_x2.setText('Bot-right x:')
        lb_roi_x2.setFont(self.font2)
        lb_roi_x2.setFixedWidth(80)

        lb_roi_y2 = QLabel()
        lb_roi_y2.setText('y:')
        lb_roi_y2.setFont(self.font2)
        lb_roi_y2.setFixedWidth(20)

        self.tx_roi_x1 = QLineEdit()
        self.tx_roi_x1.setText('0')
        self.tx_roi_x1.setFont(self.font2)
        self.tx_roi_x1.setFixedWidth(50)

        self.tx_roi_y1 = QLineEdit()
        self.tx_roi_y1.setText('0')
        self.tx_roi_y1.setFont(self.font2)
        self.tx_roi_y1.setFixedWidth(50)

        self.tx_roi_x2 = QLineEdit()
        self.tx_roi_x2.setText('1')
        self.tx_roi_x2.setFont(self.font2)
        self.tx_roi_x2.setFixedWidth(50)

        self.tx_roi_y2 = QLineEdit()
        self.tx_roi_y2.setText('1')
        self.tx_roi_y2.setFont(self.font2)
        self.tx_roi_y2.setFixedWidth(50)

        self.pb_roi_draw = QPushButton('Draw ROI')
        self.pb_roi_draw.setFont(self.font2)
        self.pb_roi_draw.clicked.connect(self.draw_roi)
        self.pb_roi_draw.setFixedWidth(105)
        self.pb_roi_draw.setVisible(True)

        self.pb_roi_plot = QPushButton('Plot Spec.')
        self.pb_roi_plot.setFont(self.font2)
        self.pb_roi_plot.clicked.connect(self.plot_spectrum)
        self.pb_roi_plot.setFixedWidth(105)
        self.pb_roi_plot.setVisible(True)

        self.pb_roi_hide = QPushButton('Hide ROI')
        self.pb_roi_hide.setFont(self.font2)
        self.pb_roi_hide.clicked.connect(self.hide_roi)
        self.pb_roi_hide.setFixedWidth(105)
        self.pb_roi_hide.setVisible(True)

        self.pb_roi_show = QPushButton('Show ROI')
        self.pb_roi_show.setFont(self.font2)
        self.pb_roi_show.clicked.connect(self.show_roi)
        self.pb_roi_show.setFixedWidth(105)
        self.pb_roi_show.setVisible(True)

        self.pb_roi_reset = QPushButton('Reset ROI')
        self.pb_roi_reset.setFont(self.font2)
        self.pb_roi_reset.clicked.connect(self.reset_roi)
        self.pb_roi_reset.setFixedWidth(105)
        self.pb_roi_reset.setVisible(True)

        self.pb_roi_export = QPushButton('Export Spec.')
        self.pb_roi_export.setFont(self.font2)
        self.pb_roi_export.clicked.connect(self.export_spectrum)
        self.pb_roi_export.setFixedWidth(105)
        self.pb_roi_export.setVisible(True)

        lb_file_index = QLabel()
        lb_file_index.setFont(self.font2)
        lb_file_index.setText('  File index for export:')
        lb_file_index.setFixedWidth(155)

        self.tx_file_index = QLineEdit()
        self.tx_file_index.setFixedWidth(50)
        self.tx_file_index.setFont(self.font2)
        self.tx_file_index.setText(str(self.roi_file_id))

        self.lst_roi = QListWidget()
        self.lst_roi.setFont(self.font2)
        self.lst_roi.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_roi.setFixedWidth(80)
        self.lst_roi.setFixedHeight(100)

        lb_lst_roi = QLabel()
        lb_lst_roi.setFont(self.font2)
        lb_lst_roi.setText('ROI list:')
        lb_lst_roi.setFixedWidth(80)

        # hbox_roi_tl = QHBoxLayout()
        # hbox_roi_tl.addWidget(lb_roi_x1)
        # hbox_roi_tl.addWidget(self.tx_roi_x1)
        # hbox_roi_tl.addWidget(lb_roi_y1)
        # hbox_roi_tl.addWidget(self.tx_roi_y1)
        # hbox_roi_tl.setAlignment(QtCore.Qt.AlignLeft)
        #
        # hbox_roi_bd = QHBoxLayout()
        # hbox_roi_bd.addWidget(lb_roi_x2)
        # hbox_roi_bd.addWidget(self.tx_roi_x2)
        # hbox_roi_bd.addWidget(lb_roi_y2)
        # hbox_roi_bd.addWidget(self.tx_roi_y2)
        # hbox_roi_bd.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button1 = QHBoxLayout()
        hbox_roi_button1.addWidget(self.pb_roi_draw)
        hbox_roi_button1.addWidget(self.pb_roi_reset)
        hbox_roi_button1.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button2 = QHBoxLayout()
        hbox_roi_button2.addWidget(self.pb_roi_show)
        hbox_roi_button2.addWidget(self.pb_roi_hide)
        hbox_roi_button2.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button3 = QHBoxLayout()
        hbox_roi_button3.addWidget(self.pb_roi_plot)
        hbox_roi_button3.addWidget(self.pb_roi_export)
        hbox_roi_button3.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button4 = QHBoxLayout()
        hbox_roi_button4.addWidget(lb_file_index)
        hbox_roi_button4.addWidget(self.tx_file_index)
        hbox_roi_button4.setAlignment(QtCore.Qt.AlignLeft)

        vbox_roi = QVBoxLayout()
        vbox_roi.setContentsMargins(0, 0, 0, 0)
        vbox_roi.addLayout(hbox_roi_button1)
        vbox_roi.addLayout(hbox_roi_button2)
        vbox_roi.addLayout(hbox_roi_button3)
        vbox_roi.addLayout(hbox_roi_button4)
        vbox_roi.setAlignment(QtCore.Qt.AlignLeft)

        vbox_lst = QVBoxLayout()
        vbox_lst.addWidget(lb_lst_roi, 0, QtCore.Qt.AlignTop)
        vbox_lst.addWidget(self.lst_roi, 0, QtCore.Qt.AlignTop)
        vbox_lst.addWidget(lb_empty)
        vbox_lst.setAlignment(QtCore.Qt.AlignLeft)

        box_roi = QHBoxLayout()
        box_roi.addLayout(vbox_roi)
        box_roi.addLayout(vbox_lst)
        box_roi.addWidget(lb_empty, 0, QtCore.Qt.AlignLeft)
        box_roi.setAlignment(QtCore.Qt.AlignLeft)
        # box_roi.setAlignment(QtCore.Qt.AlignTop)

        box_roi_tot = QVBoxLayout()
        box_roi_tot.addWidget(lb_roi)
        box_roi_tot.addWidget(lb_info)
        box_roi_tot.addLayout(box_roi)
        # box_roi_tot.addWidget(lb_empty)
        box_roi_tot.setAlignment(QtCore.Qt.AlignLeft)

        return box_roi_tot

    def layout_roi_normalization(self):
        lb_empty = QLabel()
        lb_fit_edge = QLabel()
        lb_fit_edge.setFont(self.font1)
        lb_fit_edge.setText('ROI normalization')
        lb_fit_edge.setFixedWidth(150)

        lb_fit_pre_s = QLabel()
        lb_fit_pre_s.setText('Pre -edge start:')
        lb_fit_pre_s.setFont(self.font2)
        lb_fit_pre_s.setFixedWidth(120)

        lb_fit_pre_e = QLabel()
        lb_fit_pre_e.setText('end:')
        lb_fit_pre_e.setFont(self.font2)
        lb_fit_pre_e.setFixedWidth(40)

        lb_fit_post_s = QLabel()
        lb_fit_post_s.setText('Post-edge start:')
        lb_fit_post_s.setFont(self.font2)
        lb_fit_post_s.setFixedWidth(120)

        lb_fit_post_e = QLabel()
        lb_fit_post_e.setText('end:')
        lb_fit_post_e.setFont(self.font2)
        lb_fit_post_e.setFixedWidth(40)

        self.tx_fit_pre_s = QLineEdit()
        self.tx_fit_pre_s.setFont(self.font2)
        self.tx_fit_pre_s.setFixedWidth(60)

        self.tx_fit_pre_e = QLineEdit()
        self.tx_fit_pre_e.setFont(self.font2)
        self.tx_fit_pre_e.setFixedWidth(60)

        self.tx_fit_post_s = QLineEdit()
        self.tx_fit_post_s.setFont(self.font2)
        self.tx_fit_post_s.setFixedWidth(60)

        self.tx_fit_post_e = QLineEdit()
        self.tx_fit_post_e.setFont(self.font2)
        self.tx_fit_post_e.setFixedWidth(60)

        lb_fit_roi = QLabel()
        lb_fit_roi.setFont(self.font2)
        lb_fit_roi.setText('Norm Spec(ROI)')
        lb_fit_roi.setFixedWidth(120)

        self.pb_fit_roi = QPushButton('Norm Spec')
        self.pb_fit_roi.setFont(self.font2)
        self.pb_fit_roi.clicked.connect(self.fit_edge)
        self.pb_fit_roi.setFixedWidth(90)
        self.pb_fit_roi.setVisible(True)

        self.pb_save_fit_roi = QPushButton('Save Spec')
        self.pb_save_fit_roi.setFont(self.font2)
        self.pb_save_fit_roi.clicked.connect(self.save_normed_roi)
        self.pb_save_fit_roi.setFixedWidth(90)
        self.pb_save_fit_roi.setVisible(True)

        lb_fit_img = QLabel()
        lb_fit_img.setFont(self.font2)
        lb_fit_img.setText('Norm image')
        lb_fit_img.setFixedWidth(120)

        self.pb_fit_img = QPushButton('Norm Img')
        self.pb_fit_img.setFont(self.font2)
        self.pb_fit_img.clicked.connect(self.fit_edge_img)
        self.pb_fit_img.setFixedWidth(90)
        self.pb_fit_img.setVisible(True)

        self.pb_save_fit_img = QPushButton('Save Img')
        self.pb_save_fit_img.setFont(self.font2)
        self.pb_save_fit_img.clicked.connect(self.save_normed_img)
        self.pb_save_fit_img.setFixedWidth(90)
        self.pb_save_fit_img.setEnabled(False)


        hbox_fit_pre = QHBoxLayout()
        hbox_fit_pre.addWidget(lb_fit_pre_s)
        hbox_fit_pre.addWidget(self.tx_fit_pre_s)
        hbox_fit_pre.addWidget(lb_fit_pre_e)
        hbox_fit_pre.addWidget(self.tx_fit_pre_e)
        hbox_fit_pre.setAlignment(QtCore.Qt.AlignLeft)

        hbox_fit_post = QHBoxLayout()
        hbox_fit_post.addWidget(lb_fit_post_s)
        hbox_fit_post.addWidget(self.tx_fit_post_s)
        hbox_fit_post.addWidget(lb_fit_post_e)
        hbox_fit_post.addWidget(self.tx_fit_post_e)
        hbox_fit_post.setAlignment(QtCore.Qt.AlignLeft)

        hbox_fit_pb = QHBoxLayout()
        hbox_fit_pb.addWidget(lb_fit_roi)
        hbox_fit_pb.addWidget(self.pb_fit_roi)
        hbox_fit_pb.addWidget(self.pb_save_fit_roi)
        hbox_fit_pb.setAlignment(QtCore.Qt.AlignLeft)

        hbox_fit_pb_img = QHBoxLayout()
        hbox_fit_pb_img.addWidget(lb_fit_img)
        hbox_fit_pb_img.addWidget(self.pb_fit_img)
        hbox_fit_pb_img.addWidget(self.pb_save_fit_img)
        hbox_fit_pb_img.setAlignment(QtCore.Qt.AlignLeft)

        vbox_fit = QVBoxLayout()
        vbox_fit.addWidget(lb_fit_edge)
        vbox_fit.addLayout(hbox_fit_pre)
        vbox_fit.addLayout(hbox_fit_post)
        vbox_fit.addLayout(hbox_fit_pb)
        vbox_fit.addLayout(hbox_fit_pb_img)
        vbox_fit.addWidget(lb_empty)
        vbox_fit.setAlignment(QtCore.Qt.AlignLeft)

        return vbox_fit
    
    
    
    def layout_fit2d(self):
        lb_empty = QLabel()
        lb_fit2d = QLabel()
        lb_fit2d.setFont(self.font1)
        lb_fit2d.setText('Fit 2D XANES')
        lb_fit2d.setFixedWidth(150)

        self.lb_ref_info = QLabel()
        self.lb_ref_info.setFont(self.font2)
        self.lb_ref_info.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_ref_info.setText('Reference spectrum: ')
        self.lb_ref_info.setFixedWidth(300)
        
        self.pb_ld_ref = QPushButton('Load Ref.')
        self.pb_ld_ref.setFont(self.font2)
        self.pb_ld_ref.clicked.connect(self.load_xanes_ref)
        self.pb_ld_ref.setEnabled(True)
        self.pb_ld_ref.setFixedWidth(105)

        self.pb_plt_ref = QPushButton('Plot Ref.')
        self.pb_plt_ref.setFont(self.font2)
        self.pb_plt_ref.clicked.connect(self.plot_xanes_ref)
        self.pb_plt_ref.setEnabled(True)
        self.pb_plt_ref.setFixedWidth(105)

        lb_elem = QLabel()
        lb_elem.setFont(self.font2)
        lb_elem.setText(' Elem.: ')
        lb_elem.setFixedWidth(40)

        self.tx_elem = QLineEdit(self)
        self.tx_elem.setFont(self.font2)
        self.tx_elem.setFixedWidth(60)

        hbox_ref = QHBoxLayout()
        hbox_ref.addWidget(self.pb_ld_ref)
        hbox_ref.addWidget(lb_elem)
        hbox_ref.addWidget(self.tx_elem)
        hbox_ref.addWidget(self.pb_plt_ref)
        hbox_ref.setAlignment(QtCore.Qt.AlignTop)

        #######################
        self.pb_fit2d = QPushButton('Fit 2D')
        self.pb_fit2d.setFont(self.font2)
        self.pb_fit2d.clicked.connect(self.fit_2d_xanes)
        self.pb_fit2d.setEnabled(True)
        self.pb_fit2d.setFixedWidth(105)

        self.pb_reset_ref = QPushButton('Reset Ref.')
        self.pb_reset_ref.setFont(self.font2)
        self.pb_reset_ref.clicked.connect(self.reset_xanes_ref)
        self.pb_reset_ref.setEnabled(True)
        self.pb_reset_ref.setFixedWidth(105)

        self.pb_reset_fit = QPushButton('Reset All')
        self.pb_reset_fit.setFont(self.font2)
        self.pb_reset_fit.clicked.connect(self.reset_xanes_fit)
        self.pb_reset_fit.setEnabled(True)
        self.pb_reset_fit.setFixedWidth(105)

        # lb_fit_roi = QLabel()
        # lb_fit_roi.setFont(self.font2)
        # lb_fit_roi.setText('ROI index: ')
        # lb_fit_roi.setFixedWidth(75)
        #
        # self.tx_fit_roi = QLineEdit(self)
        # self.tx_fit_roi.setFont(self.font2)
        # self.tx_fit_roi.setText('-1')
        # self.tx_fit_roi.setValidator(QIntValidator())
        # self.tx_fit_roi.setFixedWidth(25)

        hbox_fit2d = QHBoxLayout()
        hbox_fit2d.addWidget(self.pb_fit2d)
        hbox_fit2d.addWidget(self.pb_reset_ref)
        hbox_fit2d.addWidget(self.pb_reset_fit)
        # hbox_fit2d.addWidget(lb_fit_roi)
        # hbox_fit2d.addWidget(self.tx_fit_roi)
        hbox_fit2d.setAlignment(QtCore.Qt.AlignTop)
        ######################################
        
        self.pb_fit2d_iter = QPushButton('Fit 2D (iter)')
        self.pb_fit2d_iter.setFont(self.font2)
        self.pb_fit2d_iter.clicked.connect(self.fit_2d_xanes_iter)
        self.pb_fit2d_iter.setEnabled(True)
        self.pb_fit2d_iter.setFixedWidth(105)

        lb_iter_rate = QLabel()
        lb_iter_rate.setFont(self.font2)
        lb_iter_rate.setText(' Rate:')
        lb_iter_rate.setFixedWidth(50)

        lb_iter_num = QLabel()
        lb_iter_num.setFont(self.font2)
        lb_iter_num.setText(' #iter.')
        lb_iter_num.setFixedWidth(40)

        self.tx_iter_rate = QLineEdit(self)
        self.tx_iter_rate.setFont(self.font2)
        self.tx_iter_rate.setText('0.005')
        self.tx_iter_rate.setFixedWidth(50)

        self.tx_iter_num = QLineEdit(self)
        self.tx_iter_num.setFont(self.font2)
        self.tx_iter_num.setText('5')
        self.tx_iter_num.setValidator(QIntValidator())
        self.tx_iter_num.setFixedWidth(60)

        
        hbox_iter = QHBoxLayout()
        hbox_iter.addWidget(self.pb_fit2d_iter)
        hbox_iter.addWidget(lb_iter_rate)
        hbox_iter.addWidget(self.tx_iter_rate)
        hbox_iter.addWidget(lb_iter_num)
        hbox_iter.addWidget(self.tx_iter_num)
        hbox_iter.setAlignment(QtCore.Qt.AlignTop)
        ##########################
        
        #############################

        self.pb_mask1 = QPushButton('Gen. Mask1')
        self.pb_mask1.setFont(self.font2)
        self.pb_mask1.clicked.connect(self.generate_mask1)
        self.pb_mask1.setEnabled(True)
        self.pb_mask1.setFixedWidth(105)

        self.tx_mask1 = QLineEdit(self)
        self.tx_mask1.setFont(self.font2)
        self.tx_mask1.setText('>0.1')
        self.tx_mask1.setFixedWidth(50)

        lb_mask1 = QLabel()
        lb_mask1.setFont(self.font2)
        lb_mask1.setText('Thresh: ')
        lb_mask1.setFixedWidth(50)

        self.pb_mask1_rm = QPushButton('Remov Mask1')
        self.pb_mask1_rm.setFont(self.font2)
        self.pb_mask1_rm.clicked.connect(self.rm_mask1)
        self.pb_mask1_rm.setEnabled(True)
        self.pb_mask1_rm.setFixedWidth(105)



        self.pb_mask2 = QPushButton('Gen. Mask2')
        self.pb_mask2.setFont(self.font2)
        self.pb_mask2.clicked.connect(self.generate_mask2)
        self.pb_mask2.setEnabled(True)
        self.pb_mask2.setFixedWidth(105)

        self.tx_mask2 = QLineEdit(self)
        self.tx_mask2.setFont(self.font2)
        self.tx_mask2.setText('>0.5')
        self.tx_mask2.setFixedWidth(50)

        lb_mask2 = QLabel()
        lb_mask2.setFont(self.font2)
        lb_mask2.setText('Thresh: ')
        lb_mask2.setFixedWidth(50)

        self.pb_mask2_rm = QPushButton('Remov Mask2')
        self.pb_mask2_rm.setFont(self.font2)
        self.pb_mask2_rm.clicked.connect(self.rm_mask2)
        self.pb_mask2_rm.setEnabled(True)
        self.pb_mask2_rm.setFixedWidth(105)


        hbox_mask1 = QHBoxLayout()
        hbox_mask1.addWidget(self.pb_mask1)
        hbox_mask1.addWidget(lb_mask1)
        hbox_mask1.addWidget(self.tx_mask1)
        hbox_mask1.addWidget(self.pb_mask1_rm)
        hbox_mask1.setAlignment(QtCore.Qt.AlignLeft)


        hbox_mask2 = QHBoxLayout()
        hbox_mask2.addWidget(self.pb_mask2)
        hbox_mask2.addWidget(lb_mask2)
        hbox_mask2.addWidget(self.tx_mask2)
        hbox_mask2.addWidget(self.pb_mask2_rm)
        hbox_mask2.setAlignment(QtCore.Qt.AlignTop)

        ########################

        ##########################
        self.pb_plot_roi = QPushButton('Plot ROI fit.')
        self.pb_plot_roi.setFont(self.font2)
        self.pb_plot_roi.clicked.connect(lambda return_flag: self.plot_roi_fit(1))
        self.pb_plot_roi.setEnabled(False)
        self.pb_plot_roi.setFixedWidth(105)

        lb_fit_roi = QLabel()
        lb_fit_roi.setFont(self.font2)
        lb_fit_roi.setText(' ROI #: ')
        lb_fit_roi.setFixedWidth(50)


        self.tx_fit_roi = QLineEdit(self)
        self.tx_fit_roi.setFont(self.font2)
        self.tx_fit_roi.setText('-1')
        self.tx_fit_roi.setValidator(QIntValidator())
        self.tx_fit_roi.setFixedWidth(50)
        
        self.pb_export_roi_fit = QPushButton('Export ROI fit')
        self.pb_export_roi_fit.setFont(self.font2)
        self.pb_export_roi_fit.clicked.connect(self.export_roi_fit)
        self.pb_export_roi_fit.setEnabled(False)
        self.pb_export_roi_fit.setFixedWidth(105)

        hbox_plot = QHBoxLayout()
        hbox_plot.addWidget(self.pb_plot_roi)
        hbox_plot.addWidget(lb_fit_roi)
        hbox_plot.addWidget(self.tx_fit_roi)
        hbox_plot.addWidget(self.pb_export_roi_fit)
        hbox_plot.setAlignment(QtCore.Qt.AlignTop)

        ############################################

        self.pb_colormix = QPushButton('Color mix')
        self.pb_colormix.setFont(self.font2)
        self.pb_colormix.clicked.connect(self.xanes_colormix)
        self.pb_colormix.setEnabled(False)
        self.pb_colormix.setFixedWidth(105)

        lb_fit_color = QLabel()
        lb_fit_color.setFont(self.font2)
        lb_fit_color.setText(' Color: ')
        lb_fit_color.setFixedWidth(50)

        self.tx_fit_color = QLineEdit(self)
        self.tx_fit_color.setFont(self.font2)
        self.tx_fit_color.setText('r, g, b')
        self.tx_fit_color.setFixedWidth(50)

        self.pb_save = QPushButton('Save 2DFit')
        self.pb_save.setFont(self.font2)
        self.pb_save.clicked.connect(self.save_2Dfit)
        self.pb_save.setEnabled(False)
        self.pb_save.setFixedWidth(105)

        hbox_save = QHBoxLayout()
        hbox_save.addWidget(self.pb_colormix)
        hbox_save.addWidget(lb_fit_color)
        hbox_save.addWidget(self.tx_fit_color)
        hbox_save.addWidget(self.pb_save)
        hbox_save.setAlignment(QtCore.Qt.AlignTop)

        ##########################
        
        vbox_assemble = QVBoxLayout()
        vbox_assemble.addWidget(lb_fit2d)
        vbox_assemble.addWidget(self.lb_ref_info)
        # vbox_assemble.addWidget(self.pb_ld_ref)
        vbox_assemble.addLayout(hbox_ref)
        vbox_assemble.addLayout(hbox_fit2d)
        vbox_assemble.addLayout(hbox_iter)
        vbox_assemble.addLayout(hbox_mask1)
        vbox_assemble.addLayout(hbox_mask2)
        vbox_assemble.addLayout(hbox_plot)
        vbox_assemble.addLayout(hbox_save)
        vbox_assemble.setAlignment(QtCore.Qt.AlignLeft)
        
        
        ##################################
        # self.lst_ref = QListWidget()
        # self.lst_ref.setFont(self.font2)
        # self.lst_ref.setSelectionMode(QAbstractItemView.MultiSelection)
        # self.lst_ref.setFixedWidth(80)
        # self.lst_ref.setFixedHeight(100)
        #
        # lb_lst_ref = QLabel()
        # lb_lst_ref.setFont(self.font2)
        # lb_lst_ref.setText('Ref. list:')
        # lb_lst_ref.setFixedWidth(80)
        #
        # vbox_ref = QVBoxLayout()
        # vbox_ref.addWidget(lb_lst_ref)
        # vbox_ref.addWidget(self.lst_ref)
        # vbox_ref.setAlignment(QtCore.Qt.AlignLeft)
        ##################################
        
        # hbox_assemble = QHBoxLayout()
        # hbox_assemble.addLayout(vbox_assemble)
        # hbox_assemble.addLayout(vbox_ref)
        # hbox_assemble.setAlignment(QtCore.Qt.AlignTop)
        
        vbox_tot = QVBoxLayout()
        # vbox_tot.addWidget(lb_fit2d)
        # vbox_tot.addLayout(hbox_assemble)
        # vbox_tot.setAlignment(QtCore.Qt.AlignLeft)
        return vbox_assemble
        pass

    def layout_xanes_prep(self):
        lb_empty = QLabel()
        lb_prep = QLabel()
        lb_prep.setFont(self.font1)
        lb_prep.setText('Preparation')
        lb_prep.setFixedWidth(150)

        self.pb_norm_txm = QPushButton('Norm. TMX (-log)')
        self.pb_norm_txm.setFont(self.font2)
        self.pb_norm_txm.clicked.connect(self.norm_txm)
        self.pb_norm_txm.setEnabled(False)
        self.pb_norm_txm.setFixedWidth(150)

        self.pb_align = QPushButton('Align Img')
        self.pb_align.setFont(self.font2)
        self.pb_align.clicked.connect(self.xanes_align_img)
        self.pb_align.setEnabled(False)
        self.pb_align.setFixedWidth(150)

        self.pb_align_roi = QPushButton('Align Img (ROI)')
        self.pb_align_roi.setFont(self.font2)
        self.pb_align_roi.clicked.connect(self.xanes_align_img_roi)
        self.pb_align_roi.setEnabled(False)
        self.pb_align_roi.setFixedWidth(150)

        self.pb_rmbg = QPushButton('Remove Bkg (opt.)')
        self.pb_rmbg.setFont(self.font2)
        self.pb_rmbg.clicked.connect(self.remove_bkg)
        self.pb_rmbg.setEnabled(False)
        self.pb_rmbg.setFixedWidth(150)

        self.pb_apply_shft = QPushButton('Apply shift')
        self.pb_apply_shft.setFont(self.font2)
        self.pb_apply_shft.clicked.connect(self.apply_shift)
        self.pb_apply_shft.setFixedWidth(150)

        self.pb_save_shft = QPushButton('Save shift list')
        self.pb_save_shft.setFont(self.font2)
        self.pb_save_shft.clicked.connect(self.save_shift)
        self.pb_save_shft.setFixedWidth(150)

        self.pb_load_shft = QPushButton('Load shift list')
        self.pb_load_shft.setFont(self.font2)
        self.pb_load_shft.clicked.connect(self.load_shift)
        self.pb_load_shft.setFixedWidth(150)

        self.lb_shift = QLabel()
        self.lb_shift.setFont(self.font2)
        self.lb_shift.setText('  Shift list: None')
        self.lb_shift.setFixedWidth(150)

        lb_ali_ref = QLabel()
        lb_ali_ref.setFont(self.font2)
        lb_ali_ref.setText('  Ref. image: ')
        lb_ali_ref.setFixedWidth(80)

        lb_ali_roi = QLabel()
        lb_ali_roi.setFont(self.font2)
        lb_ali_roi.setText('  ROI index: ')
        lb_ali_roi.setFixedWidth(80)

        self.tx_ali_ref = QLineEdit(self)
        self.tx_ali_ref.setFont(self.font2)
        self.tx_ali_ref.setText('-1')
        self.tx_ali_ref.setValidator(QIntValidator())
        self.tx_ali_ref.setFixedWidth(50)

        self.tx_ali_roi = QLineEdit(self)
        self.tx_ali_roi.setFont(self.font2)
        self.tx_ali_roi.setText('-1')
        self.tx_ali_roi.setValidator(QIntValidator())
        self.tx_ali_roi.setFixedWidth(50)


        self.pb_filt = QPushButton('Median filter')
        self.pb_filt.setFont(self.font2)
        self.pb_filt.clicked.connect(self.xanes_img_smooth)
        self.pb_filt.setEnabled(False)
        self.pb_filt.setFixedWidth(150)

        lb_filt = QLabel()
        lb_filt.setFont(self.font2)
        lb_filt.setText('  kernal size: ')
        lb_filt.setFixedWidth(80)

        self.tx_filt = QLineEdit(self)
        self.tx_filt.setFont(self.font2)
        self.tx_filt.setText('3')
        self.tx_filt.setValidator(QIntValidator())
        self.tx_filt.setFixedWidth(50)

        hbox_filt = QHBoxLayout()
        hbox_filt.addWidget(self.pb_filt)
        hbox_filt.addWidget(lb_filt)
        hbox_filt.addWidget(self.tx_filt)
        hbox_filt.setAlignment(QtCore.Qt.AlignLeft)

        hbox_prep = QHBoxLayout()
        hbox_prep.addWidget(self.pb_norm_txm)
        hbox_prep.addWidget(self.pb_rmbg)
        hbox_prep.setAlignment(QtCore.Qt.AlignLeft)

        hbox_ali = QHBoxLayout()
        hbox_ali.addWidget(self.pb_align)
        # hbox_norm.addWidget(self.lb_ali)
        hbox_ali.addWidget(lb_ali_ref)
        hbox_ali.addWidget(self.tx_ali_ref)
        hbox_ali.setAlignment(QtCore.Qt.AlignLeft)

        hbox_ali_roi = QHBoxLayout()
        hbox_ali_roi.addWidget(self.pb_align_roi)
        hbox_ali_roi.addWidget(lb_ali_roi)
        hbox_ali_roi.addWidget(self.tx_ali_roi)
        hbox_ali_roi.setAlignment(QtCore.Qt.AlignLeft)

        hbox_shft = QHBoxLayout()
        hbox_shft.addWidget(self.pb_save_shft)
        hbox_shft.addWidget(self.pb_load_shft)
        hbox_shft.setAlignment(QtCore.Qt.AlignLeft)

        hbox_shft1 = QHBoxLayout()
        hbox_shft1.addWidget(self.pb_apply_shft)
        hbox_shft1.addWidget(self.lb_shift)
        hbox_shft1.setAlignment(QtCore.Qt.AlignLeft)

        vbox_prep = QVBoxLayout()
        vbox_prep.addWidget(lb_prep)
        vbox_prep.addLayout(hbox_prep)
        vbox_prep.addLayout(hbox_filt)
        vbox_prep.addLayout(hbox_ali)
        vbox_prep.addLayout(hbox_ali_roi)
        vbox_prep.addLayout(hbox_shft)
        vbox_prep.addLayout(hbox_shft1)
        vbox_prep.addWidget(lb_empty)

        return vbox_prep

    def layout_canvas(self):
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(10)

        self.canvas1 = MyCanvas(obj=self)

        self.sl1 = QScrollBar(QtCore.Qt.Horizontal)
        self.sl1.setMaximum(0)
        self.sl1.setMinimum(0)
        self.sl1.valueChanged.connect(self.sliderval)

        self.cb1 = QComboBox()
        self.cb1.setFont(self.font2)
        self.cb1.addItem('Raw image')
        # self.cb1.addItem('Background')
        self.cb1.currentIndexChanged.connect(self.update_canvas_img)

        self.pb_del = QPushButton('Del. single image')
        self.pb_del.setToolTip(
            'Delete single image, it will delete the same slice in other images (e.g., "raw image", "aligned image", "background removed" ')
        self.pb_del.setFont(self.font2)
        self.pb_del.clicked.connect(self.delete_single_img)
        self.pb_del.setEnabled(False)
        self.pb_del.setFixedWidth(150)

        hbox_can_l = QHBoxLayout()
        hbox_can_l.addWidget(self.cb1)
        hbox_can_l.addWidget(self.pb_del)

        self.lb_x_l = QLabel()
        self.lb_x_l.setFont(self.font2)
        self.lb_x_l.setText('x: ')
        self.lb_x_l.setFixedWidth(80)

        self.lb_y_l = QLabel()
        self.lb_y_l.setFont(self.font2)
        self.lb_y_l.setText('y: ')
        self.lb_y_l.setFixedWidth(80)

        self.lb_z_l = QLabel()
        self.lb_z_l.setFont(self.font2)
        self.lb_z_l.setText('intensity: ')
        self.lb_z_l.setFixedWidth(120)

        lb_cmap = QLabel()
        lb_cmap.setFont(self.font2)
        lb_cmap.setText('colormap: ')
        lb_cmap.setFixedWidth(80)

        cmap = ['gray', 'bone', 'viridis', 'terrain', 'gnuplot', 'bwr', 'plasma', 'PuBu', 'summer', 'rainbow', 'jet']
        self.cb_cmap = QComboBox()
        self.cb_cmap.setFont(self.font2)
        for i in cmap:
            self.cb_cmap.addItem(i)
        self.cb_cmap.setCurrentText('terrain')
        self.cb_cmap.currentIndexChanged.connect(self.change_colormap)
        self.cb_cmap.setFixedWidth(80)

        self.pb_adj_cmap = QPushButton('Auto Contrast')
        self.pb_adj_cmap.setFont(self.font2)
        self.pb_adj_cmap.clicked.connect(self.auto_contrast)
        self.pb_adj_cmap.setEnabled(True)
        self.pb_adj_cmap.setFixedWidth(120)

        lb_cmax = QLabel()
        lb_cmax.setFont(self.font2)
        lb_cmax.setText('cmax: ')
        lb_cmax.setFixedWidth(40)
        lb_cmin = QLabel()
        lb_cmin.setFont(self.font2)
        lb_cmin.setText('cmin: ')
        lb_cmin.setFixedWidth(40)

        self.tx_cmax = QLineEdit(self)
        self.tx_cmax.setFont(self.font2)
        self.tx_cmax.setFixedWidth(60)
        self.tx_cmax.setText('1.')
        self.tx_cmax.setValidator(QDoubleValidator())
        self.tx_cmax.setEnabled(True)

        self.tx_cmin = QLineEdit(self)
        self.tx_cmin.setFont(self.font2)
        self.tx_cmin.setFixedWidth(60)
        self.tx_cmin.setText('0.')
        self.tx_cmin.setValidator(QDoubleValidator())
        self.tx_cmin.setEnabled(True)

        self.pb_set_cmap = QPushButton('Set')
        self.pb_set_cmap.setFont(self.font2)
        self.pb_set_cmap.clicked.connect(self.set_contrast)
        self.pb_set_cmap.setEnabled(True)
        self.pb_set_cmap.setFixedWidth(60)

        hbox_chbx_l = QHBoxLayout()
        hbox_chbx_l.addWidget(self.lb_x_l)
        hbox_chbx_l.addWidget(self.lb_y_l)
        hbox_chbx_l.addWidget(self.lb_z_l)
        hbox_chbx_l.addWidget(lb_empty)
        hbox_chbx_l.setAlignment(QtCore.Qt.AlignLeft)

        hbox_cmap = QHBoxLayout()
        hbox_cmap.addWidget(lb_cmap)
        hbox_cmap.addWidget(self.cb_cmap)
        hbox_cmap.addWidget(self.pb_adj_cmap)
        hbox_cmap.addWidget(lb_cmin)
        hbox_cmap.addWidget(self.tx_cmin)
        hbox_cmap.addWidget(lb_cmax)
        hbox_cmap.addWidget(self.tx_cmax)
        hbox_cmap.addWidget(self.pb_set_cmap)
        hbox_chbx_l.addWidget(lb_empty)
        hbox_cmap.setAlignment(QtCore.Qt.AlignLeft)

        vbox_can1 = QVBoxLayout()
        vbox_can1.addWidget(self.canvas1)
        vbox_can1.addWidget(self.sl1)
        vbox_can1.addLayout(hbox_can_l)
        vbox_can1.addLayout(hbox_chbx_l)
        vbox_can1.addLayout(hbox_cmap)
        vbox_can1.setAlignment(QtCore.Qt.AlignLeft)

        return vbox_can1


    def check_xanes_fit_requirement(self, img_stack):
        n_ref = len(self.spectrum_ref)
        return_flag = 1
        if n_ref < 2:
            self.msg += ';   # of reference spectrum need to be larger than 2, fitting fails ...'
            return_flag = 0
        elif img_stack.shape[0] < n_ref:
            self.msg += ';   # of stack image is less then # of reference spectrum, fitting fails ...'
            return_flag = 0
        return return_flag



    def load_xanes_ref(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = 'txt files (*.txt)'
        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
        if fn:
            try:
                print(fn)
                fn_ref = fn.split('/')[-1]
                print(f'selected reference: {fn_ref}')
                self.lb_ref_info.setText(self.lb_ref_info.text() + '\n' + f'ref #{self.num_ref}: ' + fn_ref)
                self.lb_ref_info.setStyleSheet('color: rgb(200, 50, 50);')
                QApplication.processEvents()
                # self.spectrum_ref[f'ref{self.num_ref}'] = np.array(pd.read_csv(fn, sep=' '))
                self.spectrum_ref[f'ref{self.num_ref}'] = np.loadtxt(fn)
                self.num_ref += 1
            except:
                print('un-supported xanes reference format')
                

    def reset_xanes_ref(self):
        self.num_ref = 0
        self.lb_ref_info.setText('Reference spectrum:')
        self.spectrum_ref = {}
        self.xanes_fit_cost = 0
        self.tx_elem.setText('')
        self.elem_label = []





    def fit_2d_xanes(self):
        self.pb_fit2d.setDisabled(True)
        QApplication.processEvents()
        canvas = self.canvas1

        img_stack = canvas.img_stack
        try:
            img_stack = self.img_update
            self.msg = 'Fit 2D xanes: using "Image Update"'
        except:
            img_stack = self.img_xanes
            self.msg = 'Fit 2D xanes: using "Raw image"'

        return_flag = self.check_xanes_fit_requirement(img_stack)
        self.update_msg()

        if return_flag:
            try:
                self.xanes_2d_fit, self.xanes_fit_cost = fit_2D_xanes_non_iter(img_stack, self.xanes_eng, self.spectrum_ref)
                if self.cb1.findText('XANES Fit') < 0:
                    self.cb1.addItem('XANES Fit')
                if self.cb1.findText('XANES Fit error') < 0:
                    self.cb1.addItem('XANES Fit error')
                self.cb1.setCurrentText('XANES Fit')
                self.update_canvas_img()
                self.msg = '2D fitting finished. "XANES Fit" has been added for imshow'
                self.pb_plot_roi.setEnabled(True)
                self.pb_export_roi_fit.setEnabled(True)
                self.pb_colormix.setEnabled(True)
                self.pb_save.setEnabled(True)
            except:
                print('fitting fails ...')
                self.msg = 'fitting fails ...'

        self.update_msg()
        self.pb_fit2d.setEnabled(True)
        QApplication.processEvents()



    def reset_xanes_fit(self):
        self.reset_xanes_ref()
        self.xanes_2d_fit = None
        self.pb_plot_roi.setDisabled(True)



    def fit_2d_xanes_iter(self):
        self.pb_fit2d_iter.setEnabled(False)
        QApplication.processEvents()
        canvas = self.canvas1
        img_stack = self.img_update
        if img_stack.shape[0] < 4:
            img_stack = self.img_xanes
            self.msg = 'Fit 2D xanes: using "Raw image"'
        else:
            self.msg = 'Fit 2D xanes: using "Image Update"'
        return_flag = self.check_xanes_fit_requirement(img_stack)
        self.update_msg()
        if return_flag:
            try:
                learning_rate = float(self.tx_iter_rate.text())
                num_iter = int(self.tx_iter_num.text())
                coef0 = self.xanes_2d_fit

                if coef0 is None:
                    self.msg = 'Using random initial guess. It may take few minutes ...'
                else:
                    self.msg = 'Using existing fitting as initial guess'
                self. update_msg()

                self.pb_fit2d_iter.setEnabled(False)
                self.xanes_2d_fit, self.xanes_fit_cost = fit_2D_xanes_iter(img_stack, self.xanes_eng, self.spectrum_ref, coef0, learning_rate, num_iter)
                self.pb_fit2d_iter.setEnabled(True)
                QApplication.processEvents()

                if self.cb1.findText('XANES Fit') < 0:
                    self.cb1.addItem('XANES Fit')
                if self.cb1.findText('XANES Fit error') < 0:
                    self.cb1.addItem('XANES Fit error')
                self.cb1.setCurrentText('XANES Fit')
                self.update_canvas_img()
                self.msg = 'Iterative fitting finished'

            except:
                print('iterative fitting fails ...')
                self.msg = 'iterative fitting fails ...'
            finally:
                self.pb_plot_roi.setEnabled(True)
                self.update_msg()
        elf.pb_fit2d_iter.setEnabled(True)



    def plot_xanes_ref(self):
        plt.figure()
        legend = []
        elem = self.tx_elem.text()
        elem = elem.replace(' ','')
        elem = elem.replace(';', ',')
        elem = elem.split(',')
        if elem[0] == '':
            elem = []
        try:
            for i in range(self.num_ref):
                try:
                    plot_label = elem[i]
                except:
                    plot_label = f'ref_{i}'
                self.elem_label.append(plot_label)
                spec = self.spectrum_ref[f'ref{i}']
                line, = plt.plot(spec[:,0], spec[:,1], label=plot_label)
                legend.append(line)
            print(legend)
            plt.legend(handles=legend)
            plt.show()
        except:
            self.msg = 'un-recognized reference spectrum format'
            self.update_msg()

    def generate_mask1(self):
        try:
            tmp = np.squeeze(self.canvas1.current_img)
            mask = np.ones(tmp.shape)

            tmp1 = self.tx_mask1.text()
            if tmp1[0] == '<':
                thresh = float(tmp1[1:])
                mask[tmp < thresh] = 0
            elif tmp1[0] == '>':
                thresh = float(tmp1[1:])
                mask[tmp > thresh] = 0
            else:
                thresh = float(tmp1)
                mask[tmp > thresh] = 0
            self.canvas1.mask = self.canvas1.mask * mask
            self.mask1 = mask
            self.update_canvas_img()
        except:
            self.msg = 'invalid mask '
            self.update_msg()


    def generate_mask2(self):
        try:
            tmp = deepcopy(self.canvas1.current_img)
            mask = np.ones(tmp.shape)
            tmp1 = self.tx_mask2.text()
            if tmp1[0] == '<':
                thresh = float(tmp1[1:])
                mask[tmp < thresh] = 0
            elif tmp1[0] == '>':
                thresh = float(tmp1[1:])
                mask[tmp > thresh] = 0
            else:
                thresh = float(tmp1)
                mask[tmp > thresh] = 0
            self.canvas1.mask = self.canvas1.mask * mask
            self.mask2 = mask
            self.update_canvas_img()
        except:
            self.msg = 'invalid mask '
            self.update_msg()


    def rm_mask1(self):
        self.canvas1.mask = self.mask2
        self.mask1 = np.array([1])
        self.update_canvas_img()

    def rm_mask2(self):
        self.canvas1.mask = self.mask1
        self.mask2 = np.array([1])
        self.update_canvas_img()


    def _roi_fit(self):
        roi_selected = 1
        canvas = self.canvas1
        roi_list = canvas.roi_list
        img = deepcopy(self.img_update)
        x_data = self.xanes_eng
        try:
            n = int(self.tx_fit_roi.text())
            roi_selected = 'roi_' + str(n)
        except:
            print('index should be integer')
            n = 0
        n_roi = self.lst_roi.count()
        if n > n_roi or n < 0:
            self.msg = 'roi not exist'
            roi_selected = 0
            self.update_msg()
        if roi_selected:
            print(f'{roi_selected}')
            try:
                roi_cord = np.int32(np.array(roi_list[roi_selected]))
                print(f'{roi_cord}')
                x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(x1, x2)
                x2 = max(x1, x2)
                y1 = min(y1, y2)
                y2 = max(y1, y2)
                cord = [x1, x2, y1, y2]
                prj = img[:, y1:y2, x1:x2]
                fit_coef = self.xanes_2d_fit[:, y1:y2, x1:x2]
                fit_coef = np.mean(np.mean(fit_coef, axis=1), axis=1)
                s = prj.shape
                y_data = np.mean(np.mean(prj, axis=1), axis=1)                
                y_fit = 0
                for i in range(self.num_ref):
                    ref = self.spectrum_ref[f'ref{i}']
                    tmp = interp1d(ref[:,0], ref[:,1], kind='cubic')
                    ref_interp = tmp(x_data).reshape(y_data.shape)
                    y_fit += fit_coef[i] * ref_interp
                fit_success = 1
            except:
                self.msg = 'something wrong, no fitting performed'
                self.update_msg()
                y_data = np.zeros(x_data.shape)
                y_fit = y_data
                fit_success = 0
                cord = [0,0,0,0]
                fit_coef = [0]
        else:
            y_data = np.zeros(x_data.shape)
            y_fit = y_data
            fit_success = 0
            cord = [0,0,0,0]
            fit_coef = [0]
        return x_data, y_data, y_fit, fit_coef, cord, fit_success


    def plot_roi_fit(self, return_flag):
        x_data, y_data, y_fit, fit_coef, cord, fit_success = self._roi_fit()
        elem = self.tx_elem.text()
        elem = elem.replace(' ', '')
        elem = elem.replace(';', ',')
        elem = elem.split(',')
        if elem[0] == '':
            elem = []
        if fit_success:
            title = ''
            for i in range(self.num_ref):
                try:
                    plot_label = elem[i]
                except:
                    plot_label = f'ref_{i}'
                self.elem_label.append(plot_label)
                title += plot_label + f': {fit_coef[i]:.3f}, '
            plt.figure()
            legend = []
            line_raw, = plt.plot(x_data, y_data, 'b.', label='Experiment data')
            legend.append(line_raw)
            line_fit, = plt.plot(x_data, y_fit, color='r', label='Fitted')
            legend.append(line_fit)
            plt.legend(handles=legend)
            plt.title(title)
            plt.show()
        if return_flag:
            return x_data, y_data, y_fit, fit_coef, cord, fit_success
            
            

    def export_roi_fit(self):
        x_data, y_data, y_fit, fit_coef, cord, fit_success = self.plot_roi_fit(return_flag=1)
        dir_path = self.fpath + '/ROI_fit'
        try:
            os.makedirs(dir_path)
            make_directory_success = True
        except:
            if os.path.exists(dir_path):
                make_directory_success = True
            else:
                print('access directory: ' + dir_path + ' failed')
                make_directory_success = False
        if make_directory_success and fit_success:
            n = int(self.tx_fit_roi.text())
            label_raw = 'roi_' + str(n)
            label_fit = label_raw + '_fit'
            fn_spec = dir_path + '/' + 'spectrum_' + label_fit + '.txt'
            fn_cord = dir_path + '/' + 'coordinates_' + label_fit + '.txt'
            roi_dict_spec = {'X_eng': pd.Series(self.xanes_eng)}
            roi_dict_cord = {}
            if fit_success:
                roi_dict_spec[label_raw] = pd.Series(y_data)
                roi_dict_spec[label_fit] = pd.Series(y_fit)
                roi_dict_spec[label_raw + '_fit_coef'] = pd.Series(fit_coef)
                roi_dict_cord[label_raw] = pd.Series([cord[0], cord[1], cord[2], cord[3]], index=['x1', 'y1', 'x2', 'y2'])
            df_spec = pd.DataFrame(roi_dict_spec)
            df_cord = pd.DataFrame(roi_dict_cord)
            with open(fn_spec, 'w') as f:
                df_spec.to_csv(f, float_format='%.4f', sep=' ', index=False)
            with open(fn_cord, 'w') as f:
                df_cord.to_csv(f, float_format='%.4f', sep=' ')
            self.msg = 'Fitted ROI spectrum file saved:    ' + fn_spec
            self.update_msg()
        else:
            self.msg = 'export fails'
            self.update_msg()


    def select_file(self):
        self.tx_hdf_xanes.setEnabled(True)
        self.tx_hdf_eng.setEnabled(True)

    def xanes_colormix(self):
        color = self.tx_fit_color.text()
        color = color.replace(' ','')
        color = color.replace(';', ',')
        color = color.split(',')
        if color[0] == '':
            color = ['r', 'g', 'b', 'c', 'p', 'y']
        color_vec = self.convert_rgb_vector(color)
        img = self.xanes_2d_fit * self.mask1 * self.mask2
        
        s = img.shape
        img_color = np.zeros([s[1], s[2], 3])
        cR, cG, cB = 0, 0, 0 
        for i in range(s[0]):
            cR += img[i] * color_vec[i][0]
            cG += img[i] * color_vec[i][1]
            cB += img[i] * color_vec[i][2]
        img_color[:,:,0] = cR
        img_color[:,:,1] = cG
        img_color[:,:,2] = cB
        plt.figure()
        plt.imshow(img_color)
        plt.show()
        print('plot the colormix ')


    def convert_rgb_vector(self, color):
        n = len(color)
        vec = np.zeros([n, 3])
        for i in range(n):
            if color[i] == 'r': vec[i] = [1, 0, 0]
            if color[i] == 'g': vec[i] = [0, 1, 0] 
            if color[i] == 'b': vec[i] = [0, 0, 1]
            if color[i] == 'c': vec[i] = [0, 1, 1] 
            if color[i] == 'p': vec[i] = [1, 0, 1] 
            if color[i] == 'y': vec[i] = [1, 1, 0]
        return vec 

    def save_2Dfit(self):
        pre_s = float(self.tx_fit_pre_s.text())
        pre_e = float(self.tx_fit_pre_e.text())
        post_s = float(self.tx_fit_post_s.text())
        post_e = float(self.tx_fit_post_e.text())
        try:
            options = QFileDialog.Option()
            options |= QFileDialog.DontUseNativeDialog
            file_type = 'hdf files (*.h5)'
            fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
            hf = h5py.File(fn, 'w')
            hf.create_dataset('xanes_fit', data = self.xanes_2d_fit* self.mask1 * self.mask2)
            hf.create_dataset('X_eng', data = self.xanes_eng)
            hf.create_dataset('pre_edge', data = [pre_s, pre_e])
            hf.create_dataset('post_edge', data = [post_s, post_e])
            hf.create_dataset('unit', data = 'keV')
            for i in range(self.num_ref):
                hf.create_dataset(f'ref{(i)}', data=self.elem_label[i])
            hf.close()

            print(f'xanes_fit has been saved to file: {fn}')
            self.msg = f'xanes_fit has been saved to file: {fn}'
        except:
            self.msg = 'file saving fails ...'
        finally:
            self.update_msg()


    def remove_bkg(self):
        '''
        Treat if it is fluorescent image.
        calculate the mean value of 5% of the lowest pixel value, and substract this value from the whole image
        '''
        self.pb_rmbg.setText('normalizing ..')
        self.pb_rmbg.setEnabled(False)
        QApplication.processEvents()

        canvas = self.canvas1
        prj = deepcopy(canvas.img_stack)
        prj[np.isnan(prj)] = 0
        prj[np.isinf(prj)] = 0
        prj[prj < 0] = 0
        s = prj.shape
        prj_sort = np.sort(prj)  # sort each slice
        prj_sort = prj_sort[:, :, 0:int(s[1] * s[2] * 0.05)]
        slice_avg = np.mean(np.mean(prj_sort, axis=2), axis=1)  # average for each slice
        prj = np.array([prj[i] - slice_avg[i] for i in range(s[0])])
        prj[prj < 0] = 0
        self.img_update = deepcopy(prj)
        del prj, slice_avg
        self.pb_rmbg.setEnabled(True)
        self.pb_rmbg.setText('Norm. Bkg. (opt.) ')
        QApplication.processEvents()
        if self.cb1.findText('Image updated') < 0:
            self.cb1.addItem('Image updated')
        self.cb1.setCurrentText('Image updated')
        self.update_canvas_img()
        

    def xanes_img_smooth(self):
        self.pb_filt.setEnabled(False)
        self.pb_filt.setText('Smoothing ...')
        QApplication.processEvents()
        canvas = self.canvas1
        img_stack = deepcopy(canvas.img_stack)
        kernal_size = int(self.tx_filt.text())
        for i in range(img_stack.shape[0]):
            img_stack[i] = medfilt2d(img_stack[i], kernal_size)
        self.img_update = deepcopy(img_stack)
        if self.cb1.findText('Image updated') < 0:
            self.cb1.addItem('Image updated')
        self.pb_filt.setEnabled(True)
        self.pb_filt.setText('Median filter')
        self.msg = 'Image smoothed'
        self.cb1.setCurrentText('Image updated')
        self.update_msg()
        self.update_canvas_img()

        QApplication.processEvents()


    def plot_spectrum(self):
        canvas = self.canvas1
        img_stack = deepcopy(canvas.img_stack)
        # img_stack = self.img_update
        plt.figure();
        roi_color = canvas.roi_color
        roi_list = canvas.roi_list
        x = self.xanes_eng
        legend = []
        try:
            for item in self.lst_roi.selectedItems():
                plt.hold(True)
                print(item.text())
                plot_color = roi_color[item.text()]
                roi_cord = np.int32(np.array(roi_list[item.text()]))
                plot_label = item.text()
                x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(x1, x2)
                x2 = max(x1, x2)
                y1 = min(y1, y2)
                y2 = max(y1, y2)
                roi_spec = np.mean(np.mean(img_stack[:, y1:y2, x1:x2, ], axis=1), axis=1)
                line, = plt.plot(x, roi_spec, marker='.', color=plot_color, label=plot_label)
                legend.append(line)
            print(legend)
            plt.legend(handles=legend)
            plt.show()
        except:
            self.msg = 'no spectrum available for current image stack ...'
            self.update_msg()
        

    def show_roi(self):
        plt.figure()
        canvas = self.canvas1
        current_img = canvas.current_img
        current_colormap = canvas.colormap
        cmin, cmax = canvas.cmin, canvas.cmax
        s = current_img.shape
        plt.imshow(current_img * self.mask1 * self.mask2, cmap=current_colormap, vmin=cmin, vmax=cmax)
        plt.axis('equal')
        plt.axis('off')
        plt.colorbar()
        roi_color = canvas.roi_color
        roi_list = canvas.roi_list
        for item in self.lst_roi.selectedItems():
            plt.hold(True)
            print(item.text())
            plot_color = roi_color[item.text()]
            roi_cord = np.int32(np.array(roi_list[item.text()]))
            plot_label = item.text()
            x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
            x = [x1, x2, x2, x1, x1]
            y = [y1, y1, y2, y2, y1]
            plt.plot(x, y, '-', color=plot_color, linewidth=1.0, label=plot_label)
            roi_name = '#' + plot_label.split('_')[-1]
            plt.annotate(roi_name, xy=(x1, y1 - 40),
                         bbox={'facecolor': plot_color, 'alpha': 0.5, 'pad': 2},
                         fontsize=10)
        # self.pb_roi_showhide.setText('Hide ROI')
        plt.show()

    def hide_roi(self):
        self.update_canvas_img()


    def export_spectrum(self):
        self.show_roi()
        self.tx_file_index
        try:
            os.makedirs(self.fpath + '/ROI')
            make_director_success = True
        except:
            if os.path.exists(self.fpath + '/ROI'):
                make_director_success = True
            else:
                print(self.fpath + '/ROI failed')
                make_director_success = False
                self.msg = 'Access to directory: "' + self.path + '/ROI' + '" fails'
                self.update_msg()
        if make_director_success:
            fn_spec = 'spectrum_roi_from_' + self.cb1.currentText() + '_' + self.tx_file_index.text() + '.txt'
            fn_spec = self.fpath + '/ROI/' + fn_spec

            fn_cord = 'coordinates_roi_from_' + self.cb1.currentText() + '_' + self.tx_file_index.text() + '.txt'
            fn_cord = self.fpath + '/ROI/' + fn_cord

            canvas = self.canvas1
            img_stack = deepcopy(canvas.img_stack)
            # img_stack = deepcopy(self.img_update)
            roi_list = canvas.roi_list
            x = self.xanes_eng
            roi_dict_spec = {'X_eng': pd.Series(x)}
            roi_dict_cord = {}
            for item in self.lst_roi.selectedItems():
                roi_cord = np.int32(np.array(roi_list[item.text()]))
                plot_label = item.text()
                x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(x1, x2)
                x2 = max(x1, x2)
                y1 = min(y1, y2)
                y2 = max(y1, y2)
                area = (x2 - x1) * (y2 - y1)
                roi_spec = np.mean(np.mean(img_stack[:, y1:y2, x1:x2, ], axis=1), axis=1)
                roi_spec = np.around(roi_spec, 3)
                roi_dict_spec[plot_label] = pd.Series(roi_spec)
                roi_dict_cord[plot_label] = pd.Series([x1, y1, x2, y2, area], index=['x1', 'y1', 'x2', 'y2', 'area'])
            df_spec = pd.DataFrame(roi_dict_spec)
            df_cord = pd.DataFrame(roi_dict_cord)

            with open(fn_spec, 'w') as f:
                df_spec.to_csv(f, float_format='%.4f', sep=' ', index=False)
            with open(fn_cord, 'w') as f:
                df_cord.to_csv(f, float_format='%.4f', sep=' ')

            self.roi_file_id += 1
            self.tx_file_index.setText(str(self.roi_file_id))
            print(fn_spec + '  saved')
            self.msg = 'ROI spectrum file saved:   ' + fn_spec
            self.update_msg()
        

    def reset_roi(self):
        canvas = self.canvas1
        self.lst_roi.clear()
        canvas.current_roi = [0, 0, 0, 0]
        canvas.current_color = 'red'
        canvas.roi_list = {}
        canvas.roi_count = 0
        canvas.roi_color = {}
        self.update_canvas_img()
        s = canvas.current_img.shape
        self.tx_roi_x1.setText('0.0')
        self.tx_roi_x2.setText('{:3.1f}'.format(s[1]))
        self.tx_roi_y1.setText('{:3.1f}'.format(0))
        self.tx_roi_y2.setText('{:3.1f}'.format(s[0]))
        pass
    

    def draw_roi(self):
        self.pb_roi_draw.setEnabled(False)
        QApplication.processEvents()
        canvas = self.canvas1
        canvas.draw_roi()
        pass
    

    def fit_edge(self):
        try:
            pre_s = float(self.tx_fit_pre_s.text())
            pre_e = float(self.tx_fit_pre_e.text())
            post_s = float(self.tx_fit_post_s.text())
            post_e = float(self.tx_fit_post_e.text())

            canvas = self.canvas1
            img_stack = deepcopy(canvas.img_stack)
            roi_list = canvas.roi_list
            x_eng = deepcopy(self.xanes_eng)        
            num_roi_sel = len(self.lst_roi.selectedItems())
            roi_spec_fit = np.zeros([len(x_eng), num_roi_sel+1])
            roi_spec_fit[:,0] = x_eng
            
            n = 0
            for item in self.lst_roi.selectedItems():
                n = n + 1
                plt.figure()  # generate figure for each roi
                plt.subplot(211)
                roi_cord = np.int32(np.array(roi_list[item.text()]))
                plot_label = item.text()
                x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
                roi_spec = np.mean(np.mean(img_stack[:, y1:y2, x1:x2, ], axis=1), axis=1)
                # roi_spec_tmp = deepcopy(roi_spec)
                roi_spec_fit[:,n], y_pre_fit, y_post_fit = normalize_1D_xanes(roi_spec, x_eng, [pre_s, pre_e], [post_s, post_e])

                # roi_spec_fit[:, n] = deepcopy(roi_spec)

                
                # # fit pre-edge
                # xs, xe = find_nearest(x_eng, pre_s), find_nearest(x_eng, pre_e)
                # pre_eng = x_eng[xs:xe]
                # pre_spec = roi_spec[xs:xe]
                # print(f'{pre_spec.shape}')
                # if len(pre_eng) > 1:
                #     y_pre_fit = fit_curve(pre_eng, pre_spec, x_eng)
                #     roi_spec_tmp = roi_spec - y_pre_fit
                #     pre_fit_flag = True
                # elif len(pre_eng) <= 1:
                #     y_pre_fit = np.ones(x_eng.shape) * roi_spec[xs]
                #     roi_spec_tmp = roi_spec - y_pre_fit
                #     pre_fit_flag = True
                # else:
                #     print('invalid pre-edge assignment')
                #     self.msg = 'invalid pre-edge assignment'
                #     self.update_msg()
                #
                # # fit post-edge
                # xs, xe = find_nearest(x_eng, post_s), find_nearest(x_eng, post_e)
                # post_eng = x_eng[xs:xe]
                # post_spec = roi_spec_tmp[xs:xe]
                # if len(post_eng) > 1:
                #     y_post_fit = fit_curve(post_eng, post_spec, x_eng)
                #     post_fit_flag = True
                # elif len(post_eng) <= 1:
                #     y_post_fit = np.ones(x_eng.shape) * roi_spec_tmp[xs]
                #     post_fit_flag = True
                # else:
                #     print('invalid pre-edge assignment')
                #     self.msg = 'invalid pre-edge assignment'
                #     self.update_msg()
                #
                # if pre_fit_flag and post_fit_flag:
                #     roi_spec_fit[:, n] = roi_spec_tmp * 1.0 / y_post_fit
                plt.subplots_adjust(hspace = 0.5)
                plt.plot(x_eng, roi_spec, '.', color='gray')
                plt.plot(x_eng, y_pre_fit, 'b', linewidth=1)
                plt.plot(x_eng, y_post_fit + y_pre_fit, 'r', linewidth=1)
                plt.title('pre-post edge fitting for ' + plot_label)
    
                plt.subplot(212)  # plot normalized spectrum
                plt.plot(x_eng, roi_spec_fit[:, n])
                plt.title('normalized spectrum for ' + plot_label)
                plt.show()    
            self.roi_spec = roi_spec_fit
        except:
            self.msg = 'Fitting error ...'
            self.update_msg()


    def save_normed_roi(self):
        try:
            os.makedirs(self.fpath + '/ROI/fitted_roi')
        except:
            print(self.fpath + '/ROI failed')
            pass
        try:
            fn_spec = 'fitted_spectrum_roi_from_' + self.cb1.currentText() + '_' + self.tx_file_index.text() + '.txt'
            fn_spec = self.fpath + '/ROI/fitted_roi/' + fn_spec
    
            fn_cord = 'fitted_coordinates_roi_from_' + self.cb1.currentText() + '_' + self.tx_file_index.text() + '.txt'
            fn_cord = self.fpath + '/ROI/fitted_roi/' + fn_cord
    
            canvas = self.canvas1            
            roi_spec = deepcopy(self.roi_spec)
            roi_spec = np.around(roi_spec, 3)
            roi_list = canvas.roi_list
            roi_dict_spec = {'X_eng': pd.Series(roi_spec[:, 0])}
            roi_dict_cord = {}
            n = 0
            for item in self.lst_roi.selectedItems():
                n = n + 1
                plot_label = item.text()
                roi_cord = np.int32(np.array(roi_list[item.text()]))
                x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(x1, x2)
                x2 = max(x1, x2)
                y1 = min(y1, y2)
                y2 = max(y1, y2)
                area = (x2 - x1) * (y2 - y1)            
                roi_dict_spec[plot_label] = pd.Series(roi_spec[:, n])
                roi_dict_cord[plot_label] = pd.Series([x1, y1, x2, y2, area], index=['x1', 'y1', 'x2', 'y2', 'area'])
            df_spec = pd.DataFrame(roi_dict_spec)
            df_cord = pd.DataFrame(roi_dict_cord)
    
            # with open(fn_spec, 'w') as f:
            #     df_spec.to_csv(f, sep=' ', index=False)
            with open(fn_cord, 'w') as f:
                df_cord.to_csv(f, sep=' ')

            np.savetxt(fn_spec, np.array(df_spec), '%2.4f')

            print(fn_spec + '  saved')
            self.msg = 'Fitted ROI spectrum saved:   ' + fn_spec
        except:
            self.msg = 'Save fitted roi spectrum fails ...'
        finally:
            self.update_msg()

        
        
        
    # ##########################################
    def fit_edge_img(self):
        pre_s = float(self.tx_fit_pre_s.text())
        pre_e = float(self.tx_fit_pre_e.text())
        post_s = float(self.tx_fit_post_s.text())
        post_e = float(self.tx_fit_post_e.text())
        canvas = self.canvas1
        try:
            self.pb_fit_img.setText('wait ...')
            self.pb_fit_img.setEnabled(False)
            QApplication.processEvents()
            img_norm = deepcopy(canvas.img_stack)
            s0 = img_norm.shape
            x_eng = deepcopy(self.xanes_eng)
            img_norm = normalize_2D_xanes(img_norm, x_eng, [pre_s, pre_e], [post_s, post_e])
            self.msg = '2D Spectra image normalized'
            self.img_update = deepcopy(img_norm)
            self.cb1.setCurrentText('Image updated')
            self.update_canvas_img()
            self.pb_fit_img.setText('Norm image')
            self.pb_fit_img.setEnabled(True)
            self.pb_save_fit_img.setEnabled(True)
            if self.cb1.findText('Image updated') < 0:
                self.cb1.addItem('Image updated')
            self.cb1.setCurrentText('Image updated')
            QApplication.processEvents()
            del img_norm
            # del img_pre, img_post, b0_pre, b0_post, b1_pre, b1_post
        except:
            self.msg = 'fails to normalize 2D spectra image'
        finally:
            self.update_canvas_img()
            self.update_msg()



    def save_normed_img(self):
        try:
            options = QFileDialog.Option()
            options |= QFileDialog.DontUseNativeDialog
            file_type = 'hdf files (*.h5)'
            fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
            with h5py.File(fn, 'w') as hf:
                hf.create_dataset('img_spec_norm', data = self.img_update* self.mask1 * self.mask2)
            print(f'normalized 2D spectra image has been saved to file: {fn}')
            self.msg = f'xanes_fit has been saved to file: {fn}'
        except:
            self.msg = 'file saving fails ...'
        finally:
            self.update_msg()
        pass


    def delete_single_img(self):
        canvas = self.canvas1
        if canvas.img_stack.shape[0] < 2:
            self.msg = 'cannot delete image in single-image-stack'
            self.update_msg()
        else:
            msg = QMessageBox()
            msg.setText('This will delete the image in \"raw data\", and \"image_update\" ')
            msg.setWindowTitle('Delete single image')
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

            reply = msg.exec_()
            if reply == QMessageBox.Ok:
                img_type = self.cb1.currentText()
                current_slice = self.sl1.value()
                try:
                    self.img_xanes = np.delete(self.img_xanes, current_slice, axis=0)
                except:
                    print('cannot delete img_xanes')
                try:
                    self.img_update = np.delete(self.img_update, current_slice, axis=0)
                except:
                    print('cannot delete img_update')
                try:
                    self.xanes_eng = np.delete(self.xanes_eng, current_slice, axis=0)
                    st = '{0:3.1f}, {1:3.1f}, ..., {2:3.1f}  (totally, {3} angles)'.format(self.xanes_eng[0],self.xanes_eng[1],self.xanes_eng[-1],len(self.xanes_eng))
                    self.lb_ang1.setText(st)  # update angle information showing on the label
                except:
                    print('cannot delete energy')
                self.msg = 'image #{} has been deleted'.format(current_slice)
                self.update_msg()
                self.update_canvas_img()


    def load_image(self):
        self.default_layout()
        self.pb_ld.setEnabled(False)
        flag_read_from_file = 0
        flag_read_from_database = 0
        if len(self.tx_hdf_xanes.text()):
            dataset_xanes = self.tx_hdf_xanes.text()
        else:
            dataset_xanes = 'img_xanes'
        if len(self.tx_hdf_eng.text()):
            dataset_eng = self.tx_hdf_eng.text()
        else:
            dataset_eng = 'X_eng'

        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        if self.rd_hdf.isChecked() == True:
            file_type = 'hdf files (*.h5)'
            flag_read_from_file = 1
            flag_read_from_database =0
        elif self.rd_tif.isChecked() == True:
            file_type = 'tiff files (*.tif, *.tiff)'
            flag_read_from_file = 1
            flag_read_from_database = 0
        else:
            flag_read_from_database = 1
            flag_read_from_file = 0
            
        if flag_read_from_file:
            fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
            if fn:
                print(fn)
                fn_relative = fn.split('/')[-1]
                self.fpath = fn[:-len(fn_relative)-1]
                print(f'current path: {self.fpath}')
#                self.fn = fn
                self.fn_relative = fn_relative

                self.lb_ip.setStyleSheet('color: rgb(200, 50, 50);')
                if self.rd_hdf.isChecked() == True:  # read hdf file
                    f = h5py.File(fn, 'r')

                    # read energy
                    try:
                        self.xanes_eng = np.array(f[dataset_eng])  # Generally, it is in unit of keV
                        # if min(self.xanes_eng) < 4000:  # make sure it is in unit of KeV
                        #     self.xanes_eng = self.xanes_eng * 1000
                        st = '{0:2.4f}, {1:2.4f}, ..., {2:2.4f}  (totally, {3} energies)'.format(self.xanes_eng[0],
                                                                                                 self.xanes_eng[1],
                                                                                                 self.xanes_eng[-1],
                                                                                                 len(self.xanes_eng))
                        self.tx_fit_pre_s.setText('{:2.4f}'.format(min(self.xanes_eng) - 0.001))
                        self.tx_fit_pre_e.setText('{:2.4f}'.format(min(self.xanes_eng) + 0.010))
                        self.tx_fit_post_e.setText('{:2.4f}'.format(max(self.xanes_eng) + 0.001))
                        self.tx_fit_post_s.setText('{:2.4f}'.format(max(self.xanes_eng) - 0.010))
                        self.lb_ang1.setText(st)

                    except:
                        self.xanes_eng = np.array([0])
                        self.lb_ang1.setText('No energy data ...')
                        self.lb_ang2.setVisible(True)
                        self.tx_ang.setVisible(True)
                        self.pb_ang.setVisible(True)
                        self.msg = self.msg + ';  Energy list not exist'
                        self.update_msg()


                    # read xanes-scan image
                    try:
                        self.img_xanes = np.array(f[dataset_xanes])
                        self.img_update = deepcopy(self.img_xanes)
                        self.rot_cen = self.img_xanes.shape[2] / 2
                        print('Image size: ' + str(self.img_xanes.shape))
                        self.pb_norm_txm.setEnabled(True)

                        self.pb_align.setEnabled(True)
                        self.pb_del.setEnabled(True)  # delete single image
                        self.pb_filt.setEnabled(True)  # delete single image
                        self.pb_rmbg.setEnabled(True)
                        self.pb_align_roi.setEnabled(True)
                        self.msg = 'image shape: {0}'.format(self.img_xanes.shape)
                        self.lb_ip.setText('File loaded:   {}'.format(fn))
                        if (len(self.xanes_eng) != self.img_xanes.shape[0]):
                            self.msg = 'number of energy does not match number of images, try manual input ...'
                    except:
                        self.img_xanes = np.zeros([1, 100, 100])
                        print('xanes image not exist')
                        self.lb_ip.setText('File loading fails ...')
                        self.msg = 'xanes image not exist'

                    finally:
                        self.update_canvas_img()
                        self.update_msg()
    

                    f.close()
    
                else:  # read tiff file
                    try:
                        self.img_xanes = np.array(io.imread(fn))
                        # self.tx_sli_st.setText(str(self.img_xanes.shape[1] // 2))
                        # self.tx_sli_end.setText(str(self.img_xanes.shape[1] // 2 + 1))
                        self.msg = 'image shape: {0}'.format(self.img_xanes.shape)
                        self.update_msg()
                        self.pb_norm_txm.setEnabled(True)  # remove background
                        self.lb_ip.setText('File loaded:   {}'.format(fn))
                        self.pb_del.setEnabled(True)  # delete single image
                        QApplication.processEvents()
                    except:
                        self.img_xanes = np.zeros([1, 100, 100])
                        print('image not exist')
                    finally:
                        self.img_update = deepcopy(self.img_xanes)
                        self.update_canvas_img()
                        self.xanes_eng = np.array([])
                        self.lb_ang1.setText('No energy data ...')
                        self.lb_ang2.setVisible(True)
                        self.tx_ang.setVisible(True)
                        self.pb_ang.setVisible(True)
        elif flag_read_from_database:
            print('read_from_databroker, not implemented yet')
        else:
            print('Something happened in loading file ... :(')
        self.pb_ld.setEnabled(True)

    def update_msg(self):
        self.lb_msg.setFont(self.font1)
        self.lb_msg.setText('Message: ' + self.msg)
        self.lb_msg.setStyleSheet('color: rgb(200, 50, 50);')

    def manu_energy_input(self):
        energy_list_old = deepcopy(self.xanes_eng)
        com = 'self.xanes_eng = np.array(' + self.tx_ang.text() + ')'
        try:
            exec(com)
            if len(self.xanes_eng) != self.img_xanes.shape[0] or self.img_xanes.shape[0] <=3:
                self.msg = 'number of energy does not match number of images'
                self.xanes_eng = deepcopy(energy_list_old)
            else:
                st = '{0:2.4f}, {1:2.4f}, ..., {2:2.4f}  totally, {3} energies'.format(self.xanes_eng[0], self.xanes_eng[1], self.xanes_eng[-1], len(self.xanes_eng))
                self.lb_ang1.setText(st)
                self.msg = 'energy list has been updated'
        except:
            self.msg = 'un-recognized python command '
        finally:
            self.update_msg()


    def sliderval(self):
        canvas = self.canvas1
        img_index = self.sl1.value()
        img = canvas.img_stack[img_index]
        img = np.array(img)
        # print(f'{canvas.mask.shape}')
        canvas.update_img_one(img, img_index=img_index)


    def norm_txm(self):
        self.pb_norm_txm.setText('wait ...')
        QApplication.processEvents()

        canvas = self.canvas1
        tmp  = canvas.img_stack
        tmp = -np.log(tmp)
        tmp[np.isinf(tmp)] = 0
        tmp[np.isnan(tmp)] = 0
        self.img_update = deepcopy(tmp)
        self.pb_norm_txm.setText('Norm. TMX (-log)')
        del tmp
        QApplication.processEvents()

        if self.cb1.findText('Image updated') < 0:
            self.cb1.addItem('Image updated')
        print('img = -log(img) \n"img_updated" has been created or updated')
        self.msg = 'img = -log(img)'
        self.update_msg()

        self.cb1.setCurrentText('Image updated')
        self.update_canvas_img()


    def xanes_align_img(self):
        self.pb_align.setText('Aligning ...')
        QApplication.processEvents()
        self.pb_align.setEnabled(False)

        canvas = self.canvas1
        prj = deepcopy(canvas.img_stack) * self.mask1 * self.mask2
        img_ali = deepcopy(canvas.img_stack)
        n = prj.shape[0]
        self.shift_list = []
        try:
            ref_index = int(self.tx_ali_ref.text())

            if  ref_index <0 or ref_index >= prj.shape[0]:   # sequential align next image with its previous image
                self.shift_list.append([0, 0])
                for i in range(1, n):
                    print('Aligning image slice ' + str(i))

                    _, rsft, csft = align_img(prj[i - 1], prj[i])
                    img_ali[i] = shift(canvas.img_stack[i], [rsft, csft], mode='constant', cval=0)
                    self.shift_list.append([rsft, csft])
                    self.msg = f'Aligned image slice {i}, row_shift: {rsft}, col_shift: {csft}'
                    self.update_msg()
                    QApplication.processEvents()
            else:                                            # align all images with image_stack[ref_index]
                for i in range(0, n):
                    print('Aligning image slice ' + str(i))
                    self.msg = 'Aligning image slice ' + str(i)
                    self.update_msg()
                    _, rsft, csft = align_img(prj[ref_index], prj[i])
                    self.msg = f'Aligned image slice {i}, row_shift: {rsft}, col_shift: {csft}'
                    self.update_msg()
                    img_ali[i] = shift(canvas.img_stack[i], [rsft, csft], mode='constant', cval=0)
                    self.shift_list.append([rsft, csft])
                    QApplication.processEvents()

            self.img_update = deepcopy(img_ali)

            if self.cb1.findText('Image updated') < 0:
                self.cb1.addItem('Image updated')
                self.cb1.setCurrentText('Image updated')
            self.pb_align.setText('Align Img')
            self.pb_align.setEnabled(True)
            print('Image aligned.\n Item "Aligned Image" has been added.')
            self.msg = 'Image aligning finished'
        except:
            self.msg = 'image stack has only one image slice, aligning aborted... '
        finally:
            self.update_msg()
            self.update_canvas_img()
            del prj, img_ali



    def xanes_align_img_roi(self):
        self.pb_align_roi.setText('Aligning ...')
        QApplication.processEvents()
        self.pb_align_roi.setEnabled(False)
        canvas = self.canvas1
        roi_list = canvas.roi_list
        img_ali = deepcopy(canvas.img_stack)
        self.shift_list = []
        try:
            n = int(self.tx_ali_roi.text())
            roi_selected = 'roi_' + str(n)
        except:
            print('index should be integer')
            n = 0
            roi_selected = 'None'
        n_roi = self.lst_roi.count()
        if n > n_roi or n < 0:
            n = 0
            roi_selected = 'None'

        if not roi_selected == 'None':
            print(f'{roi_selected}')
            try:
                roi_cord = np.int32(np.array(roi_list[roi_selected]))
                print(f'{roi_cord}')
                x1, y1, x2, y2 = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                prj = (img_ali * self.mask1 * self.mask2)[:, y1: y2, x1: x2]
                s = prj.shape
                ref_index = int(self.tx_ali_ref.text())
                if ref_index < 0 or ref_index >= s[0]:  # sequantial align next image with its previous image
                    print(f'sequency : {ref_index}')
                    self.shift_list.append([0, 0])
                    for i in range(1, s[0]):
                        print('Aligning image slice ' + str(i))
                        _, rsft, csft = align_img(prj[i - 1], prj[i])
                        img_ali[i] = shift(img_ali[i], [rsft, csft], mode='constant', cval=0)
                        self.msg = f'Aligned image slice {i}, row_shift: {rsft}, col_shift: {csft}' 
                        self.update_msg()
                        self.shift_list.append([rsft, csft])

                        QApplication.processEvents()
                else:  # align all images with image_stack[ref_index]
                    for i in range(0, s[0]):
                        print('Aligning image slice ' + str(i))
                        _, rsft, csft = align_img(prj[ref_index], prj[i])
                        self.shift_list.append([rsft, csft])
                        self.msg = f'Aligned image slice {i}, row_shift: {rsft}, col_shift: {csft}' 
                        self.update_msg()
                        img_ali[i] = shift(img_ali[i], [rsft, csft], mode='constant', cval=0)
                        QApplication.processEvents()
                self.img_update = deepcopy(img_ali)
                if self.cb1.findText('Image updated') < 0:
                    self.cb1.addItem('Image updated')
                    self.update_canvas_img()

                print('Image aligned.\n Item "Aligned Image" has been added.')
                self.msg = 'Image aligning finished'
            except:
                self.msg = 'image stack has only one image slice, aligning aborted... '
            finally:
                self.pb_align_roi.setText('Align Img (ROI)')
                self.pb_align_roi.setEnabled(True)
                self.update_msg()
                self.update_canvas_img()
                del prj, img_ali
        else:
            self.pb_align_roi.setText('Align Img')
            self.pb_align_roi.setEnabled(True)
            self.msg = 'Invalid roi index for aligning, nothing applied'
            self.update_msg()


    def apply_shift(self):
        canvas =self.canvas1
        prj = deepcopy(canvas.img_stack)
        img_ali = np.zeros(prj.shape)
        n = prj.shape[0]
        n1 = len(self.shift_list)
        if n!=n1:
            self.msg = 'number of shifts does not match number of images, aligning will not perform'
        else:
            for i in range(min(n, n1)):
                rsft, csft = self.shift_list[i]
                print(f'Aligned image slice {i}, row_shift: {rsft}, col_shift: {csft}')
                img_ali[i] = shift(prj[i], [rsft, csft], mode='constant', cval=0)
                self.msg = f'Aligned image slice {i}, row_shift: {rsft}, col_shift: {csft}'
                self.update_msg()
            self.img_update = deepcopy(img_ali)
            if self.cb1.findText('Image updated') < 0:
                self.cb1.addItem('Image updated')
            self.cb1.setCurrentText('Image updated')
            self.update_canvas_img()
            self.msg = 'Applied shift to current image stack, Image upated'
        self.update_msg()


    def load_shift(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = 'txt files (*.txt)'
        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
        if fn:
            try:
                print(fn)
                fn_shift = fn.split('/')[-1]
                print(f'selected shift list: {fn_shift}')
                self.msg = f'selected shift list: {fn_shift}'
                self.lb_shift.setText('  '+ fn_shift)
                QApplication.processEvents()
                # self.spectrum_ref[f'ref{self.num_ref}'] = np.array(pd.read_csv(fn, sep=' '))
                self.shift_list = np.loadtxt(fn)
            except:
                print('un-recognized shift list')
            finally:
                self.update_msg()


    def save_shift(self):
        num = len(self.shift_list)
        if num == 0:
            self.msg = 'shift list not exist'
        elif num != self.img_xanes.shape[0]:
            self.msg = 'number of shifts not match number of images'
        else:
            fn = self.fpath + '/row_col_shift.txt'
            np.savetxt(fn, self.shift_list, '%3.2f')
            self.msg = fn + ' saved.'
        self.update_msg()

    def update_canvas_img(self):
        canvas = self.canvas1
        slide = self.sl1
        type_index = self.cb1.currentText()
        #        self.pb_sh1.setEnabled(False)
        QApplication.processEvents()

        canvas.draw_line = False
        self.pb_adj_cmap.setEnabled(True)
        self.pb_set_cmap.setEnabled(True)
        self.pb_del.setEnabled(True)

        if len(canvas.mask.shape) > 1:
            print(f'canvas_mask.shape = {canvas.mask.shape}')
 #           plt.figure();plt.imshow(canvas.mask);plt.show()
            sh = canvas.img_stack.shape
            canvas.special_info = None
            canvas.title = []
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0] - 1, 0))
        if type_index == 'Raw image':
            self.pb_roi_draw.setEnabled(True)
            canvas.x, canvas.y = [], []
            canvas.axes.clear()  # this is important, to clear the current image before another imshow()
            sh = self.img_xanes.shape
            canvas.img_stack = self.img_xanes
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.title = [f'#{i:3d},   {self.xanes_eng[i]:2.4f} keV' for i in range(len(self.xanes_eng))]
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0] - 1, 0))
            self.current_image = self.img_xanes[0]
        elif type_index == 'Image updated':
            img = self.img_update
            self.pb_roi_draw.setEnabled(True)
            canvas.x, canvas.y = [], []
            canvas.axes.clear()  # this is important, to clear the current image before another imshow()
            sh = img.shape
            canvas.img_stack = img
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.title = [f'#{i:3d},   {self.xanes_eng[i]:2.4f} keV' for i in range(len(self.xanes_eng))]
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0] - 1, 0))
            self.current_image = img[0]
        elif type_index == 'XANES Fit':
            img = self.xanes_2d_fit
            self.pb_roi_draw.setEnabled(True)
            canvas.x, canvas.y = [], []
            canvas.axes.clear()  # this is important, to clear the current image before another imshow()
            sh = img.shape
            canvas.img_stack = img
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.title = self.elem_label
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0] - 1, 0))
            self.current_image = img[0]
        elif type_index == 'XANES Fit error':
            img = self.xanes_fit_cost
            self.pb_roi_draw.setEnabled(True)
            canvas.x, canvas.y = [], []
            canvas.axes.clear()  # this is important, to clear the current image before another imshow()
            canvas.img_stack = img
            canvas.special_info = None
            canvas.current_img_index = 0
            canvas.title = []
            canvas.update_img_stack()
            slide.setMaximum(0)
            self.current_image = img[0]
        # elif type_index == 'Intensity plot':
        #     self.pb_roi_draw.setEnabled(False)
        #     canvas.axes.clear()  # this is important, to clear the current image before another imshow()
        #     self.pb_adj_cmap.setEnabled(True)
        #     self.pb_set_cmap.setEnabled(True)
        #     self.pb_del.setEnabled(True)
        #     self.sl1.setMaximum(0)
        #     canvas.title = []
        #     canvas.draw_line = True
        #     canvas.overlay_flag = False
        #
        #     canvas.add_line()
        #     canvas.draw_line = False
        #     canvas.overlay_flag = True
        #     canvas.colorbar_on_flag = True

        # elif type_index == 'ROI spectrum':
        #     self.pb_roi_draw.setEnabled(False)
        #     canvas.axes.clear()  # this is important, to clear the current image before another imshow()
        #     self.pb_adj_cmap.setEnabled(False)
        #     self.pb_set_cmap.setEnabled(False)
        #     self.pb_del.setEnabled(False)
        #     self.sl1.setMaximum(0)
        #     canvas.x = self.xanes_eng
        #     canvas.y = self.roi_spec
        #     canvas.draw_line = True
        #     canvas.overlay_flag = False
        #     canvas.plot_label = 'roi1'
        #     canvas.legend_flag = True
        #     canvas.add_line()
        #     canvas.legend_flag = False
        #     canvas.draw_line = False
        #     canvas.overlay_flag = True
        #     canvas.colorbar_on_flag = True
        #     canvas.x, canvas.y = [], []

        QApplication.processEvents()


    def update_roi_list(self, mode='add', item_name=''):
        # de-select all the existing selection
        if mode == 'add':
            for i in range(self.lst_roi.count()):
                item = self.lst_roi.item(i)
                item.setSelected(False)

            item = QListWidgetItem(item_name)
            self.lst_roi.addItem(item)
            self.lst_roi.setCurrentItem(item)
        elif mode == 'del_all':
            self.lst_roi.clear()
        elif mode == 'del':
            for selectItem in self.lst_roi.selectedItems():
                self.lst_roi.removeItemWidget(selectItem)
        else:
            pass


    def change_colormap(self):
        canvas = self.canvas1
        cmap = self.cb_cmap.currentText()
        canvas.colormap = cmap
        canvas.colorbar_on_flag = True
        canvas.update_img_one(canvas.current_img, canvas.current_img_index)


    def auto_contrast(self):
        canvas = self.canvas1
        cmin, cmax = canvas.auto_contrast()
        self.tx_cmax.setText('{:6.3f}'.format(cmax))
        self.tx_cmin.setText('{:6.3f}'.format(cmin))


    def set_contrast(self):
        canvas = self.canvas1
        cmax = np.float(self.tx_cmax.text())
        cmin = np.float(self.tx_cmin.text())
        canvas.set_contrast(cmin, cmax)


class MyCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=120, obj=[]):
        self.obj = obj
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = self.fig.add_subplot(111)
        #        self.roi = xanes_roi(fig=self.fig, ax=self.axes)
        self.axes.axis('off')
        self.cmax = 1
        self.cmin = 0
        self.img_stack = np.zeros([1, 100, 100])
        self.current_img = self.img_stack[0]
        self.current_img_index = 0
        self.mask = np.array([1])
        self.colorbar_on_flag = True
        self.colormap = 'terrain'
        self.title = []
        self.draw_line = False
        self.overlay_flag = True
        self.x, self.y, = [], []
        self.plot_label = ''
        self.legend_flag = False
        self.roi_list = {}
        self.roi_color = {}
        self.roi_count = 0
        self.current_roi = [0, 0, 0, 0]
        self.color_list = ['red', 'brown', 'orange', 'olive', 'green', 'cyan', 'blue', 'pink', 'purple', 'gray']
        self.current_color = 'red'
        self.special_info = None
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setParent(parent)

        self.mpl_connect('motion_notify_event', self.mouse_moved)

    def mouse_moved(self, mouse_event):
        if mouse_event.inaxes:
            x, y = mouse_event.xdata, mouse_event.ydata
            self.obj.lb_x_l.setText('x: {:3.2f}'.format(x))
            self.obj.lb_y_l.setText('y: {:3.2f}'.format(y))
            row = int(np.max([np.min([self.current_img.shape[0], y]), 0]))
            col = int(np.max([np.min([self.current_img.shape[1], x]), 0]))
            try:
                z = self.current_img[row][col]
                self.obj.lb_z_l.setText('intensity: {:3.3f}'.format(z))
            except:
                self.obj.lb_z_l.setText('')

    def update_img_stack(self):
        self.axes = self.fig.add_subplot(111)
        if self.img_stack.shape[0] == 0:
            img_blank = np.zeros([100, 100])
            return self.update_img_one(img_blank, img_index=self.current_img_index)
        return self.update_img_one(self.img_stack[0], img_index=0)


    def update_img_one(self, img=np.array([]), img_index=0):
        if len(img) == []: img = self.current_img
        self.current_img = img
        self.current_img_index = img_index

        self.im = self.axes.imshow(img*self.mask, cmap=self.colormap, vmin=self.cmin, vmax=self.cmax)

        self.axes.axis('on')
        self.axes.set_aspect('equal', 'box')
        if len(self.title) == self.img_stack.shape[0]:
            self.axes.set_title('current image: ' + self.title[img_index])
        else:
            self.axes.set_title('current image: ' + str(img_index))
        self.axes.title.set_fontsize(10)
        if self.colorbar_on_flag:
            self.add_colorbar()
            self.colorbar_on_flag = False
        self.add_line()
        self.draw()


    def add_line(self):
        if self.draw_line:
            if self.overlay_flag:
                self.axes.plot(self.x, self.y, '-', color=self.current_color, linewidth=1.0, label=self.plot_label)
            else:
                self.rm_colorbar()
                line, = self.axes.plot(self.x, self.y, '.-', color=self.current_color, linewidth=1.0, label=self.plot_label)
                if self.legend_flag:
                    self.axes.legend(handles=[line])
                self.axes.axis('on')
                self.axes.set_aspect('auto')
                self.draw()
        else:
            pass

    def draw_roi(self):
        self.cidpress = self.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.mpl_connect('button_release_event', self.on_release)


    def on_press(self, event):
        x1, y1 = event.xdata, event.ydata
        self.current_roi[0] = x1
        self.current_roi[1] = y1


    def on_release(self, event):
        x2, y2 = event.xdata, event.ydata
        self.current_roi[2] = x2
        self.current_roi[3] = y2
        self.roi_add_to_list()
        self.roi_display(self.current_roi)
        self.roi_disconnect()


    def roi_disconnect(self):
        self.mpl_disconnect(self.cidpress)
        self.mpl_disconnect(self.cidrelease)


    def roi_display(self, selected_roi):
        x1, y1 = selected_roi[0], selected_roi[1]
        x2, y2 = selected_roi[2], selected_roi[3]
        self.x = [x1, x2, x2, x1, x1]
        self.y = [y1, y1, y2, y2, y1]
        self.draw_line = True
        self.add_line()
        self.draw_line = False
        roi_name = '#' + str(self.roi_count - 1)
        self.axes.annotate(roi_name, xy=(x1, y1 - 40),
                           bbox={'facecolor': self.current_color, 'alpha': 0.5, 'pad': 2},
                           fontsize=10)
        self.draw()
        # self.roi_count += 1
        self.obj.tx_roi_x1.setText('{:4.1f}'.format(x1))
        self.obj.tx_roi_y1.setText('{:4.1f}'.format(y1))
        self.obj.tx_roi_x2.setText('{:4.1f}'.format(x2))
        self.obj.tx_roi_y2.setText('{:4.1f}'.format(y2))
        self.obj.pb_roi_draw.setEnabled(True)

        QApplication.processEvents()


    def roi_add_to_list(self):
        roi_name = 'roi_' + str(self.roi_count)
        self.roi_list[roi_name] = deepcopy(self.current_roi)
        self.current_color = self.color_list[self.roi_count % 10]
        self.roi_color[roi_name] = self.current_color
        self.roi_count += 1
        self.obj.update_roi_list(mode='add', item_name=roi_name)


    def set_contrast(self, cmin, cmax):
        self.cmax = cmax
        self.cmin = cmin
        self.colorbar_on_flag = True
        self.update_img_one(self.current_img*self.mask, self.current_img_index)


    def auto_contrast(self):
        img = self.current_img
        self.cmax = np.max(img)
        self.cmin = np.min(img)
        self.colorbar_on_flag = True
        self.update_img_one(self.current_img*self.mask, self.current_img_index)
        return self.cmin, self.cmax


    def rm_colorbar(self):
        try:
            self.cb.remove()
            self.draw()
        except:
            pass


    def add_colorbar(self):
        if self.colorbar_on_flag:
            try:
                self.cb.remove()
                self.draw()
            except:
                pass
            self.divider = make_axes_locatable(self.axes)
            self.cax = self.divider.append_axes('right', size='3%', pad=0.06)
            self.cb = self.fig.colorbar(self.im, cax=self.cax, orientation='vertical')
            self.cb.ax.tick_params(labelsize=10)
            self.draw()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    xanes = App()
    xanes.show()
    sys.exit(app.exec_())
