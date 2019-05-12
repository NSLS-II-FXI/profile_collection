import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import cm, inch
from PyPDF2 import PdfFileMerger, PdfFileReader
import shutil
import os
import skimage.io
import glob
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap
from datetime import datetime

#from reportlab.platypus import Paragraph, Frame, Image


global PDF_ARGS
PDF_ARGS = {}



def reset_pdf():
    _,_, PDF_ARGS['year'], PDF_ARGS['month'], PDF_ARGS['day'] = get_current_date()
    PDF_ARGS['fn_log'] = f'/NSLS2/xf18id1/DATA/FXI_log/TXM_log_test_{PDF_ARGS["year"]}{PDF_ARGS["month"]}{PDF_ARGS["day"]}.pdf'
    PDF_ARGS['temp_folder'] = '/home/xf18id/.ipython/profile_collection/startup/temp'
    PDF_ARGS['temp_img_folder'] = PDF_ARGS['temp_folder'] + '/img'
    PDF_ARGS['fn_tmp'] = PDF_ARGS['temp_folder'] + '/tmp.pdf'
    PDF_ARGS['fn_tmp_txt'] = PDF_ARGS['temp_folder'] + '/current_log.txt'
    PDF_ARGS['C_canvas'] = canvas.Canvas(PDF_ARGS['fn_tmp'], pagesize=letter, bottomup = 1)
    PDF_ARGS['num_of_letter'] = 100
    PDF_ARGS['Width'], PDF_ARGS['Height'] = letter     
    PDF_ARGS['Font_size'] = 9
    PDF_ARGS['LineSpace'] = 1.2 * PDF_ARGS['Font_size']
    PDF_ARGS['Cursor_x'] = 0.45 * inch
    PDF_ARGS['Cursor_y'] = PDF_ARGS['Height'] - 0.5*inch
    insert_time_title()
    PDF_ARGS['Cursor_y'] = PDF_ARGS['Height'] - inch
    if not os.path.exists(PDF_ARGS['temp_img_folder']):
        os.makedirs(PDF_ARGS['temp_img_folder'], exist_ok=True)





def clean_tmp_fig():
    reset_pdf()
    files = glob.glob(PDF_ARGS['temp_img_folder'] + '/*.png')
    if len(files):
        for f in files:
            os.remove(f)
    #PDF_ARGS['tmp_flag'] = 0



def check_page_is_full():
    if PDF_ARGS['Cursor_y'] <= inch:         # current canvas is full, need to save it to pdf and create new canvas
        PDF_ARGS['C_canvas'].showPage()
        PDF_ARGS['C_canvas'].save()
        merge_pdf(PDF_ARGS['fn_log'], PDF_ARGS['fn_tmp'], PDF_ARGS['fn_log'])
        PDF_ARGS['f'] = open(PDF_ARGS['fn_tmp_txt'], 'w') # write current text to current_log.txt 
        reset_pdf()
#        PDF_ARGS['Cursor_y'] = PDF_ARGS['Height'] - inch
        return 1
    else:
        PDF_ARGS['f'] = open(PDF_ARGS['fn_tmp_txt'], 'a+')
        return 0


def get_current_date():
    now = datetime.now()
    year = np.str(now.year)
    mon  = '{:02d}'.format(now.month)
    day  = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minu = '{:02d}'.format(now.minute)
    sec  = '{:02d}'.format(now.second) 
    current_date = year + '-' + mon + '-' + day
    current_time = hour + ':' + minu + ':' + sec
    return current_date, current_time, year, mon, day


def insert_time_title():
    PDF_ARGS['Cursor_x'] = 0.45 * inch
    PDF_ARGS['Cursor_y'] = PDF_ARGS['Height'] - 0.5*inch
    current_date, current_time, _, _, _ = get_current_date()
    text = 'FXI: ' + current_date + '  ' + current_time
    PDF_ARGS['C_canvas'].setFont('Courier', PDF_ARGS['Font_size'] - 1.5)
    textObj = PDF_ARGS['C_canvas'].beginText(PDF_ARGS['Cursor_x'], PDF_ARGS['Cursor_y'])
    textObj.textLines(text)
    PDF_ARGS['C_canvas'].drawText(textObj)   

    
def obtain_image_file_name():
    current_date, current_time, _, _, _ = get_current_date()
    current_date = ''.join(current_date.split('-'))
    current_time = ''.join(current_time.split(':'))
    fn = f'{PDF_ARGS["temp_img_folder"]}/{current_date}_{current_time}.png'
    return fn


def insert_text(text, write_txt_flag=1):
    check_page_is_full()    
    if write_txt_flag:
        PDF_ARGS['f'].write(text+'\n')
        PDF_ARGS['f'].close()
    PDF_ARGS['Cursor_x'] = 0.6 * inch
    PDF_ARGS['C_canvas'].setFont('Courier', PDF_ARGS['Font_size'])
    num_of_letter = PDF_ARGS['num_of_letter']
    num_of_line = 1
    text_split = split_str(text, num_of_letter)
    num_of_line = len(text_split.split('\n'))
#    if len(text) > num_of_letter:
#        text_split = text[:num_of_letter] + '\n' + text[num_of_letter:]
#        num_of_line += 1
    textObj = PDF_ARGS['C_canvas'].beginText(PDF_ARGS['Cursor_x'], PDF_ARGS['Cursor_y'])
    textObj.textLines(text_split)
    PDF_ARGS['C_canvas'].drawText(textObj)
    PDF_ARGS['Cursor_y'] -= num_of_line * PDF_ARGS['Font_size'] + PDF_ARGS['LineSpace']



def insert_fig(ratio=0.4):
    try:
        fig = plt.gcf()
    #    fig.set_figheight(2) # set figure height to 2 inch
        fn = obtain_image_file_name()
       # fn =  PDF_ARGS['temp_folder'] + f'/tmp_fig_{PDF_ARGS["tmp_flag"]}.png'
#        fn = f'/home/xf18id/.ipython/profile_collection/startup/temp/tmp_fig_{PDF_ARGS["tmp_flag"]}.png'
        fig.savefig(fn,format='png')
        insert_pic(fn, ratio)
#        PDF_ARGS['tmp_flag'] += 1 
    except:
        print('cannot insert figure')

    

def insert_pic(fn='', ratio=0.3):
    try:
        img = skimage.io.imread(fn)
        PDF_ARGS['Cursor_x'] = 0.6 * inch;
        img_height, img_width = img.shape[0], img.shape[1]
        scale_w = float(img_width) / PDF_ARGS['Width']
        scale_h = float(img_height) / PDF_ARGS['Height']
        scale = max(scale_w, scale_h) / ratio
        img_width /= scale
        img_height /= scale
        PDF_ARGS['Cursor_x'] = (PDF_ARGS['Width'] - img_width)/2
        PDF_ARGS['Cursor_y'] -= img_height - PDF_ARGS['LineSpace']
        if check_page_is_full():
            PDF_ARGS['Cursor_y'] -= img_height - PDF_ARGS['LineSpace']
            PDF_ARGS['Cursor_x'] = (PDF_ARGS['Width'] - img_width)/2
        
        PDF_ARGS['C_canvas'].drawImage(fn, PDF_ARGS['Cursor_x'], PDF_ARGS['Cursor_y'], width=img_width, height=img_height)
        PDF_ARGS['Cursor_x'] = 0.6 * inch        
        PDF_ARGS['Cursor_y'] -= inch/2 
        insert_text(fn)    
        #PDF_ARGS['tmp_flag'] += 1 
   
    except:
        print('picture file not found') 



def insert_log(comment=''):
    export_pdf(1)
    line = wh_pos(comment, 0)
    current_date, current_time, _,_,_ = get_current_date()
    line[0] = f'FXI log    {current_date} {current_time}\n'
    txt=''
    for i in range(len(line)):
        txt += line[i]
        txt += '\n'
    PDF_ARGS['Font_size'] = 5.5
    PDF_ARGS['num_of_letter'] = 170
    insert_text(txt, 0)
    export_pdf(1)
    reset_pdf()
    

def insert_screen_shot(ratio=0.6):
    fn = obtain_image_file_name()
    cmd = f'import {fn}'
    os.system(cmd)
    insert_pic(fn, ratio)
#    if flag==2: # current monitor screen
#        img = QApplication.primaryScreen().grabWindow(0)
#    elif flag==3: # the whole desk screen
#        a = QApplication.desktop()
#        img = QApplication.primaryScreen().grabWindow(a.winId())
#    else:
#        pass    
#    fn = PDF_ARGS['temp_folder'] + '/screenshot.png'
#    img.save(fn, 'png')

###############################

def export_pdf(merge_flag=0):
    if len(PDF_ARGS) == 0:
        reset_pdf()
    try:
        PDF_ARGS['C_canvas'].showPage()
        PDF_ARGS['C_canvas'].save()
        if merge_flag:
            merge_pdf(PDF_ARGS['fn_log'], PDF_ARGS['fn_tmp'], PDF_ARGS['fn_log'])
        reset_pdf()
    except:
        pass


def merge_log():
    '''
    merge "current_log.txt" to "/NSLS2/xf18id1/DATA/FXI_log/TXM_log_{year}{month}{day}.pdf"
    '''
    export_pdf()
    reset_pdf()
    num = 0
    try:
        with open(PDF_ARGS['fn_tmp_txt'], 'r') as fp:
            tmp_txt = ''
            for line in fp:
                if line[0] == '\n':
                    print(f'tmp_txt={tmp_txt}')
                    insert_text(tmp_txt, 0)
                    tmp_txt = ''
                else:
                    tmp_txt += line         
                    num += 1
        if num:
            insert_text(tmp_txt, 0)
            export_pdf()
            try:
                merge_pdf(PDF_ARGS['fn_log'], PDF_ARGS['fn_tmp'], PDF_ARGS['fn_log'])
                print('merged log file ("current_log.txt") into TXM main log')
                os.remove(PDF_ARGS['fn_tmp_txt'])
                reset_pdf()
            except:
                print('merge fails... need manual check...')
    except:
        pass


def merge_pdf(fn1, fn2, fout):
    '''
    merge pdf file fn1 and fn2 to fout
    if fn1 does not exist, it will copy fn2 to fout
    '''
    if os.path.exists(fn1): 
        merger = PdfFileMerger()  
        flag1 = 0
        flag2 = 0
        try:
            f1 = PdfFileReader(fn1, 'rb')
            flag1 = 1
        except:
            try:
                os.remove(fn1)
            except:
                pass
            finally:
                print(f'file: crashed.\nGenerate new file "{fout}"')
                flag1 = 0        
        try:
            f2 = PdfFileReader(fn2,'rb')
            flag2 = 1
        except:
            try:
                os.remove(fn2)
            except:
                pass
            finally:
                print(f'file: crashed.\nGenerate new file "{fout}"')
            flag2 = 0

        if flag1:     
            merger.append(f1)
        if flag2:           
            merger.append(f2)                 
        try:        
            merger.write(fout)
        except:
            print('file merging fails, skip merging...')
    else:
        shutil.copy2(fn2, fout)
            
    
def split_str(text, max_len=70):
    text_split = text.split('\n')
    t = ''
    for i in range(len(text_split)):
        tmp = text_split[i]
        while (len(tmp)>max_len):
            t += tmp[:max_len]
            t += '\n'
            tmp = tmp[max_len:]
        t += tmp
        t += '\n'    
    return t


def demo_pdf():
    reset_pdf()
#    PDF_ARGS['C_canvas'].saveState()
    insert_text('fasdfadsfijhfp;sadiofhwaeiuhfg;ailksdmfcoiawehjfpioqawnfvcaikewmFASDFADSFADSFADSFADSF')
    x = np.linspace(-3,3,100)
    y = np.cos(x)
    plt.figure()
    plt.plot(x,y,'r.-')
    insert_fig()
    export_pdf()


def run_pdf():
    merge_log()
#    clean_tmp_fig()

























