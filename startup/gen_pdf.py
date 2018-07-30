import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import cm, inch
from PyPDF2 import PdfFileMerger, PdfFileReader
#import shutil
import os
import skimage.io
from datetime import datetime
#from reportlab.platypus import Paragraph, Frame, Image

Cursor_y=None
Width=None 
Height=None
LineSpace=None
Font_size=None
C_canvas=None



def reset():
    global C_canvas, Cursor_x, Cursor_y, Width, Height, Linspace, Font_size
    Width, Height = letter 
    Cursor_y = Height - inch
    Font_size = 12
    LineSpace = 2 * Font_size
    C_canvas = canvas.Canvas('tmp.pdf',  pagesize=letter, bottomup = 1)
    Width, Height = letter     
    Font_size = 12
    LineSpace = 2 * Font_size
    insert_time_title()
    Cursor_y = Height - inch


def insert_time_title():
    global Width, Height, Cursor_y, LineSpace, Font_size 
    global C_canvas
    Cursor_x = 0.5 * inch
    Cursor_y = Height - 0.5*inch
    now = datetime.now()
    year = np.str(now.year)
    mon  = '{:02d}'.format(now.month)
    day  = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minu = '{:02d}'.format(now.minute)
    current_date = year + '-' + mon + '-' + day
    current_time = hour + ':' + minu
    text = 'FXI: ' + current_date + '  ' + current_time
    C_canvas.setFont('Courier', Font_size-2)
    textObj = C_canvas.beginText(Cursor_x, Cursor_y)
    textObj.textLines(text)
    C_canvas.drawText(textObj)   
    


def insert_text(text):
    global Width, Height, Cursor_y, LineSpace, Font_size 
    global C_canvas
    check_page_is_full()
    Cursor_x = 0.75*inch
    C_canvas.setFont('Courier', Font_size)
    num_of_letter = 70
    num_of_line = 1
    text_split = deepcopy(text)
    if len(text) > num_of_letter:
        text_split = text[:num_of_letter] + '\n' + text[num_of_letter:]
        num_of_line += 1
    textObj = C_canvas.beginText(Cursor_x, Cursor_y)
    textObj.textLines(text_split)
    C_canvas.drawText(textObj)
    Cursor_y -= num_of_line * Font_size + LineSpace


def check_page_is_full():
    global Cursor_y, C_canvas
    if Cursor_y <= inch:
        C_canvas.showPage()
        C_canvas.save()
        
        if os.path.exists('txm_log.pdf'): 
            merger = PdfFileMerger()  
            f1 = PdfFileReader('txm_log.pdf', 'rb')
            f2 = PdfFileReader('tmp.pdf','rb')
            merger.append(f1)
            merger.append(f2)
            merger.write('txm_log.pdf')
        else:
            os.rename('tmp.pdf', 'txm_log.pdf')
        reset()
        Cursor_y = Height - inch

    

def insert_fig(im_size='norm'):
    global Width, Height, Cursor_y, LineSpace, Font_size 
    global C_canvas
    try:
        if im_size == 'norm':
            ratio = 0.4
        elif im_size == 'large':
            ratio = 0.6
        else:
            ratio = 0.4
        fig = plt.gcf()
    #    fig.set_figheight(2) # set figure height to 2 inch
        fig.savefig('/tmp/tmp_fig.png',format='png')
        check_page_is_full()
        img = io.imread('/tmp/tmp_fig.png')
        img_height, img_width = img.shape[0], img.shape[1]
        scale_w = float(img_width) / Width
        scale_h = float(img_height) / Height
        scale = max(scale_w, scale_h) / ratio
        img_width /= scale
        img_height /= scale
        Cursor_x = (Width - img_width)/2
        Cursor_y -= img_height-LineSpace
    #    C_canvas.scale(1/scale, 1/scale)
        C_canvas.drawImage('/tmp/tmp_fig.png', Cursor_x, Cursor_y, width=img_width, height=img_height)
    #    C_canvas.restoreState()
    #    C_canvas.saveState()
        Cursor_x = 0.75 * inch
        Cursor_y -= inch/2
    except:
        print('cannot insert figure')


    

def insert_pic(fn='', im_size='norm'):
    global Width, Height, Cursor_y, LineSpace, Font_size 
    global C_canvas

    try:
        if im_size == 'norm':
            ratio = 0.3
        elif im_size == 'large':
            ratio = 0.5
        img = io.imread(fn)
        Cursor_x = 0.75*inch;
        img_height, img_width = img.shape[0], img.shape[1]
        scale_w = float(img_width) / Width
        scale_h = float(img_height) / Height
        scale = max(scale_w, scale_h) / ratio
        img_width /= scale
        img_height /= scale
        Cursor_x = (Width - img_width)/2
        Cursor_y -= img_height-LineSpace
        C_canvas.drawImage(fn, Cursor_x, Cursor_y, width=img_width, height=img_height)
        Cursor_x = 0.75 * inch        
        Cursor_y -= inch/2        
    except:
        print('picture file not found')        



   

def demo():
    global C_canvas
    reset()
    C_canvas.saveState()
    insert_text('fasdfadsfijhfp;sadiofhwaeiuhfg;ailksdmfcoiawehjfpioqawnfvcaikewmFASDFADSFADSFADSFADSF')
    x = np.linspace(-3,3,100)
    y = np.cos(x)
#    plt.figure()
#    plt.plot(x,y,'r.-')
#    insert_fig()
#    plt.close('all')
    img = np.random.random([500,600])
    plt.figure()
    plt.imshow(img)
    insert_fig()
#    insert_pic('Ce.tif','large')
    insert_text('see where am I')
 #   insert_fig()
    insert_text('where it is')
    C_canvas.showPage()
    C_canvas.save()



