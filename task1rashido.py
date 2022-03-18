import numpy as np
import logging 
from playsound import playsound

from PyQt5 import QtCore, QtGui, QtWidgets,uic
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QColorDialog, QFileDialog, QFrame, QWidget, QInputDialog, QLineEdit,QComboBox , QMainWindow
import os
import numpy as np
from PyQt5.QtWidgets import QMessageBox
import sys 
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QColorDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QColor ,QKeySequence
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import io
from numpy.fft import fft, fftfreq, ifft
from scipy.fftpack import fft, ifft, rfft
from scipy.fftpack.basic import irfft
from scipy.fftpack.helper import rfftfreq
from scipy import signal
import cmath
from scipy.io.wavfile import write
from pyqtgraph import PlotWidget
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.exporters
from fpdf import FPDF
import statistics
from pyqtgraph import PlotWidget
import pyqtgraph
from pyqtgraph import *
import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotItem
#from matplotlib.pyplot import draw
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QColorDialog, QFileDialog, QFrame, QWidget, QInputDialog, QLineEdit,QComboBox , QLabel
import os
import numpy as np
from PyQt5.QtWidgets import QMessageBox
import sys 
from PyQt5.QtGui import QColor , QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QColorDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyqtgraph.graphicsItems.ScatterPlotItem import Symbols
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from matplotlib.figure import Figure
import io
from numpy.fft import fft, fftfreq, ifft
import scipy.fftpack  as fp
from scipy import signal
import cmath
import cv2
from scipy.io import wavfile
import scipy.io

from skimage.io import imread    # pip install scikit-image


############################
#### CONNECT MAIN WINDOW
############################

class UI(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(UI, self).__init__(*args, **kwargs)

        ############################
        #### LOAD UI FILE
        ############################

        uic.loadUi(r'F:/SBME 3/semester 2/CV/task1/task1.ui', self)


        ############################
        #### BUTTON CONNECTIONS
        ############################
        self.imageview = self.findChild(QLabel , "imageview")
        self.actionopen.triggered.connect(lambda : self.open())
        self.comboBox.currentIndexChanged.connect(lambda : self.spatialFiltering())

        ############################
        #### GLOBAL VARIABLES
        ############################
        self.disply_width = 350
        self.display_height = 350
    


    ############################
    #### FUNCTION DEFINITIONS
    ############################
    def open(self): 
        self.imagePath = QFileDialog.getOpenFileName(self,"Open File" ,"This PC" , "All Files (*);;PNG Files(*.png);; Jpg Files(*.jpg)")
        self.cvimg = cv2.imread(self.imagePath[0])
        self.Imgorigin = self.cvimg.copy()
        #cv image to Qpixmap
        qt_img = self.convertCvQt()
        # display it
        self.displayImage(qt_img)
    
    def convertCvQt(self):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(self.cvimg, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)  
    
    def displayImage(self , qtimg  ):
        self.imageview.clear()
        self.imageview.setPixmap(qtimg)
        
    def spatialFiltering(self):
        imgFiltered = cv2.cvtColor(self.Imgorigin , cv2.COLOR_BGR2HSV)
        imgValues = imgFiltered[:, :, 2]
        filter = self.comboBox.currentText()
        
        if filter =='Normal':
            pass
        
        if filter == 'Low pass filter':
            F1, F2, fftOG = self.toFFT(imgValues)
            imgValues, fftFiltered, F2 = self.lowPassFiltering(imgValues, F2)


        elif filter == 'Median filter':
            # median acts as a=low-pass filter ---- blurring effect
            # src : source file ---- ksize: int kernel size
            imgValues = cv2.medianBlur(imgValues, 21)
            _, _, fftFiltered = self.toFFT(imgValues)

        elif filter == 'High pass filter':
            # # kernel = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
            # # imgValues = ndimage.convolve(imgValues, kernel)
            # imgValues = cv2.boxFilter(imgValues, -1, (15, 15))
            # imgValues = imgFiltered[:, :, 2] - imgValues
            F1, F2, fftOG = self.toFFT(imgValues)
            imgValues, fftHP, F2 = self.highPassFiltering(imgValues, F2)


        elif filter == 'Laplacian':
            # laplacian acts as hig-pass filter ---- edge detector
            # src : source file ---- ddepth : depth of output image ---- ksize : blurring kernel size
            imgValues = cv2.Laplacian(imgValues, -1, (11, 11))
            _, _, fftFiltered = self.toFFT(imgValues)

        imgFiltered[:, :, 2] = imgValues
        imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_HSV2BGR)
        self.cvimg = imgFiltered
        imgqt = self.convertCvQt()
        print("gg")
        self.displayImage(imgqt)

    def toFFT(self,img):
        F1 = fp.fft2((img).astype(float))
        F2 = fp.fftshift(F1)
        fft = (20*np.log10(0.1 + F2)).astype(int)
        return F1, F2, fft

    def highPassFiltering(self,img, F2):
        (w, h) = img.shape
        half_w, half_h = int(w/2), int(h/2)
        n = 50    # high pass size
        F2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0    # select all but the first 50x50 (low) frequencies
        fft = (20*np.log10(0.1 + F2)).astype(int)
        img = fp.ifft2(fp.ifftshift(F2)).real
        return img, fft, F2  

    def lowPassFiltering(self,img, F2):
        (w, h) = img.shape
        half_w, half_h = int(w/2), int(h/2)
        n = 30    # low-pass size
        Fblank = np.zeros((w, h), np.uint8)
        # select the first 30x30 frequencies
        Fblank[half_w-n:half_w+n+1, half_h-n:half_h+n+1] = 1
        F2 = Fblank*F2
        fft = (20*np.log10(0.1 + F2)).astype(int)
        img = fp.ifft2(fp.ifftshift(F2)).real
        return img, fft, F2
      
############################
#### CALL MAIN FUNCTION
############################

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = UI()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


    