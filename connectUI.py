############################
#### IMPORTS
############################

import os
import sys

import cv2
import numpy as np
import pyqtgraph as pg
import scipy.fftpack as fp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene


############################
#### CONNECT MAIN WINDOW
############################
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        ############################
        #### LOAD UI FILE
        ############################

        uic.loadUi(r'task1.ui', self)

        ############################
        #### BUTTON CONNECTIONS
        ############################
        self.cvimg = [[]]
        self.actionopen.triggered.connect(lambda: self.open())
        self.Equalize.clicked.connect(lambda: self.equalize(self.cvimg))
        self.comboBox.currentIndexChanged.connect(
            lambda: self.spatialFiltering())

        ############################
        #### GLOBAL VARIABLES
        ############################
        self.imageview = self.findChild(QLabel, "imageview")
        self.disply_width = 550
        self.display_height = 500

        self.histoCanvas = MplCanvas(self, width=5.5, height=4.5, dpi=90)
        self.histoLayout = QtWidgets.QVBoxLayout()
        self.histoLayout.addWidget(self.histoCanvas)

        self.fftCanvas = MplCanvas(self, width=5.5, height=4.5, dpi=90)
        self.fftLayout = QtWidgets.QVBoxLayout()
        self.fftLayout.addWidget(self.fftCanvas)

        self.graph = pg.PlotItem()
        pg.PlotItem.hideAxis(self.graph, 'left')
        pg.PlotItem.hideAxis(self.graph, 'bottom')

    ############################
    #### FUNCTION DEFINITIONS
    ############################
    def open(self):
        self.imagePath = QFileDialog.getOpenFileName(
            self, "Open File", "This PC",
            "All Files (*);;PNG Files(*.png);; Jpg Files(*.jpg)")
        self.cvimg = cv2.imread(self.imagePath[0])
        self.Imgorigin = self.cvimg.copy()
        #cv image to Qpixmap
        qt_img = self.convertCvQt(self.cvimg)
        # display it
        self.displayImage(qt_img)

    def convertCvQt(self, img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h,
                                            bytes_per_line,
                                            QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height,
                                        Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def displayImage(self, qtimg):
        self.imageview.clear()
        self.imageview.setPixmap(qtimg)

    def displayFFT(self):
        pass

    def spatialFiltering(self):
        imgFiltered = cv2.cvtColor(self.Imgorigin, cv2.COLOR_BGR2HSV)
        imgValues = imgFiltered[:, :, 2]
        filter = self.comboBox.currentText()

        if filter == 'Normal':
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
        imgqt = self.convertCvQt(self.cvimg)
        self.displayImage(imgqt)

    def toFFT(self, img):
        F1 = fp.fft2((img).astype(float))
        F2 = fp.fftshift(F1)
        fft = (20 * np.log10(0.1 + F2)).astype(int)
        return F1, F2, fft

    def highPassFiltering(self, img, F2):
        (w, h) = img.shape
        half_w, half_h = int(w / 2), int(h / 2)
        n = 50  # high pass size
        F2[half_w - n:half_w + n + 1, half_h - n:half_h + n +
           1] = 0  # select all but the first 50x50 (low) frequencies
        fft = (20 * np.log10(0.1 + F2)).astype(int)
        img = fp.ifft2(fp.ifftshift(F2)).real
        return img, fft, F2

    def lowPassFiltering(self, img, F2):
        (w, h) = img.shape
        half_w, half_h = int(w / 2), int(h / 2)
        n = 30  # low-pass size
        Fblank = np.zeros((w, h), np.uint8)
        # select the first 30x30 frequencies
        Fblank[half_w - n:half_w + n + 1, half_h - n:half_h + n + 1] = 1
        F2 = Fblank * F2
        fft = (20 * np.log10(0.1 + F2)).astype(int)
        img = fp.ifft2(fp.ifftshift(F2)).real
        return img, fft, F2

    def createHistoArray(self, img):
        Histo = np.zeros(shape=(256, 1))
        shape = img.shape
        for horizontal in range(shape[0]):
            for vertical in range(shape[1]):
                temp = img[horizontal, vertical]
                Histo[temp, 0] = Histo[temp, 0] + 1
        return Histo

    #Reditributes the grayscale to be applied on images later on
    #Ex: if we have an Input = [0,1,2,3,4,5,6,7] , Output = [0,1,2,3,3,3,4,5]

    def redistributeGrayScale(self, histoArray, img):
        cumu = np.array([])
        cumu = np.append(cumu, img[0, 0])
        shape = img.shape
        for i in range(255):
            temp = histoArray[0, i + 1] + cumu[i]
            cumu = np.append(cumu, temp)
        max = np.amax(img)
        cumu = np.round((cumu / (shape[0] * shape[1])) * max)
        return cumu

    #Maps new values for the pixels of the image according to the new distribution of GrayScale generated earlier

    def mapPixels(self, cumu, original):
        img = np.full_like(original, 0)
        shape = original.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                temp = original[i, j]
                img[i, j] = cumu[temp]
        return img

    #Call this function to equalize your histogram

    def showHistogram(self, data):
        no_of_bins = np.arange(256)
        data_rav = data.ravel()  #spreads image pixels into one dimension
        # histogram = plt.hist(data_rav, bins=no_of_bins)
        self.histoCanvas.axes.hist(data_rav, bins=no_of_bins)
        self.histoCanvas.draw()
        self.histoWidget.setCentralItem(self.graph)
        self.histoWidget.setLayout(self.histoLayout)

    def equalize(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        histoArray = self.createHistoArray(img_gray)
        histoArray = np.transpose(histoArray)
        redistributedGrayScale = self.redistributeGrayScale(
            histoArray, img_gray)
        equalized_img = self.mapPixels(redistributedGrayScale, img_gray)
        self.showHistogram(equalized_img)
        img = self.convertCvQt(equalized_img)
        self.displayImage(img)


############################
#### CALL MAIN FUNCTION
############################


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
