############################
#### IMPORTS
############################

from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtGui import *
from pyqtgraph import PlotWidget, plot
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene
import numpy as np
import sys
import os


############################
#### CONNECT MAIN WINDOW
############################

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        ############################
        #### LOAD UI FILE
        ############################

        uic.loadUi(r'GUI.ui', self)


        ############################
        #### BUTTON CONNECTIONS
        ############################



        ############################
        #### GLOBAL VARIABLES
        ############################



    ############################
    #### FUNCTION DEFINITIONS
    ############################


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
