import os

import cv2
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import uic
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time
import numpy as np
import qdarkstyle
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QFileDialog, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib import pyplot as plt
import math as m
# from skimage.color import rgb2gray
# from skimage.io import imread
import cv2
import numpy as np
from scipy import fftpack
from scipy.spatial.transform import Rotation as R
# import numpy as np
from collections import Counter
import cmath
import threading

from matplotlib.figure import Figure
from scipy.stats import norm


class Phantom(qtw.QWidget):
    def __init__(self):
        super().__init__()

        uic.loadUi("src/ui/Phantom.ui", self)
        # self.splitter.setWidth(50, 1)

        self.df = None
        self.df_custom = None
        self.Running_K_Space = 0
        self.Reload_K_Space = 0

        self.figure_sequence = Figure(dpi=80)
        self.figure_sequence_custom = Figure(dpi=80)

        self.phantom_layout()

        self.canvas_sequence, self.axis_sequence_RF, self.axis_sequence_SS, self.axis_sequence_PG, self.axis_sequence_FG, self.axis_sequence_RO = self.sequence_layout(
            self.figure_sequence, self.verticalLayout_4)

        self.canvas_sequence_custom, self.axis_sequence_custom_RF, self.axis_sequence_custom_SS, self.axis_sequence_custom_PG, self.axis_sequence_custom_FG, self.axis_sequence_custom_RO = self.sequence_layout(
            self.figure_sequence_custom, self.verticalLayout_6)

        self.axes_sequence = [self.axis_sequence_RF, self.axis_sequence_SS, self.axis_sequence_PG, self.axis_sequence_FG,
                              self.axis_sequence_RO]
        self.axes_sequence_custom = [self.axis_sequence_custom_RF, self.axis_sequence_custom_SS,
                                     self.axis_sequence_custom_PG, self.axis_sequence_custom_FG,
                                     self.axis_sequence_custom_RO]

        self.pushButton_openSequence.clicked.connect(lambda: self.sequence_read())
        self.pushButton_apply.clicked.connect(lambda: self.custom_sequence())
        self.pushButton_clear.clicked.connect(lambda: self.clear_all())
        self.pushButton_openPhantom.clicked.connect(lambda: self.phantom_read())
        self.comboBox_kspace_size.currentIndexChanged.connect(lambda: self.start_threading())

        # self.sequence_custom_layout()

    def sequence_layout(self, figure, layout):
        ######################## Sequence Layout #########################
        axis_sequence_RF = figure.add_subplot(5, 1, 1)
        axis_sequence_RF.set_ylabel("RF")

        axis_sequence_SS = figure.add_subplot(5, 1, 2, sharex=axis_sequence_RF)
        axis_sequence_SS.set_ylabel("SS")

        axis_sequence_PG = figure.add_subplot(5, 1, 3, sharex=axis_sequence_RF)
        axis_sequence_PG.set_ylabel("PG")

        axis_sequence_FG = figure.add_subplot(5, 1, 4, sharex=axis_sequence_RF)
        axis_sequence_FG.set_ylabel("FG")

        axis_sequence_RO = figure.add_subplot(5, 1, 5, sharex=axis_sequence_RF)
        axis_sequence_RO.set_ylabel("RO")
        axis_sequence_RO.set_frame_on(False)

        canvas_sequence = FigureCanvas(figure)
        figure.subplots_adjust(hspace=0.5)

        # self.toolbar = NavigationToolbar(self.canvas_sequence, self)
        # self.canvas_sequence.figure.tight_layout()

        layout.addWidget(canvas_sequence)

        axes_sequence = [axis_sequence_RF, axis_sequence_SS, axis_sequence_PG, axis_sequence_FG]
        for axis in axes_sequence:  ## removing axes from the figure
            axis.set_frame_on(False)
            axis.axes.get_xaxis().set_visible(False)

        return canvas_sequence, axis_sequence_RF, axis_sequence_SS, axis_sequence_PG, axis_sequence_FG, axis_sequence_RO

    def sequence_read(self):
        sequence_path = QFileDialog.getOpenFileName(
            self, "Open File", "sequence", filter="Json files (*.json)")[0]
        if sequence_path != "":
            dictionary = json.load(open(sequence_path))
            self.df = pd.DataFrame(dictionary)
            self.plotting_sequence(
                self.axes_sequence, self.canvas_sequence, self.df)

    def plotting_sequence(self, axes, canvas, dataFrame):
        colors = ['b', 'g', 'blueviolet', 'orange', 'red']
        for color, axis in enumerate(axes):
            axis.clear()
            axis.plot(np.linspace(0, dataFrame['RO'].Pos + dataFrame['RO'].Duration, 2), np.zeros(shape=2),
                      color=colors[color],
                      linewidth=0.7)

        RF_Duration = np.linspace(-dataFrame['RF'].Duration / 2,
                                  dataFrame['RF'].Duration / 2, 100)
        axes[0].plot(RF_Duration + dataFrame['RF'].Duration / 2,
                     dataFrame['RF'].Amp * np.sinc(2 * RF_Duration),
                     color=colors[0])
        axes[0].set_ylabel("RF")

        #
        SS_Duration = np.linspace(0, dataFrame['SS'].Duration, 100)
        SS_step = dataFrame['RF'].Amp * \
                  np.sinc(RF_Duration) > dataFrame['RF'].Amp / 4
        axes[1].plot(SS_Duration, (dataFrame['SS'].Amp * SS_step),
                     color=colors[1])
        axes[1].set_ylabel("SS")

        #
        PG_Duration = np.linspace(0, dataFrame['PG'].Duration, 100)
        for i in range(-dataFrame['PG'].Amp, dataFrame['PG'].Amp, 1):
            axes[2].plot(PG_Duration + dataFrame['PG'].Pos,
                         (i * SS_step), color=colors[2])
        axes[2].set_ylabel("PG")

        #
        FG_Duration = np.linspace(0, dataFrame['FG'].Duration, 100)
        axes[3].plot(FG_Duration + dataFrame['FG'].Pos,
                     (dataFrame['FG'].Amp * SS_step), color=colors[3])
        axes[3].set_ylabel("FG")

        #
        RO_Duration = np.linspace(0, dataFrame['RO'].Duration, 100)
        axes[4].plot(RO_Duration + dataFrame['RO'].Pos,
                     (dataFrame['RO'].Amp * SS_step), color=colors[4])
        axes[4].set_ylabel("RO")

        # Setting Y limits of custom sequenec to be the same as loaded sequence
        if axes is self.axes_sequence_custom:
            for axis, axis_custom in zip(self.axes_sequence, self.axes_sequence_custom):
                ymin, ymax = axis.get_ylim()
                axis_custom.set_ylim(ymin, ymax)
        #
        canvas.draw()

    def custom_sequence(self):
        print("Entered")
        if (self.df is not None):
            self.df_custom = self.df.copy()
            print(self.df_custom)
            self.df_custom['RF'].Amp = self.spinBox_RF.value()
            print(self.df_custom)

            self.plotting_sequence(self.axes_sequence_custom, self.canvas_sequence_custom, self.df_custom)

    def clear_all(self):
        for axis, axis_custom in zip(self.axes_sequence, self.axes_sequence_custom):
            axis.clear()
            axis_custom.clear()
        self.canvas_sequence.draw()
        self.canvas_sequence_custom.draw()

    def phantom_layout(self):

        ################ Phantom Original Layout #######################

        self.figure_Orig_Fourier = Figure(figsize=(20, 20), dpi=100)
        self.axis_Orig_Fourier = self.figure_Orig_Fourier.add_subplot()
        self.canvas_Orig_Fourier = FigureCanvas(self.figure_Orig_Fourier)
        self.figure_Orig_Fourier.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.axis_Orig_Fourier.set_facecolor('black')
        self.gridLayout.addWidget(self.canvas_Orig_Fourier)

        self.figure_Orig_Spat = Figure(figsize=(20, 20), dpi=100)
        self.axis_Orig_Spat = self.figure_Orig_Spat.add_subplot()
        self.canvas_Orig_Spat = FigureCanvas(self.figure_Orig_Spat)
        self.figure_Orig_Spat.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.axis_Orig_Spat.set_facecolor('black')
        self.gridLayout.addWidget(self.canvas_Orig_Spat)

        self.figure_kspace = Figure(figsize=(20, 20), dpi=100)
        self.axis_kspace = self.figure_kspace.add_subplot()
        self.canvas_kspace = FigureCanvas(self.figure_kspace)
        self.figure_kspace.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.axis_kspace.set_facecolor('black')
        self.gridLayout_2.addWidget(self.canvas_kspace)

        self.figure_reconstruct = Figure(figsize=(20, 20), dpi=100)
        self.axis_reconstruct = self.figure_reconstruct.add_subplot()
        self.canvas_reconstruct = FigureCanvas(self.figure_reconstruct)
        self.figure_reconstruct.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.axis_reconstruct.set_facecolor('black')
        self.gridLayout_2.addWidget(self.canvas_reconstruct)

        self.axes_phantom = [self.axis_Orig_Spat, self.axis_Orig_Fourier,
                             self.axis_kspace, self.axis_reconstruct]
        for axis in self.axes_phantom:  # removing axes from the figure
            axis.set_xticks([])
            axis.set_yticks([])
    
    def phantom_read(self):

        phantom_path = QFileDialog.getOpenFileName(self, "Open File", "src/docs/phantom images", filter="Images files ("
                                                                                                        "*.jpg *.jpeg "
                                                                                                       "*.png)")[0]
        if phantom_path != "":
            # print("running kspace",self.Running_K_Space)
            if self.Running_K_Space == 1:
                self.Reload_K_Space = 1
            else:
                self.Reload_K_Space = 0
            self.img = cv2.imread(phantom_path, cv2.IMREAD_GRAYSCALE)
            w, h = int(self.figure_Orig_Spat.get_figwidth() * self.figure_Orig_Spat.dpi), int(
                self.figure_Orig_Spat.get_figheight() * self.figure_Orig_Spat.dpi)
            self.img = cv2.resize(self.img, (w, h), interpolation=cv2.INTER_AREA)
            self.axis_Orig_Spat.imshow(self.img, cmap='gray')
            self.canvas_Orig_Spat.draw()

            # Compute the Fourier Transform of the image
            f = np.fft.fft2(self.img)

            # Shift the zero-frequency component to the center of the spectrum
            fshift = np.fft.fftshift(f)

            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            # Compute the magnitude spectrum of the Fourier Transform
            self.axis_Orig_Fourier.imshow(magnitude_spectrum, cmap='gray')
            self.canvas_Orig_Fourier.draw()


            self.start_threading()

    def start_threading(self):
        # Generate_kspace
        StreamThread = threading.Thread(target=self.generate_kspace)
        # StreamThread.daemon = True
        StreamThread.start()

            
            

    def generate_kspace(self):

        self.axis_kspace.clear()
        self.canvas_kspace.draw()

        IMG = cv2.resize(self.img,
                         (int(self.comboBox_kspace_size.currentText()), int(self.comboBox_kspace_size.currentText())))

        IMG_vector = np.zeros((IMG.shape[0], IMG.shape[1], 3), dtype=np.float_)
        IMG_K_Space = np.zeros((IMG.shape[0], IMG.shape[1]), dtype=np.complex_)
        X_Rotation = self.Rx(np.radians(90)) * self.Ry(0) * self.Rz(0)

        self.axis_kspace.imshow(abs((IMG_K_Space)), cmap='gray')


        for Krow in range(IMG.shape[0]):
            self.Running_K_Space = 1
            # print("reload Kspace",self.Reload_K_Space)
            if self.Reload_K_Space == 1:
                Krow = 0
                IMG_K_Space[:,:] = 0
                self.Reload_K_Space = 0
                return
            print(Krow)
            Gy_Phase = ((2 * np.pi) / IMG.shape[0]) * Krow
            IMG_vector[:, :, :] = 0
            # construct our vectors
            for i in range(0, IMG.shape[0]):
                for j in range(0, IMG.shape[1]):
                    IMG_vector[i][j][2] = IMG[i][j]

            # simulate RF
            for i in range(0, IMG.shape[0]):
                for j in range(0, IMG.shape[1]):
                    IMG_vector[i][j] = IMG_vector[i][j] * X_Rotation

            # simulate Gy
            for i in range(0, IMG.shape[0]):
                Z_Rotation = self.Rx(0) * self.Ry(0) * self.Rz((Gy_Phase / IMG.shape[0]) + ((Gy_Phase / IMG.shape[0]) * i))
                for j in range(0, IMG.shape[1]):
                    IMG_vector[i][j] = IMG_vector[i][j] * Z_Rotation

            # simulate Gx
            for Kcol in range(0, IMG.shape[1]):
                # stepi = 2*np.pi/(IMG.shape[0])*(Krow)
                Gx_phase = 2 * np.pi / (IMG.shape[0]) * (Kcol)
                for i in range(0, IMG.shape[0]):
                    Z_Rotation = self.Rx(0) * self.Ry(0) * self.Rz((((2 * np.pi) / IMG.shape[0]) * i))
                    for j in range(0, IMG.shape[1]):
                        # theta=Gy_Phase*i + stepj*j
                        IMG_vector[j][i] = IMG_vector[j][i] * Z_Rotation
                        IMG_K_Space[Krow][Kcol] += (
                                np.sqrt(np.square(IMG_vector[i][j][0]) + np.square(IMG_vector[i][j][1])) * np.exp(
                            complex(0, -(Gy_Phase * i + Gx_phase * j))))
            self.axis_kspace.imshow(20 * np.log(abs(np.fft.fftshift(IMG_K_Space))), cmap='gray')
            self.canvas_kspace.draw()
            IMG_back = np.fft.ifft2(np.fft.ifftshift(IMG_K_Space))
            self.axis_reconstruct.imshow(abs(IMG_back), cmap='gray')
            self.canvas_reconstruct.draw()

        IMG_K_Space_shift = np.fft.fftshift(IMG_K_Space)

        k_space_magnitude_spectrum = 20 * np.log(np.abs(IMG_K_Space_shift))

        IMG_back = np.fft.ifft2(np.fft.ifftshift(IMG_K_Space_shift))

        self.axis_kspace.imshow(k_space_magnitude_spectrum, cmap='gray')
        self.canvas_kspace.draw()
        self.axis_reconstruct.imshow(abs(IMG_back), cmap='gray')
        self.canvas_reconstruct.draw()
        print("finished generating K Space")
        self.Running_K_Space = 0
        return

    def Rx(self, theta):
        return np.matrix([[1, 0, 0],
                          [0, m.cos(theta), -m.sin(theta)],
                          [0, m.sin(theta), m.cos(theta)]])

    def Ry(self, theta):
        return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                          [0, 1, 0],
                          [-m.sin(theta), 0, m.cos(theta)]])

    def Rz(self, theta):
        return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                          [m.sin(theta), m.cos(theta), 0],
                          [0, 0, 1]])
