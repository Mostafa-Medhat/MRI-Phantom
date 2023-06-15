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
from PyQt5.QtWidgets import QFileDialog, QSizePolicy, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import multiprocessing
from phantominator import shepp_logan

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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import Qt

class select:
    RF = 0
    PG = 1
    FG = 2
    RO = 3
    DR = 4









class Phantom(qtw.QWidget):
    def __init__(self):
        super().__init__()

        uic.loadUi("src/ui/Phantom.ui", self)

        self.i = 0
        self.df = None
        self.df_custom = None
        self.Running_K_Space = 0
        self.Reload_K_Space = 0
        self.old_bright = 0
        self.old_contrast = 0
        self.brightness = 1
        self.prev_x = None
        self.prev_y = None

        self.IMG = None
        self.IMG_Vec = None
        self.T1 = None
        self.T2 = None
        self.Kx = -1
        self.IMG_K_Space = None
        self.sliceMatrix = None 


        self.img_t1 = None
        self.img_t2 = None
        self.img_pd = None

        self.combined_matrix = None

        self.Gmiddle = 1

        self.prep_dic = {"IR Prep": "IRseqTest.json", "T2 Prep": "T2prep.json",
                         "Tagging Prep": "tagging.json"}
        self.aqu_dic = {"GRE Seq": "GRE.json", "Spin Echo Seq": "ٍSpinEcho.json", "SSFP Seq": ""}

        self.figure_sequence = Figure(dpi=80)
        self.figure_sequence_custom = Figure(dpi=80)

        self.phantom_layout()
        self.drawer_layout()

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
        self.pushButton_startReconstruct.clicked.connect(lambda: self.start_K_Space_threading())
        self.horizontalSlider_brightness.sliderReleased.connect(lambda: self.phantom_brightness())
        self.horizontalSlider_contrast.sliderReleased.connect(lambda: self.phantom_contrast())
        # self.canvas_Orig_Spat.mpl_connect('button_press_event', self.getPixel)
        self.canvas_Orig_Spat.mpl_connect('motion_notify_event', self.update_brightness)

        self.canvas_Orig_Spat.mpl_connect('button_press_event', self.getPixel)
        self.comboBox_contrastType.currentIndexChanged.connect(lambda: self.show_contrast())
        self.pushButton_Draw.clicked.connect(lambda: self.start_K_Space_threading())


    def drawer_layout(self,):
        self.figure_viewer_one = Figure(figsize=(50,50), dpi=100)
        self.axis_viewer_one = self.figure_viewer_one.add_subplot()
        self.canvas_viewer_one = FigureCanvas(self.figure_viewer_one)
        self.axis_viewer_one.set_facecolor('black')
        self.canvas_viewer_one.figure.set_facecolor("#19232D")
        self.verticalLayout_3.addWidget(self.canvas_viewer_one)

        self.figure_viewer_two = Figure(figsize=(50,50), dpi=100)
        self.axis_viewer_two = self.figure_viewer_two.add_subplot()
        self.canvas_viewer_two = FigureCanvas(self.figure_viewer_two)
        self.axis_viewer_two.set_facecolor('black')
        self.canvas_viewer_two.figure.set_facecolor("#19232D")
        self.verticalLayout_5.addWidget(self.canvas_viewer_two)

        self.viewer_axes = [self.axis_viewer_one,self.axis_viewer_two]
        for axis in self.viewer_axes:  # removing axes from the figure
            axis.set_xticks([])
            axis.set_yticks([])



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
            axis.plot(np.linspace(0, dataFrame['TR'].Pos, 2), np.zeros(shape=2),
                      color=colors[color],
                      linewidth=0.7)
            axis.axvline(int(dataFrame["TE"].Pos), color='yellow')
            axis.axvline(int(dataFrame["TR"].Pos), color='black')

        RF_Duration = np.linspace(-dataFrame['RF'].Duration / 2,
                                  dataFrame['RF'].Duration / 2, 100)
        axes[0].plot(RF_Duration + dataFrame['RF'].Duration / 2,
                     dataFrame['RF'].Amp * np.sinc(2 * RF_Duration),
                     color=colors[0])
        axes[0].plot(RF_Duration + (dataFrame['RF'].Duration / 2) + dataFrame['TR'].Pos,
                     dataFrame['RF'].Amp * np.sinc(2 * RF_Duration), color=colors[0])
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
        for i in range(-dataFrame['PG'].Amp, dataFrame['PG'].Amp + 1, 1):
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
        if (self.df is not None):
            self.df_custom = self.df.copy()
            print(self.df_custom)
            self.df_custom['RF'].Amp = self.spinBox_RF.value()
            self.df_custom['TR'].Pos = self.spinBox_TR.value()
            self.df_custom['TE'].Pos = self.spinBox_TE.value()
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
            self.img = cv2.imread(phantom_path, cv2.IMREAD_GRAYSCALE)
            self.img_copy = self.img

            self.w, self.h = int(self.figure_Orig_Spat.get_figwidth() * self.figure_Orig_Spat.dpi), int(
                self.figure_Orig_Spat.get_figheight() * self.figure_Orig_Spat.dpi)
            self.img = cv2.resize(self.img, (self.w, self.h), interpolation=cv2.INTER_AREA)
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
            self.generate_contrast()
            # self.get_combined_values()

    def generate_contrast(self):
        self.combined_matrix = np.zeros((self.img.shape[0], self.img.shape[1], 3))
        
        self.img_pd, self.img_t1, self.img_t2 = self.pdt1t2(self.img)

    def show_contrast(self):
        if (self.comboBox_contrastType.currentText() == "Original"):
            self.axis_Orig_Spat.imshow(self.img, cmap='gray')
        elif (self.comboBox_contrastType.currentText() == "T1"):
            self.axis_Orig_Spat.imshow(self.img_t1, cmap='gray')
        elif (self.comboBox_contrastType.currentText() == "T2"):
            self.axis_Orig_Spat.imshow(self.img_t2, cmap='gray')
        elif (self.comboBox_contrastType.currentText() == "PD"):
            self.axis_Orig_Spat.imshow(self.img_pd, cmap='gray')

        self.canvas_Orig_Spat.draw()

    def getPixel(self, event):
        # Get the position of the mouse click
        x = int(round(event.xdata))     # Rows
        y = int(round(event.ydata))     # Columns

        # Get the pixel value at the clicked position
        pd_value = self.combined_matrix[y, x, 0]
        t1_value = self.combined_matrix[y, x, 1]
        t2_value = self.combined_matrix[y, x, 2]
        self.label_pixel.setText(
            f'Pixel at ({x}, {y}): PD value = {round(pd_value, 2)} , T1 Value = {round(t1_value, 2)} , T2 Value = {round(t2_value, 2)}')

    def start_K_Space_threading(self):
        # self.process = multiprocessing.Process(StreamThread)

        if self.Running_K_Space == 1:
            self.Reload_K_Space = 1
            while self.Reload_K_Space and multiprocessing.current_process().is_alive():
                print("retrying")
                time.sleep(1)
        else:
            self.Reload_K_Space = 0

        K_Space_Thread = threading.Thread(
            target=self.vecK_Space)  # replace with this (self.Run_Sequence) to run the custom sequence

        K_Space_Thread.start()

    

    def Rx(self, angle):
        theta = np.radians(angle)
        return np.array([[1, 0, 0],
                          [0, m.cos(theta), -m.sin(theta)],
                          [0, m.sin(theta), m.cos(theta)]])

    def Ry(self, angle):
        theta = np.radians(angle)
        return np.array([[m.cos(theta), 0, m.sin(theta)],
                          [0, 1, 0],
                          [-m.sin(theta), 0, m.cos(theta)]])

    def Rz(self, angle):
        theta = np.radians(angle)
        return np.array([[m.cos(theta), -m.sin(theta), 0],
                          [m.sin(theta), m.cos(theta), 0],
                          [0, 0, 1]])

    

    def brightnessDrag(self, event):
        if event.button == 1 and event.inaxes and event.ydata is not None:
            dy = event.ydata - self.prev_y
            if abs(dy) > 0:
                # Update the brightness value based on the vertical motion
                self.brightness += dy * 0.01
                self.brightness = max(0.0, min(self.brightness, 1.0))
                # Set the new brightness value for the image
                self.img = (self.img_copy * self.brightness)
                self.img = cv2.resize(self.img, (self.w, self.h), interpolation=cv2.INTER_AREA)
                self.axis_Orig_Spat.imshow(self.img, cmap='gray')
                self.canvas_Orig_Spat.draw()
        self.prev_y = event.ydata

    def update_brightness(self, event):
        if event.button == Qt.LeftButton:
            curr_x, curr_y = event.xdata, event.ydata

            if self.prev_x is not None and self.prev_y is not None:
                # Calculate the absolute difference in x-coordinate and y-coordinate between the current and previous event
                dx, dy = abs(curr_x - self.prev_x), abs(curr_y - self.prev_y)
                if dx > dy:
                    contrast = (event.xdata / self.canvas_Orig_Spat.width())
                    contrast = int(contrast * 4)
                    print(contrast)
                    print("Horizontal")
                    self.img = np.power(self.img_copy, contrast)
                else:
                    # brightness = 1 - (event.ydata / self.canvas_Orig_Spat.height())
                    # brightness = int(brightness * 100)
                    print(curr_y - self.prev_y)
                    print("Vertical")
                    self.img = self.img - ((curr_y - self.prev_y) / 3)  # We want to add change not the absolute value

                    ################### Overbright & Underbright conditions ##############

            # Update previous event coordinates
            self.prev_x, self.prev_y = curr_x, curr_y

            self.img = cv2.resize(self.img, (self.w, self.h), interpolation=cv2.INTER_AREA)
            self.axis_Orig_Spat.imshow(self.img, cmap='gray', vmin=0, vmax=255)
            self.canvas_Orig_Spat.draw()


    def phantom_brightness(self):
        self.i += 1
        print(self.i)
        brightness = int(self.horizontalSlider_brightness.value())
        self.img = cv2.addWeighted(self.img_copy, 1, self.img_copy, 0, brightness)
        self.img = cv2.resize(self.img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.axis_Orig_Spat.imshow(self.img, cmap='gray', vmin=0, vmax=255)
        new_brightness = cv2.addWeighted(self.img, 1, self.img, 0, brightness)
        self.axis_Orig_Spat.imshow(new_brightness, cmap='gray', vmin = 0)
        self.canvas_Orig_Spat.draw()

    def phantom_contrast(self):
        contrast = int(self.horizontalSlider_contrast.value())
        contrast = (contrast / 50) + 1
        self.img = np.power(self.img_copy, contrast)
        self.img = cv2.resize(self.img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.axis_Orig_Spat.imshow(self.img, cmap='gray', vmin=0, vmax=255)
        self.canvas_Orig_Spat.draw()

    

    def vectorMagnitude(self, vector):
        return np.sqrt(np.power(vector[0], 2) + np.power(vector[1], 2) + np.power(vector[2], 2))

    #####################################################################################################################

    

    def Decay_Recovery_Matrix(self, IMG_Vectors, T1, T2, t):
        recoved_Matrix = np.zeros(np.shape(IMG_Vectors))
        for i in range(IMG_Vectors.shape[0]):
            for j in range(IMG_Vectors.shape[1]):
                decay_recovery = np.matrix([[np.exp(-t / T2[i, j]), 0, 0],
                                            [0, np.exp(-t / T2[i, j]), 0],
                                            [0, 0, np.exp(-t / T1[i, j])]])

                recoved_Matrix[i, j] = np.dot(decay_recovery, IMG_Vectors[i, j]) + np.array(
                    [0, 0, self.vectorMagnitude(IMG_Vectors[i, j]) * (1 - np.exp(-t / T1[i, j]))])

        return recoved_Matrix

    ########################## get T1, T2, PD images###########################################
    ########these functions take normal phantom image and return T1, T2, PD images

    def rescaleT1(self, t1Value):
        scaledT1 = ((t1Value - 200) / (2000 - 200)) * 255  # t1Value = (scaledT1*((2000-200)/255))+200
        return scaledT1
        # return (scaledT1*((2000-200)/255))+200

    

    def rescaleT2(self, t2Value):
        scaledT2 = ((t2Value - 40) / (500 - 40)) * 255  # t2Value = (scaledT2*((500-40)/255))+40
        return scaledT2

    def rescalePD(self, PDValue):
        scaledPD = ((PDValue - 2) / (120 - 2)) * 255  # PDValue = (scaledPD*((120-2)/255))+2
        return scaledPD

    def pdt1t2(self, image):
        PD_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        t1_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        t2_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] >= 220:
                    PD_image[i][j] = int(self.rescalePD(80))  # scalp
                    self.combined_matrix[i,j,0] = 80
                    t1_image[i][j] = int(self.rescaleT1(324))  # scalp
                    self.combined_matrix[i,j,1] = 324
                    t2_image[i][j] = int(self.rescaleT2(70))  # scalp
                    self.combined_matrix[i,j,2] = 70
                elif 120 > image[i][j] >= 95:
                    PD_image[i][j] = int(self.rescalePD(55))  # white Mater
                    self.combined_matrix[i,j,0] = 55
                    t1_image[i][j] = int(self.rescaleT1(533))  # white Mater
                    self.combined_matrix[i,j,1] = 533
                    t2_image[i][j] = int(self.rescaleT2(50))  # white Mater
                    self.combined_matrix[i,j,2] = 50
                elif 95 > image[i][j] >= 70:
                    PD_image[i][j] = int(self.rescalePD(61.7))  # white Mater
                    self.combined_matrix[i,j,0] = 61.7
                    t1_image[i][j] = int(self.rescaleT1(583))  # white Mater
                    self.combined_matrix[i,j,1] = 583
                    t2_image[i][j] = int(self.rescaleT2(80))  # white Mater
                    self.combined_matrix[i,j,2] = 80
                elif 70 > image[i][j] >= 50:
                    PD_image[i][j] = int(self.rescalePD(74.5))  # Gray Mater
                    self.combined_matrix[i,j,0] = 74.5
                    t1_image[i][j] = int(self.rescaleT1(857))  # Gray Mater
                    self.combined_matrix[i,j,1] = 857
                    t2_image[i][j] = int(self.rescaleT2(100))  # Gray Mater
                    self.combined_matrix[i,j,2] = 100
                elif 50 > image[i][j] >= 26:
                    PD_image[i][j] = int(self.rescalePD(95))  # Gray Mater
                    self.combined_matrix[i,j,0] = 95
                    t1_image[i][j] = int(self.rescaleT1(926))  # Gray Mater
                    self.combined_matrix[i,j,1] = 926
                    t2_image[i][j] = int(self.rescaleT2(120))  # Gray Mater
                    self.combined_matrix[i,j,2] = 120
                else:
                    PD_image[i][j] = int(self.rescalePD(98))  # CSF
                    self.combined_matrix[i,j,0] = 98
                    t1_image[i][j] = int(self.rescaleT1(2000))  # CSF
                    self.combined_matrix[i,j,1] = 2000
                    t2_image[i][j] = int(self.rescaleT2(500))  # CSF
                    self.combined_matrix[i,j,2] = 500
        return PD_image, t1_image, t2_image
    
    def t1t2(self,image):
        t1_val = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        t2_val = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] >= 220:
                    t1_val[i,j] = 324
                    t2_val[i,j] = 70
                elif 120 > image[i][j] >= 95:
                    t1_val[i,j] = 533
                    t2_val[i,j] = 50
                elif 95 > image[i][j] >= 70:
                    t1_val[i,j] = 583
                    t2_val[i,j] = 80
                elif 70 > image[i][j] >= 50:
                    t1_val[i,j] = 857
                    t2_val[i,j] = 100
                elif 50 > image[i][j] >= 26:
                    t1_val[i,j] = 926
                    t2_val[i,j] = 120
                else:
                    t1_val[i,j] = 2000 
                    t2_val[i,j] = 500
        return t1_val , t2_val


    
    

    ################### try the vectorization method #########################


    ####################### latest update #####################################################

    def generate_Sequence(self,dataFrame):
        timeLine = np.zeros((5,1000))
        RFTime = np.zeros(1000)
        GyTime = np.zeros(1000)
        GxTime = np.zeros(1000)
        ReadoutTime = np.zeros(1000)
        DecayRecTime = np.zeros(1000)
        
        for irf in range(np.array(dataFrame['RF'].Amp).shape[0]):
            rfPos = int(dataFrame['RF'].Pos[irf])
            RFTime[rfPos] = int(dataFrame['RF'].Amp[irf])

        for ipg in range(np.array(dataFrame['PG'].Amp).shape[0]):
            gyPos = int(dataFrame['PG'].Pos[ipg])
            gyDur = int(dataFrame['PG'].Duration[ipg])
            GyTime[gyPos:gyPos+gyDur] = int(dataFrame['PG'].Amp[ipg])

        for ifg in range(np.array(dataFrame['FG'].Amp).shape[0]):
            gxPos = int(dataFrame['FG'].Pos[ifg])
            gxDur = int(dataFrame['FG'].Duration[ifg])
            GxTime[gxPos:gxPos+gxDur] = int(dataFrame['FG'].Amp[ifg])

        for iro in range(np.array(dataFrame['RO'].Pos).shape[0]):
            roPos = int(dataFrame['RO'].Pos[iro])
            roDur = int(dataFrame['RO'].Duration[iro])
            ReadoutTime[roPos:roPos+roDur] = 1
        
        for idr in range(np.array(dataFrame['DR'].Pos).shape[0]):
            drPos = int(dataFrame['DR'].Pos[idr])
            # drDur = int(df['DR'].Duration[idr])
            DecayRecTime[drPos] = int(dataFrame['DR'].Amp[idr])

        timeLine[0] = RFTime
        timeLine[1] = GyTime
        timeLine[2] = GxTime
        timeLine[3] = ReadoutTime
        timeLine[4] = DecayRecTime

        # print(timeLine)

        return timeLine

    def gradientRotAngles(self,imgVec,maxAngle,GXorY:bool,withReadOut,changedForIteration:bool,iterationCont = 0):
        if GXorY:
            size = imgVec.shape[1]
        else:
            size = imgVec.shape[0]
        if withReadOut:
            if changedForIteration:
                Angles = np.linspace(0,(maxAngle-(maxAngle / (size)))* ((int(size/2))-iterationCont),size)#but amplitude
            else:
                Angles = np.linspace(0,(maxAngle-(maxAngle / (size))),size)#but amplitude
        else:
            if changedForIteration:
                Angles = np.linspace(0,(((maxAngle-(maxAngle / (size)))/(size)))* ((int(size/2))-iterationCont),size)
            else:
                Angles = np.linspace(0,(((maxAngle-(maxAngle / (size)))/(size))),size)
        
                
        return Angles

    def runSeq(self,timeLine,Ky = 0):
       

        # T1 = self.combined_matrix[:,:,1]
        # T2 = self.combined_matrix[:,:,2]
        # for Ky in range(self.IMG_Vec.shape[0]):
        for i in range(1000):
            if timeLine[select.RF][i] != 0:
                
                self.sliceMatrix = np.squeeze(np.matmul(self.Rx(timeLine[select.RF][i]),np.expand_dims(self.sliceMatrix,axis=(-1))),axis=(-1))
                

            if timeLine[select.PG][i] != 0:
                GyAngles = self.gradientRotAngles(self.IMG_Vec,timeLine[select.PG][i],False,timeLine[select.RO][i],True,Ky)
                
                GyAnglesMat = np.array(list(map(lambda theta: [self.Rz(theta)],GyAngles))) #to rotate rows
                self.sliceMatrix = np.squeeze(np.matmul(GyAnglesMat,np.expand_dims(self.sliceMatrix,axis=(-1))),axis=(-1))
                
            
            if timeLine[select.FG][i] != 0:
                
                
                
                GxAngles = self.gradientRotAngles(self.IMG_Vec,timeLine[select.FG][i],True,timeLine[select.RO][i],False)
                
                GxAnglesMat = np.array(list(map(lambda theta: self.Rz(theta),GxAngles))) #to rotate cols
                self.sliceMatrix = np.squeeze(np.matmul(GxAnglesMat,np.expand_dims(self.sliceMatrix,axis=(-1))),axis=(-1))
                self.Kx += 1
                if self.Kx >= self.IMG_Vec.shape[1]:
                    self.Kx = 0


            if timeLine[select.RO][i] != 0:
                
                sigmaX = np.sum(self.sliceMatrix[:, :, 0])
                sigmaY = np.sum(self.sliceMatrix[:, :, 1]) 
                self.IMG_K_Space[-(int(self.IMG_Vec.shape[0]/2))+Ky,self.Kx] = complex(sigmaX,sigmaY) #int(Kx*(Ky/self.IMG_Vec.shape[0]))

            if timeLine[select.DR][i] != 0:
                self.sliceMatrix = self.Decay_Recovery_Matrix(self.sliceMatrix,self.T1,self.T2,timeLine[select.DR][i])
    
    def vecK_Space(self):
        # print(self.combined_matrix[:,:,2])
        self.axis_kspace.clear()
        self.axis_kspace.set_yticks([])
        self.canvas_kspace.draw()

        

        self.IMG = cv2.resize(self.img,
                         (int(self.comboBox_kspace_size.currentText()), int(self.comboBox_kspace_size.currentText())))


        
        self.IMG_Vec = np.zeros((self.IMG.shape[0],self.IMG.shape[1],3))
        self.T1 , self.T2 = self.t1t2(self.IMG)
        
        print(np.array(self.combined_matrix[:,:,1]).shape)
        self.IMG_Vec[:,:,2] = self.IMG 
        self.IMG_K_Space = np.zeros((self.IMG.shape),dtype=np.complex_)
        self.sliceMatrix = self.IMG_Vec.copy()

      
        prep_sequence_path = self.prep_dic[self.comboBox_2.currentText()]  # "T2prep.json"
        acc_sequence_path = self.aqu_dic[self.comboBox_3.currentText()]  # "GRE.json"
        if prep_sequence_path != '':
            prep_dictionary = json.load(open(prep_sequence_path))
            prep_df = pd.DataFrame(prep_dictionary)

        acc_dictionary = json.load(open(acc_sequence_path))
        acc_df = pd.DataFrame(acc_dictionary)
        
        #read prep
        #read acusition
        for Ky in range(self.IMG_Vec.shape[0]):

            self.Running_K_Space = 1

            # check if k_Space relaod is needed
            if self.Reload_K_Space == 1:
                self.Reload_K_Space = 0
                self.Running_K_Space = 0
                return

            if prep_sequence_path != '':
                self.runSeq(self.generate_Sequence(prep_df))
            self.runSeq(self.generate_Sequence(acc_df),Ky=Ky)
            
            # shift + tab the under section to mke the procecessing way faster as the plotting takes time
            w, h = int(self.figure_Orig_Spat.get_figwidth() * self.figure_Orig_Spat.dpi), int(
                    self.figure_Orig_Spat.get_figheight() * self.figure_Orig_Spat.dpi)

            k_space_magnitude_spectrum = 20 * np.log(abs(np.fft.fftshift(self.IMG_K_Space)))
            k_space_magnitude_spectrum = cv2.resize(k_space_magnitude_spectrum, (w, h), interpolation=cv2.INTER_AREA)
            self.axis_kspace.imshow(k_space_magnitude_spectrum, cmap='gray')
            self.axis_kspace.set_yticks([])
            self.canvas_kspace.draw()
            # update the reconstructed image for every row added to the K_Space
            IMG_back = np.fft.ifft2(self.IMG_K_Space)
            abs_img_back = abs(IMG_back)
            abs_img_back = cv2.resize(abs_img_back, (w, h), interpolation=cv2.INTER_AREA)
            self.axis_reconstruct.imshow(abs_img_back, cmap='gray')
            self.canvas_reconstruct.draw()

        text = str(self.comboBox_2.currentText())+ " + " +str(self.comboBox_3.currentText())

        if(self.comboBox_viewer.currentText() == "Viewer 1"):
            w, h = int(self.figure_viewer_one.get_figwidth() * self.figure_viewer_one.dpi), int(
                self.figure_viewer_one.get_figheight() * self.figure_viewer_one.dpi)
            abs_img_back = cv2.resize(abs_img_back, (w, h), interpolation=cv2.INTER_AREA)
            self.label_viewerOne.setText(text)
            self.axis_viewer_one.imshow(abs_img_back, cmap='gray')
            self.canvas_viewer_one.draw()

        elif(self.comboBox_viewer.currentText() == "Viewer 2"):
            w, h = int(self.figure_viewer_two.get_figwidth() * self.figure_viewer_two.dpi), int(
                self.figure_viewer_two.get_figheight() * self.figure_viewer_two.dpi)
            abs_img_back = cv2.resize(abs_img_back, (w, h), interpolation=cv2.INTER_AREA)
            self.label_viewerTwo.setText(text)
            self.axis_viewer_two.imshow(abs_img_back, cmap='gray')
            self.canvas_viewer_two.draw()

        self.Running_K_Space = 0
        self.Reload_K_Space = 0
        return
