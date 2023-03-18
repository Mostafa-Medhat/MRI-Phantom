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

        self.i = 0
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
        self.pushButton_startReconstruct.clicked.connect(lambda: self.start_K_Space_threading())
        self.comboBox_kspace_size.currentIndexChanged.connect(lambda: self.start_K_Space_threading())
        self.horizontalSlider_brightness.sliderReleased.connect(lambda: self.phantom_brightness())
        self.horizontalSlider_contrast.sliderReleased.connect(lambda: self.phantom_contrast())
        self.canvas_Orig_Spat.mpl_connect('button_press_event', self.getPixel)

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
            # if self.Running_K_Space == 1:
            #     self.Reload_K_Space = 1
            # else:
            #     self.Reload_K_Space = 0
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

    def getPixel(self, event):
        # Get the position of the mouse click
        x = int(round(event.xdata))
        y = int(round(event.ydata))

        # Get the pixel value at the clicked position
        img_combined = np.zeros((self.img.shape[0], self.img.shape[1], 1), dtype=np.uint8)
        img_combined[:, :, 0] = self.img
        pixel_values = img_combined[y, x]
        value = (pixel_values[0] * ((120 - 2) / 255)) + 2
        self.label_pixel.setText(f'Pixel PD value at ({x}, {y}): {round(value, 4)}')

    def start_K_Space_threading(self):
        # self.process = multiprocessing.Process(StreamThread)

        if self.Running_K_Space == 1:
            self.Reload_K_Space = 1
            while self.Reload_K_Space and multiprocessing.current_process().is_alive():
                print("retrying")
                time.sleep(1)
        else:
            self.Reload_K_Space = 0

        K_Space_Thread = threading.Thread(target=self.generate_kspace)

        K_Space_Thread.start()

    def generate_kspace(self):

        # print("reintering kspace function")

        self.axis_kspace.clear()
        self.canvas_kspace.draw()

        IMG = cv2.resize(self.img,
                         (int(self.comboBox_kspace_size.currentText()), int(self.comboBox_kspace_size.currentText())))

        IMG_K_Space = np.zeros((IMG.shape[0], IMG.shape[1]), dtype=np.complex_)

        IMG_vector = np.zeros((IMG.shape[0], IMG.shape[1], 3), dtype=np.float_)

        self.axis_kspace.imshow(abs((IMG_K_Space)), cmap='gray')

        Min_KX, Max_KX, Min_KY, Max_KY = self.setGradientLimits(IMG, Gx_zero_in_middel=1, Gy_zero_in_middel=1)

        # IMG_vector[:,:,:] = 0

        # initialize our vectors
        IMG_vector[:, :, 2] = IMG[:, :]

        for Ky in range(Min_KY, Max_KY):

            self.Running_K_Space = 1

            # simulate the RF effect on our Matrix
            RF_RotatedMatrix = self.RF_Rotation(IMG_vector, 90)

            for Kx in range(Min_KX, Max_KX):

                # check if k_Space relaod is needed
                if self.Reload_K_Space == 1:
                    self.Reload_K_Space = 0
                    self.Running_K_Space = 0
                    return

                # changing the Gy & Gx steps
                Gy_step = (360 / (Max_KY - Min_KY)) * Ky
                Gx_step = (360 / (Max_KX - Min_KX)) * Kx

                # Apply the Gx & Gy effect to our vectors
                Gxy_EncodedMatrix = self.Gxy_Rotation(RF_RotatedMatrix, Gy_step, Gx_step)

                # sum all the vectors projections in x
                sigmaX = np.sum(Gxy_EncodedMatrix[:, :, 0])
                # sum all the vectors projections in y
                sigmaY = np.sum(Gxy_EncodedMatrix[:, :, 1])
                # set sigmaX as real part and sigmaY as imaginary part of the K_Space
                valueToAdd = complex(sigmaX, sigmaY)
                # save the value to the K_Space at it's relative place acording to Ky and Kx
                IMG_K_Space[-Ky, -Kx] = valueToAdd

            # updates the K_space image for every row added to it with the addition of applying fftshift to it
            self.axis_kspace.imshow(20 * np.log(abs(np.fft.fftshift(IMG_K_Space))), cmap='gray')
            self.canvas_kspace.draw()
            # update the reconstructed image for every row added to the K_Space
            IMG_back = np.fft.ifft2(np.fft.ifftshift(IMG_K_Space))
            self.axis_reconstruct.imshow(abs(IMG_back), cmap='gray')
            self.canvas_reconstruct.draw()
            # print the progress of our K_Space
            print(Ky - Min_KY + 1)

        self.Running_K_Space = 0
        self.Reload_K_Space = 0

        IMG_K_Space_shift = np.fft.fftshift(IMG_K_Space)
        # Rescale the output of the K_Space
        k_space_magnitude_spectrum = 20 * np.log(np.abs(IMG_K_Space_shift))
        # reconstruct our image back from the generated k_Space
        IMG_back = np.fft.ifft2(np.fft.ifftshift(IMG_K_Space_shift))

        self.axis_kspace.imshow(k_space_magnitude_spectrum, cmap='gray')
        self.canvas_kspace.draw()
        self.axis_reconstruct.imshow(abs(IMG_back), cmap='gray')
        self.canvas_reconstruct.draw()
        print("finished generating K Space")

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

    # function to simulate RF pulse effect on our matrix
    def RF_Rotation(self, matrix, RF_rotation_deg):
        RF_Rotated_Matrix = np.zeros(np.shape(matrix))

        for i in range(RF_Rotated_Matrix.shape[0]):
            for j in range(RF_Rotated_Matrix.shape[1]):
                # apply the rotations around the X axis to all the elements of the matrix
                RF_Rotated_Matrix[i, j] = np.dot(self.Rx(np.radians(RF_rotation_deg)), matrix[i, j])

        return RF_Rotated_Matrix

    # function to simulate Gradient x & y effect on our matrix
    def Gxy_Rotation(self, matrix, Gy_step_deg, Gx_step_deg):
        Gxy_Rotated_Matrix = np.zeros(np.shape(matrix))

        for i in range(Gxy_Rotated_Matrix.shape[0]):
            for j in range(Gxy_Rotated_Matrix.shape[1]):
                # compute the total rotation effec from gradients aroung Z axis
                Gxy_rotation_Theta = np.radians(Gy_step_deg * i + Gx_step_deg * j)
                # apply the rotations to all the elements of the matrix
                Gxy_Rotated_Matrix[i, j] = np.dot(self.Rz(Gxy_rotation_Theta), matrix[i, j])

        return Gxy_Rotated_Matrix

    # function to set the limits of Gradients (ex: [0,matrix row size] or [-matrix row size/2,matrix row size/2])
    def setGradientLimits(self, matrix, Gx_zero_in_middel=0, Gy_zero_in_middel=0):
        Set_Min_KX = 0
        Set_Max_KX = 0
        Set_Min_KY = 0
        Set_Max_KY = 0

        if Gx_zero_in_middel:
            Set_Min_KX = int(-matrix.shape[1] / 2)
            Set_Max_KX = int(matrix.shape[1] / 2)

        else:
            Set_Min_KX = int(0)
            Set_Max_KX = int(matrix.shape[1])

        if Gy_zero_in_middel:
            Set_Min_KY = int(-matrix.shape[0] / 2)
            Set_Max_KY = int(matrix.shape[0] / 2)
        else:
            Set_Min_KY = int(0)
            Set_Max_KY = int(matrix.shape[0])
        return Set_Min_KX, Set_Max_KX, Set_Min_KY, Set_Max_KY

    def phantom_brightness(self):
        self.i += 1
        print(self.i)
        brightness = int(self.horizontalSlider_brightness.value())
        new_brightness = cv2.addWeighted(self.img, 1, self.img, 0, brightness)
        self.axis_Orig_Spat.imshow(new_brightness, cmap='gray')
        self.canvas_Orig_Spat.draw()

    def phantom_contrast(self):
        contrast = int(self.horizontalSlider_contrast.value())
        contrast = contrast / 5
        new_contrast = np.clip(contrast * self.img, 0, 255).astype(np.uint8)
        self.axis_Orig_Spat.imshow(new_contrast, cmap='gray')
        self.canvas_Orig_Spat.draw()

    ######################### for the Decay Recovery effect #########################################################
    def get_T1_value(self, image):
        T1_Matrix = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                T1_Matrix[i, j] = (image[i, j] * ((2000 - 200) / 255)) + 200
        return T1_Matrix

    def get_T2_value(self, image):
        T2_Matrix = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                T2_Matrix[i, j] = (image[i, j] * ((500 - 40) / 255)) + 40
        return T2_Matrix

    def get_PD_value(self, image):
        PD_Matrix = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                PD_Matrix[i, j] = (image[i, j] * ((120 - 2) / 255)) + 2
        return PD_Matrix

    def vectorMagnitude(self, vector):
        return np.sqrt(np.power(vector[0], 2) + np.power(vector[1], 2) + np.power(vector[2], 2))

    def Decay_Recovery_Matrix(self, IMG_Vectors, T1, T2, TE=0.001, TR=0.5):
        recoved_Matrix = np.zeros(np.shape(IMG_Vectors))
        for i in range(IMG_Vectors.shape[0]):
            for j in range(IMG_Vectors.shape[1]):
                decay_recovery = np.matrix([[np.exp(-TE / T2[i, j]), 0, 0],
                                            [0, np.exp(-TE / T2[i, j]), 0],
                                            [0, 0, np.exp(-TR / T1[i, j])]])

                recoved_Matrix[i, j] = np.dot(decay_recovery, IMG_Vectors[i, j]) + np.array(
                    [0, 0, self.vectorMagnitude(IMG_Vectors[i, j]) * (1 - np.exp(-TR / T1[i, j]))])

        return recoved_Matrix
    #####################################################################################################################
