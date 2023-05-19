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


        self.img_t1 = None
        self.img_t2 = None
        self.img_pd = None

        self.combined_matrix = None

        self.Gmiddle = 1

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
        self.horizontalSlider_brightness.sliderReleased.connect(lambda: self.phantom_brightness())
        self.horizontalSlider_contrast.sliderReleased.connect(lambda: self.phantom_contrast())
        # self.canvas_Orig_Spat.mpl_connect('button_press_event', self.getPixel)
        self.canvas_Orig_Spat.mpl_connect('motion_notify_event', self.update_brightness)

        self.canvas_Orig_Spat.mpl_connect('button_press_event', self.getPixel)
        self.comboBox_contrastType.currentIndexChanged.connect(lambda: self.show_contrast())

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
        self.img_t1 = self.t1(self.img)
        self.img_t2 = self.t2(self.img)
        self.img_pd = self.pd(self.img)

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

    def generate_kspace(self):

        # print("reintering kspace function")

        self.axis_kspace.clear()
        self.axis_kspace.set_yticks([])
        self.canvas_kspace.draw()

        IMG = cv2.resize(self.img,
                         (int(self.comboBox_kspace_size.currentText()), int(self.comboBox_kspace_size.currentText())))

        IMG_K_Space = np.zeros((IMG.shape[0], IMG.shape[1]), dtype=np.complex_)

        IMG_vector = np.zeros((IMG.shape[0], IMG.shape[1], 3), dtype=np.float_)

        self.axis_kspace.imshow(abs((IMG_K_Space)), cmap='gray')

        Min_KX, Max_KX, Min_KY, Max_KY = self.setGradientLimits(IMG, Gx_zero_in_middel=1, Gy_zero_in_middel=1)
        print(Min_KX, Max_KX, Min_KY, Max_KY)

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
            w, h = int(self.figure_Orig_Spat.get_figwidth() * self.figure_Orig_Spat.dpi), int(
                self.figure_Orig_Spat.get_figheight() * self.figure_Orig_Spat.dpi)

            k_space_magnitude_spectrum = 20 * np.log(abs(np.fft.fftshift(IMG_K_Space)))
            k_space_magnitude_spectrum = cv2.resize(k_space_magnitude_spectrum, (w, h), interpolation=cv2.INTER_AREA)
            self.axis_kspace.imshow(k_space_magnitude_spectrum, cmap='gray')
            self.axis_kspace.set_yticks([])
            self.canvas_kspace.draw()
            # update the reconstructed image for every row added to the K_Space
            IMG_back = np.fft.ifft2(np.fft.ifftshift(IMG_K_Space))
            abs_img_back = abs(IMG_back)
            abs_img_back = cv2.resize(abs_img_back, (w, h), interpolation=cv2.INTER_AREA)
            self.axis_reconstruct.imshow(abs_img_back, cmap='gray')
            self.canvas_reconstruct.draw()
            # print the progress of our K_Space
            print(Ky - Min_KY + 1)

        self.Running_K_Space = 0
        self.Reload_K_Space = 0

        ####################################### Old Draw #########################################
        # IMG_K_Space_shift = np.fft.fftshift(IMG_K_Space)
        # Rescale the output of the K_Space
        # k_space_magnitude_spectrum = 20 * np.log(np.abs(IMG_K_Space_shift))
        # reconstruct our image back from the generated k_Space
        # IMG_back = np.fft.ifft2(np.fft.ifftshift(IMG_K_Space_shift))

        # self.axis_kspace.imshow(k_space_magnitude_spectrum, cmap='gray')
        # self.canvas_kspace.draw()
        # self.axis_reconstruct.imshow(IMG_back, cmap='gray')
        # self.canvas_reconstruct.draw()
        print("finished generating K Space")

        return

    def Rx(self, theta):
        return np.array([[1, 0, 0],
                          [0, m.cos(theta), -m.sin(theta)],
                          [0, m.sin(theta), m.cos(theta)]])

    def Ry(self, theta):
        return np.array([[m.cos(theta), 0, m.sin(theta)],
                          [0, 1, 0],
                          [-m.sin(theta), 0, m.cos(theta)]])

    def Rz(self, theta):
        return np.array([[m.cos(theta), -m.sin(theta), 0],
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
            self.Gmiddle = 1
        else:
            Set_Min_KY = int(0)
            Set_Max_KY = int(matrix.shape[0])
            self.Gmiddle = 0
        return Set_Min_KX, Set_Max_KX, Set_Min_KY, Set_Max_KY

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

    ######################### for the Decay Recovery effect #########################################################
    # def get_T1_value(self, image):
    #     T1_Matrix = np.zeros((image.shape[0], image.shape[1]))
    #     for i in range(image.shape[0]):
    #         for j in range(image.shape[1]):
    #             T1_Matrix[i, j] = (image[i, j] * ((2000 - 200) / 255)) + 200
    #     return T1_Matrix

    # def get_T2_value(self, image):
    #     T2_Matrix = np.zeros((image.shape[0], image.shape[1]))
    #     for i in range(image.shape[0]):
    #         for j in range(image.shape[1]):
    #             T2_Matrix[i, j] = (image[i, j] * ((500 - 40) / 255)) + 40
    #     return T2_Matrix

    # def get_PD_value(self, image):
    #     PD_Matrix = np.zeros((image.shape[0], image.shape[1]))
    #     for i in range(image.shape[0]):
    #         for j in range(image.shape[1]):
    #             PD_Matrix[i, j] = (image[i, j] * ((120 - 2) / 255)) + 2
    #     return PD_Matrix

    # def get_combined_values(self):
    #     self.combined_matrix = np.zeros((self.img.shape[0], self.img.shape[1], 3))
    #     self.combined_matrix[:, :, 0] = self.get_PD_value(self.img)
    #     self.combined_matrix[:, :, 1] = self.get_T1_value(self.img)
    #     self.combined_matrix[:, :, 2] = self.get_T2_value(self.img)

    def vectorMagnitude(self, vector):
        return np.sqrt(np.power(vector[0], 2) + np.power(vector[1], 2) + np.power(vector[2], 2))

    #####################################################################################################################

    ############## functions to read current sequence and apply it with decay recovery ##################################
    def Gx_Rotation(self, matrix, Gx_step_deg):
        Gx_Rotated_Matrix = np.zeros(np.shape(matrix))

        for i in range(Gx_Rotated_Matrix.shape[0]):
            for j in range(Gx_Rotated_Matrix.shape[1]):
                Gx_rotation_Theta = np.radians(Gx_step_deg * j)
                Gx_Rotated_Matrix[i, j] = np.dot(self.Rz(Gx_rotation_Theta), matrix[i, j])

        return Gx_Rotated_Matrix

    def Gy_Rotation(self, matrix, Gy_step_deg):
        Gy_Rotated_Matrix = np.zeros(np.shape(matrix))

        for i in range(Gy_Rotated_Matrix.shape[0]):
            for j in range(Gy_Rotated_Matrix.shape[1]):
                Gy_rotation_Theta = np.radians(Gy_step_deg * i)
                Gy_Rotated_Matrix[i, j] = np.dot(self.Rz(Gy_rotation_Theta), matrix[i, j])

        return Gy_Rotated_Matrix

    def RF(self, matrix, amp):
        TRF_rotatedMatrix = self.RF_Rotation(matrix, amp)
        return TRF_rotatedMatrix

    def Gy(self, matrix, amp, duration):
        min = 0
        max = duration
        if self.Gmiddle and duration > 1:
            min = int(-duration / 2)
            max = int(duration / 2)

        for deltaT in range(min, max):
            # Gy_step = int((amp / (max-min)) * (deltaT+1))
            Gy_rotated_Matrix = self.Gy_Rotation(matrix, amp)
        # self.Gy_Ky += 1
        return Gy_rotated_Matrix

    def Gx(self, matrix, amp, duration, readout=0):
        returned_K_Space_array = np.zeros(matrix.shape[1], dtype=np.complex_)

        min = 0
        max = duration
        if self.Gmiddle and duration > 1:
            min = int(-duration / 2)
            max = int(duration / 2)

        for deltaT in range(min, max):
            Gx_step = (amp / (max - min)) * deltaT
            Gx_rotated_Matrix = self.Gx_Rotation(matrix, Gx_step)
            if readout:
                returned_K_Space_array[-deltaT] = self.readOut(Gx_rotated_Matrix)
        if readout:
            return Gx_rotated_Matrix, returned_K_Space_array
        else:
            return Gx_rotated_Matrix

    def readOut(self, matrix):

        sigmaX = np.sum(matrix[:, :, 0])
        sigmaY = np.sum(matrix[:, :, 1])
        valueToAdd = complex(sigmaX, sigmaY)
        return valueToAdd

    def generate_sequence(self, dataFram):
        shift = 0
        Translated_Sequence = np.zeros([1000])
        for i in range(int(self.comboBox_kspace_size.currentText())):  # int(dataFram['PG'].NumOfRep)
            rfpos = dataFram['RF'].Pos
            rfduration = dataFram['RF'].Duration
            Translated_Sequence[(shift + rfpos)] = 1
            gypos = dataFram['PG'].Pos
            gyduration = dataFram['PG'].Duration
            Translated_Sequence[(shift + gypos)] = 2
            gxpos = dataFram['FG'].Pos
            # gxduration = int(self.comboBox_kspace_size.currentText())
            gxduration = dataFram['FG'].Duration
            Translated_Sequence[(shift + gxpos)] = 3
            shift += rfpos + gypos + gxpos  # rfpos+rfduration+gypos+gyduration+gxpos+gxduration
        # print(Translated_Sequence)
        return Translated_Sequence

    def Run_Sequence(self):

        self.axis_kspace.clear()
        self.axis_kspace.set_yticks([])
        self.canvas_kspace.draw()

        IMG = cv2.resize(self.img,
                         (int(self.comboBox_kspace_size.currentText()), int(self.comboBox_kspace_size.currentText())))

        IMG_K_Space = np.zeros((IMG.shape[0], IMG.shape[1]), dtype=np.complex_)

        IMG_vector = np.zeros((IMG.shape[0], IMG.shape[1], 3), dtype=np.float_)

        self.axis_kspace.imshow(abs((IMG_K_Space)), cmap='gray')

        seq = self.generate_sequence(self.df)

        Min_KX, Max_KX, Min_KY, Max_KY = self.setGradientLimits(IMG, Gx_zero_in_middel=1, Gy_zero_in_middel=1)

        IMG_vector[:, :, 2] = IMG[:, :]

        # seq = self.generate_sequence(self.df)

        Gy_counter = Min_KY

        Gx_counter = Min_KX

        T1 = self.get_T1_value(IMG)
        T2 = self.get_T2_value(IMG)
        TR = 500
        TE = 10

        rotMatrix = np.zeros(np.shape(IMG_vector))
        rotMatrix = IMG_vector

        # min = 0
        # max = IMG_vector.shape[0]
        # if self.Gmiddle and IMG_vector.shape[0] > 1:
        #     min = int(-IMG_vector.shape[0]/2)
        #     max = int(IMG_vector.shape[0]/2)

        for i in range(seq.shape[0]):
            self.Running_K_Space = 1

            # check if k_Space relaod is needed
            if self.Reload_K_Space == 1:
                self.Reload_K_Space = 0
                self.Running_K_Space = 0
                return

            # RF
            if seq[i] == 1:
                if i == 0:
                    rotMatrix = IMG_vector
                else:
                    rotMatrix = self.Decay_Recovery_Matrix(rotMatrix, T1, T2, (TR - TE))
                rotMatrix = self.RF(rotMatrix, int(self.df['RF'].Amp))

            # Gy
            if seq[i] == 2:
                rotMatrix = self.Gy(rotMatrix, amp=(360 / IMG_vector.shape[0]) * Gy_counter,
                                    duration=int(self.df['PG'].Duration))
                if Gy_counter == (Max_KY):
                    Gy_counter = Min_KY
                else:
                    Gy_counter += 1

            # Gx
            if seq[i] == 3:
                # rotMatrix = self.Gx(rotMatrix,-180,int(IMG_vector.shape[1]/2),0)
                rotMatrix = self.Decay_Recovery_Matrix(rotMatrix, T1, T2, TE)
                rotMatrix, IMG_K_Space[-Gy_counter, :] = self.Gx(rotMatrix, amp=int(self.df['FG'].Amp),
                                                                 duration=int(self.df['FG'].Duration), readout=1)
                # print(IMG_K_Space)
                # updates the K_space image for every row added to it with the addition of applying fftshift to it
                w, h = int(self.figure_Orig_Spat.get_figwidth() * self.figure_Orig_Spat.dpi), int(
                    self.figure_Orig_Spat.get_figheight() * self.figure_Orig_Spat.dpi)

                k_space_magnitude_spectrum = 20 * np.log(abs(np.fft.fftshift(IMG_K_Space)))
                k_space_magnitude_spectrum = cv2.resize(k_space_magnitude_spectrum, (w, h), interpolation=cv2.INTER_AREA)
                self.axis_kspace.imshow(k_space_magnitude_spectrum, cmap='gray')
                self.axis_kspace.set_yticks([])
                self.canvas_kspace.draw()
                # update the reconstructed image for every row added to the K_Space
                IMG_back = np.fft.ifft2(np.fft.ifftshift(IMG_K_Space))
                abs_img_back = abs(IMG_back)
                abs_img_back = cv2.resize(abs_img_back, (w, h), interpolation=cv2.INTER_AREA)
                self.axis_reconstruct.imshow(abs_img_back, cmap='gray')
                self.canvas_reconstruct.draw()
                # rotMatrix = self.Decay_Recovery_Matrix(rotMatrix,T1,T2,(TR-TE))
                print(Gy_counter)

        self.Running_K_Space = 0
        self.Reload_K_Space = 0

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

    # def rescaleT12(t12Value):
    #     scaledT12 = 255-t12Value
    #     return scaledT12

    def rescaleT2(self, t2Value):
        scaledT2 = ((t2Value - 40) / (500 - 40)) * 255  # t2Value = (scaledT2*((500-40)/255))+40
        return scaledT2

    def rescalePD(self, PDValue):
        scaledPD = ((PDValue - 2) / (120 - 2)) * 255  # PDValue = (scaledPD*((120-2)/255))+2
        return scaledPD

    def t1(self, image):
        t1_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] >= 220:
                    t1_image[i][j] = int(self.rescaleT1(324))  # scalp
                    self.combined_matrix[i,j,1] = 324
                elif 120 > image[i][j] >= 95:
                    t1_image[i][j] = int(self.rescaleT1(533))  # white Mater
                    self.combined_matrix[i,j,1] = 533
                elif 95 > image[i][j] >= 70:
                    t1_image[i][j] = int(self.rescaleT1(583))  # white Mater
                    self.combined_matrix[i,j,1] = 583
                elif 70 > image[i][j] >= 50:
                    t1_image[i][j] = int(self.rescaleT1(857))  # Gray Mater
                    self.combined_matrix[i,j,1] = 857
                elif 50 > image[i][j] >= 26:
                    t1_image[i][j] = int(self.rescaleT1(926))  # Gray Mater
                    self.combined_matrix[i,j,1] = 926
                else:
                    t1_image[i][j] = int(self.rescaleT1(2000))  # CSF
                    self.combined_matrix[i,j,1] = 2000
        return t1_image

    def t2(self, image):
        t2_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] >= 220:
                    t2_image[i][j] = int(self.rescaleT2(70))  # scalp
                    self.combined_matrix[i,j,2] = 70
                elif 120 > image[i][j] >= 95:
                    t2_image[i][j] = int(self.rescaleT2(50))  # white Mater
                    self.combined_matrix[i,j,2] = 50
                elif 95 > image[i][j] >= 70:
                    t2_image[i][j] = int(self.rescaleT2(80))  # white Mater
                    self.combined_matrix[i,j,2] = 80
                elif 70 > image[i][j] >= 50:
                    t2_image[i][j] = int(self.rescaleT2(100))  # Gray Mater
                    self.combined_matrix[i,j,2] = 100
                elif 50 > image[i][j] >= 26:
                    t2_image[i][j] = int(self.rescaleT2(120))  # Gray Mater
                    self.combined_matrix[i,j,2] = 120
                else:
                    t2_image[i][j] = int(self.rescaleT2(500))  # CSF
                    self.combined_matrix[i,j,2] = 500
        return t2_image

    def pd(self, image):
        PD_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] >= 220:
                    PD_image[i][j] = int(self.rescalePD(80))  # scalp
                    self.combined_matrix[i,j,0] = 80
                elif 120 > image[i][j] >= 95:
                    PD_image[i][j] = int(self.rescalePD(55))  # white Mater
                    self.combined_matrix[i,j,0] = 55
                elif 95 > image[i][j] >= 70:
                    PD_image[i][j] = int(self.rescalePD(61.7))  # white Mater
                    self.combined_matrix[i,j,0] = 61.7
                elif 70 > image[i][j] >= 50:
                    PD_image[i][j] = int(self.rescalePD(74.5))  # Gray Mater
                    self.combined_matrix[i,j,0] = 74.5
                elif 50 > image[i][j] >= 26:
                    PD_image[i][j] = int(self.rescalePD(95))  # Gray Mater
                    self.combined_matrix[i,j,0] = 95
                else:
                    PD_image[i][j] = int(self.rescalePD(98))  # CSF
                    self.combined_matrix[i,j,0] = 98
        return PD_image
    

    ################### try the vectorization method #########################3
    def vecK_Space(self):

        self.axis_kspace.clear()
        self.axis_kspace.set_yticks([])
        self.canvas_kspace.draw()


        IMG = cv2.resize(self.img,
                         (int(self.comboBox_kspace_size.currentText()), int(self.comboBox_kspace_size.currentText())))

        IMG_K_Space = np.zeros((IMG.shape[0], IMG.shape[1]), dtype=np.complex_)

        IMG_vector = np.zeros((IMG.shape[0], IMG.shape[1], 3), dtype=np.float_)

        # K_Space = np.zeros((IMG.shape),dtype=np.complex_)

        IMG_vector[:, :, 2] = IMG[:, :]

        sliceMatrix = IMG_vector.copy()

        for Ky in range(IMG_vector.shape[0]):

            self.Running_K_Space = 1

            # check if k_Space relaod is needed
            if self.Reload_K_Space == 1:
                self.Reload_K_Space = 0
                self.Running_K_Space = 0
                return


            sliceMatrix = IMG_vector.copy()
            sliceMatrix = np.squeeze(np.matmul(self.Rx(np.radians(90)),np.expand_dims(IMG_vector,axis=(-1))),axis=(-1))
            GyAngles = np.linspace(0,((360-(360 / (IMG_vector.shape[0]))) * Ky),IMG_vector.shape[0])
            # print("gy rotations:",GyAngles)
            GyAnglesMat = np.array(list(map(lambda theta: [self.Rz(np.radians(theta))],GyAngles))) #to rotate rows
            sliceMatrix = np.squeeze(np.matmul(GyAnglesMat,np.expand_dims(sliceMatrix,axis=(-1))),axis=(-1))
            # GxRotatedMat = GyRotatedMat
            for Kx in range(IMG_vector.shape[1]):
                GxAngles = np.linspace(0,(360-(360 / (IMG_vector.shape[1]))),IMG_vector.shape[1])
                # print("gx rotations:",GxAngles)
                GxAnglesMat = np.array(list(map(lambda theta: self.Rz(np.radians(theta)),GxAngles))) #to rotate cols
                sliceMatrix = np.squeeze(np.matmul(GxAnglesMat,np.expand_dims(sliceMatrix,axis=(-1))),axis=(-1))
                sigmaX = np.sum(sliceMatrix[:, :, 0])
                sigmaY = np.sum(sliceMatrix[:, :, 1]) 
                IMG_K_Space[-Ky,-Kx-1] = complex(sigmaX,sigmaY)

            print(Ky)

            # shift + tab the under section to mke the procecessing way faster as the plotting takes time
            w, h = int(self.figure_Orig_Spat.get_figwidth() * self.figure_Orig_Spat.dpi), int(
                    self.figure_Orig_Spat.get_figheight() * self.figure_Orig_Spat.dpi)

            k_space_magnitude_spectrum = 20 * np.log(abs(np.fft.fftshift(IMG_K_Space)))
            k_space_magnitude_spectrum = cv2.resize(k_space_magnitude_spectrum, (w, h), interpolation=cv2.INTER_AREA)
            self.axis_kspace.imshow(k_space_magnitude_spectrum, cmap='gray')
            self.axis_kspace.set_yticks([])
            self.canvas_kspace.draw()
            # update the reconstructed image for every row added to the K_Space
            IMG_back = np.fft.ifft2(IMG_K_Space)
            abs_img_back = abs(IMG_back)
            abs_img_back = cv2.resize(abs_img_back, (w, h), interpolation=cv2.INTER_AREA)
            self.axis_reconstruct.imshow(abs_img_back, cmap='gray')
            self.canvas_reconstruct.draw()
            # rotMatrix = self.Decay_Recovery_Matrix(rotMatrix,T1,T2,(TR-TE))
        
        self.Running_K_Space = 0
        self.Reload_K_Space = 0
        return
