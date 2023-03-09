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
import numpy as np
import qdarkstyle
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QFileDialog, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
from scipy.stats import norm


class Phantom(qtw.QWidget):
    def __init__(self):
        super().__init__()

        uic.loadUi("src/ui/Phantom.ui", self)
        # self.splitter.setWidth(50, 1)

        self.df = None
        self.df_custom = None

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

        self.pushButton_openSequence.clicked.connect(lambda: self.json_read())
        self.pushButton_apply.clicked.connect(lambda: self.custom_sequence())
        self.pushButton_clear.clicked.connect(lambda: self.clear_all())
        self.pushButton_openPhantom.clicked.connect(lambda: self.phantom_read())

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

    def json_read(self):
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

    def phantom_read(self):
        phantom_path = QFileDialog.getOpenFileName(self, "Open File", "src/docs/phantom images", filter="Images files ("
                                                                                                        "*.jpg *.jpeg "
                                                                                                        "*.png)")[0]
        if phantom_path != "":
            img = cv2.imread(phantom_path, cv2.IMREAD_GRAYSCALE)
            w, h = int(self.figure_Orig_Spat.get_figwidth() * self.figure_Orig_Spat.dpi), int(
                self.figure_Orig_Spat.get_figheight() * self.figure_Orig_Spat.dpi)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            self.axis_Orig_Spat.imshow(img, cmap='gray')
            self.canvas_Orig_Spat.draw()

            # Compute the Fourier Transform of the image
            f = np.fft.fft2(img)

            # Shift the zero-frequency component to the center of the spectrum
            fshift = np.fft.fftshift(f)

            # Compute the magnitude spectrum of the Fourier Transform
            magnitude_spectrum = 20 * np.log(np.abs(fshift))

            self.axis_Orig_Fourier.imshow(magnitude_spectrum, cmap='gray')
            self.canvas_Orig_Fourier.draw()

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
        self.canvas_kspace.figure.set_facecolor('black')
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
