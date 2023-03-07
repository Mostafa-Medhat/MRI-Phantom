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
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
from scipy.stats import norm


class Phantom(qtw.QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("src/ui/Phantom.ui", self)
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

        # self.sequence_custom_layout()

    def phantom_layout(self):

        ################ Phantom Original Layout #######################
        self.figure_Orig_Spat = Figure(figsize=(20, 20), dpi=100)
        self.axes_Orig_Spat = self.figure_Orig_Spat.add_subplot()
        self.canvas_Orig_Spat = FigureCanvas(self.figure_Orig_Spat)
        self.canvas_Orig_Spat.figure.set_facecolor("#19232D")
        self.verticalLayout_8.addWidget(self.canvas_Orig_Spat)

        self.figure_kspace = Figure(figsize=(20, 20), dpi=100)
        self.axes_kspace = self.figure_kspace.add_subplot()
        self.canvas_kspace = FigureCanvas(self.figure_kspace)
        self.canvas_kspace.figure.set_facecolor("#19232D")
        self.horizontalLayout_3.addWidget(self.canvas_kspace)

        self.figure_reconstruct = Figure(figsize=(20, 20), dpi=100)
        self.axes_reconstruct = self.figure_reconstruct.add_subplot()
        self.canvas_reconstruct = FigureCanvas(self.figure_reconstruct)
        self.canvas_reconstruct.figure.set_facecolor("#19232D")
        self.horizontalLayout_3.addWidget(self.canvas_reconstruct)

        self.axes_phantom = [self.axes_Orig_Spat, self.axes_kspace, self.axes_reconstruct]
        for axis in self.axes_phantom:  ## removing axes from the figure
            axis.set_xticks([])
            axis.set_yticks([])

    def sequence_layout(self, figure, layout):
        ######################## Sequence Layout #########################
        axis_sequence_RF = figure.add_subplot(5, 1, 1)
        axis_sequence_RF.set_ylabel("RF")
        axis_sequence_RF.axes.set_title("Sequence")

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



        ############################ Navy Background ##############################

        # self.canvas_sequence.figure.set_facecolor("#19232D")
        #
        # self.axes_sequence_RF.xaxis.label.set_color('white')
        # self.axes_sequence_RF.yaxis.label.set_color('white')
        # self.axes_sequence_RF.axes.tick_params(axis="x", colors="white")
        # self.axes_sequence_RF.axes.tick_params(axis="y", colors="white")
        # self.axes_sequence_RF.axes.title.set_color('white')
        #
        # self.axis_sequence_SS.xaxis.label.set_color('white')
        # self.axis_sequence_SS.yaxis.label.set_color('white')
        # self.axis_sequence_SS.axes.tick_params(axis="x", colors="white")
        # self.axis_sequence_SS.axes.tick_params(axis="y", colors="white")
        # self.axis_sequence_SS.axes.title.set_color('white')
        #
        # self.axis_sequence_PG.xaxis.label.set_color('white')
        # self.axis_sequence_PG.yaxis.label.set_color('white')
        # self.axis_sequence_PG.axes.tick_params(axis="x", colors="white")
        # self.axis_sequence_PG.axes.tick_params(axis="y", colors="white")
        # self.axis_sequence_PG.axes.title.set_color('white')
        #
        # self.axis_sequence_FG.xaxis.label.set_color('white')
        # self.axis_sequence_FG.yaxis.label.set_color('white')
        # self.axis_sequence_FG.axes.tick_params(axis="x", colors="white")
        # self.axis_sequence_FG.axes.tick_params(axis="y", colors="white")
        # self.axis_sequence_FG.axes.title.set_color('white')
        #
        # self.axis_sequence_RO.xaxis.label.set_color('white')
        # self.axis_sequence_RO.yaxis.label.set_color('white')
        # self.axis_sequence_RO.axes.tick_params(axis="x", colors="white")
        # self.axis_sequence_RO.axes.tick_params(axis="y", colors="white")
        # self.axis_sequence_RO.axes.title.set_color('white')

        #####################################################################################

    def json_read(self):

        path = QFileDialog.getOpenFileName(self, filter="Json files (*.json)")[0]
        if path == "":
            pass
        else:
            dictionary = json.load(open(path))
            self.df = pd.DataFrame(dictionary)
            self.plotting_sequence(self.axes_sequence, self.canvas_sequence, self.df)

    def plotting_sequence(self, axes, canvas, dataFrame):
        colors = ['b', 'g', 'blueviolet', 'orange', 'red']
        for color, axis in enumerate(axes):
            axis.clear()
            axis.plot(np.linspace(0, dataFrame['RO'].Pos + dataFrame['RO'].Duration, 2), np.zeros(shape=2),
                      color=colors[color],
                      linewidth=0.7)

        RF_Duration = np.linspace(-dataFrame['RF'].Duration / 2, dataFrame['RF'].Duration / 2, 100)
        axes[0].plot(RF_Duration + dataFrame['RF'].Duration / 2,
                     dataFrame['RF'].Amp * np.sinc(2 * RF_Duration),
                     color='b')

        #
        SS_Duration = np.linspace(0, dataFrame['SS'].Duration, 100)
        SS_step = dataFrame['RF'].Amp * np.sinc(RF_Duration) > dataFrame['RF'].Amp / 4
        # print(SS_step)
        axes[1].plot(SS_Duration, (dataFrame['SS'].Amp * SS_step), color='g')

        #
        PG_Duration = np.linspace(0, dataFrame['PG'].Duration, 100)
        for i in range(-dataFrame['PG'].Amp, dataFrame['PG'].Amp, 1):
            axes[2].plot(PG_Duration + dataFrame['PG'].Pos, (i * SS_step), color='blueviolet')

        #
        FG_Duration = np.linspace(0, dataFrame['FG'].Duration, 100)
        axes[3].plot(FG_Duration + dataFrame['FG'].Pos, (dataFrame['FG'].Amp * SS_step), color='orange')

        #
        RO_Duration = np.linspace(0, dataFrame['RO'].Duration, 100)
        axes[4].plot(RO_Duration + dataFrame['RO'].Pos, (dataFrame['RO'].Amp * SS_step), color='red')



        #
        canvas.draw()

    def custom_sequence(self):
        print("Entered")
        self.df_custom = self.df.copy()
        print(self.df_custom)
        self.df_custom['RF'].Amp = self.spinBox_RF.value()
        print(self.df_custom)
        # df_cpy['PG'].Amp = self.spinBox_Gradient.value()
        self.plotting_sequence(self.axes_sequence_custom, self.canvas_sequence_custom, self.df_custom)

    def phantom_read(self):
        initial_dir = "\src\docs\phantom images"
        abs_initial_dir = os.path.abspath(initial_dir)
        phantom_path = \
            QFileDialog.getOpenFileName(self, directory=abs_initial_dir, filter="Images files (*.jpg *.jpgs *.png)")[0]
        if phantom_path == "":
            pass
        else:
            img = cv2.imread(phantom_path)
            img = cv2.resize(img, (558, 282), interpolation=cv2.INTER_AREA)
            self.axes_Orig_Spat.imshow(img, cmap='gray')
            self.canvas_Orig_Spat.draw()
