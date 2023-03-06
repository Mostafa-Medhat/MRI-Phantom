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
        self.phantom_layout()
        self.sequence_layout()
        self.sequence_custom_layout()

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
        for axis in self.axes_phantom:  ## removing axes from the figure so the image would look nice
            axis.set_xticks([])
            axis.set_yticks([])

    def sequence_layout(self):
        ######################## Sequence Layout #########################
        self.figure_sequence = Figure(dpi=80)
        self.axes_sequence_RF = self.figure_sequence.add_subplot(5, 1, 1)
        self.axes_sequence_RF.set_ylabel("RF")
        self.axes_sequence_RF.axes.set_title("Sequence")

        self.axes_sequence_SS = self.figure_sequence.add_subplot(5, 1, 2, sharex=self.axes_sequence_RF)
        self.axes_sequence_SS.set_ylabel("SS")


        self.axes_sequence_PG = self.figure_sequence.add_subplot(5, 1, 3, sharex=self.axes_sequence_RF)
        self.axes_sequence_PG.set_ylabel("PG")


        self.axes_sequence_FG = self.figure_sequence.add_subplot(5, 1, 4, sharex=self.axes_sequence_RF)
        self.axes_sequence_FG.set_ylabel("FG")


        self.axes_sequence_RO = self.figure_sequence.add_subplot(5, 1, 5, sharex=self.axes_sequence_RF)
        self.axes_sequence_RO.set_ylabel("RO")
        self.axes_sequence_RO.set_frame_on(False)

        self.canvas_sequence = FigureCanvas(self.figure_sequence)
        self.figure_sequence.subplots_adjust(hspace=0.5)
        # self.toolbar = NavigationToolbar(self.canvas_sequence, self)

        # self.canvas_sequence.figure.tight_layout()

        self.verticalLayout_4.addWidget(self.canvas_sequence)

        self.axes_sequence = [self.axes_sequence_RF, self.axes_sequence_SS, self.axes_sequence_PG,
                              self.axes_sequence_FG]
        for axis in self.axes_sequence:  ## removing axes from the figure so the image would look nice
            axis.set_frame_on(False)
            axis.axes.get_xaxis().set_visible(False)


    def sequence_custom_layout(self):
        ######################## Sequence Custom Layout #########################
        self.figure_sequence_custom = Figure(dpi=80)
        self.axes_sequence_custom_RF = self.figure_sequence_custom.add_subplot(5, 1, 1)
        self.axes_sequence_custom_RF.set_ylabel("RF")
        self.axes_sequence_custom_RF.axes.set_title("Sequence Custom")

        self.axes_sequence_custom_SS = self.figure_sequence_custom.add_subplot(5, 1, 2,
                                                                               sharex=self.axes_sequence_custom_RF)
        self.axes_sequence_custom_SS.set_ylabel("SS")

        self.axes_sequence_custom_PG = self.figure_sequence_custom.add_subplot(5, 1, 3,
                                                                               sharex=self.axes_sequence_custom_RF)
        self.axes_sequence_custom_PG.set_ylabel("PG")


        self.axes_sequence_custom_FG = self.figure_sequence_custom.add_subplot(5, 1, 4,
                                                                               sharex=self.axes_sequence_custom_RF)
        self.axes_sequence_custom_FG.set_ylabel("FG")

        self.axes_sequence_custom_RO = self.figure_sequence_custom.add_subplot(5, 1, 5,
                                                                               sharex=self.axes_sequence_custom_RF)
        self.axes_sequence_custom_RO.set_ylabel("RO")
        self.axes_sequence_custom_RO.set_frame_on(False)

        self.canvas_sequence_custom = FigureCanvas(self.figure_sequence_custom)
        self.figure_sequence_custom.subplots_adjust(hspace=0.5)

        # self.toolbar = NavigationToolbar(self.canvas_sequence_custom, self)
        # self.canvas_sequence_custom.figure.tight_layout()

        self.verticalLayout_6.addWidget(self.canvas_sequence_custom)

        self.axes_sequence_custom = [self.axes_sequence_custom_RF, self.axes_sequence_custom_SS,
                                     self.axes_sequence_custom_PG,
                                     self.axes_sequence_custom_FG]
        for axis in self.axes_sequence_custom:  ## removing axes from the figure so the image would look nice
            axis.set_frame_on(False)
            axis.axes.get_xaxis().set_visible(False)

        ############################ Navy Background ##############################

        # self.canvas_sequence.figure.set_facecolor("#19232D")
        #
        # self.axes_sequence_RF.xaxis.label.set_color('white')
        # self.axes_sequence_RF.yaxis.label.set_color('white')
        # self.axes_sequence_RF.axes.tick_params(axis="x", colors="white")
        # self.axes_sequence_RF.axes.tick_params(axis="y", colors="white")
        # self.axes_sequence_RF.axes.title.set_color('white')
        #
        # self.axes_sequence_SS.xaxis.label.set_color('white')
        # self.axes_sequence_SS.yaxis.label.set_color('white')
        # self.axes_sequence_SS.axes.tick_params(axis="x", colors="white")
        # self.axes_sequence_SS.axes.tick_params(axis="y", colors="white")
        # self.axes_sequence_SS.axes.title.set_color('white')
        #
        # self.axes_sequence_PG.xaxis.label.set_color('white')
        # self.axes_sequence_PG.yaxis.label.set_color('white')
        # self.axes_sequence_PG.axes.tick_params(axis="x", colors="white")
        # self.axes_sequence_PG.axes.tick_params(axis="y", colors="white")
        # self.axes_sequence_PG.axes.title.set_color('white')
        #
        # self.axes_sequence_FG.xaxis.label.set_color('white')
        # self.axes_sequence_FG.yaxis.label.set_color('white')
        # self.axes_sequence_FG.axes.tick_params(axis="x", colors="white")
        # self.axes_sequence_FG.axes.tick_params(axis="y", colors="white")
        # self.axes_sequence_FG.axes.title.set_color('white')
        #
        # self.axes_sequence_RO.xaxis.label.set_color('white')
        # self.axes_sequence_RO.yaxis.label.set_color('white')
        # self.axes_sequence_RO.axes.tick_params(axis="x", colors="white")
        # self.axes_sequence_RO.axes.tick_params(axis="y", colors="white")
        # self.axes_sequence_RO.axes.title.set_color('white')

        #####################################################################################



    def json_read(self):

        path = QFileDialog.getOpenFileName(self, filter="Json files (*.json)")[0]
        if path == "":
            pass
        else:
            dictionary = json.load(open(path))
            # print(type(dictionary))
            self.df = pd.DataFrame(dictionary)

            RF_Duration = np.linspace(-self.df['RF'].Duration / 2, self.df['RF'].Duration / 2, 100)
            self.axes_sequence_RF.plot(RF_Duration + self.df['RF'].Duration / 2, self.df['RF'].Amp * np.sinc(2 * RF_Duration),
                                       color='b')
            self.axes_sequence_RF.plot(np.linspace(0, self.df['RO'].Pos + self.df['RO'].Duration, 2), np.zeros(shape=2), color='b',
                                       linewidth=0.7)


            SS_Duration = np.linspace(0, self.df['SS'].Duration, 100)
            SS_step = self.df['RF'].Amp * np.sinc(RF_Duration) > self.df['RF'].Amp / 4
            # print(SS_step)
            self.axes_sequence_SS.set_ylabel("SS")
            self.axes_sequence_SS.plot(SS_Duration, (self.df['SS'].Amp * SS_step), color='g')
            self.axes_sequence_SS.plot(np.linspace(0, self.df['RO'].Pos + self.df['RO'].Duration, 2), np.zeros(shape=2), color='g',
                                       linewidth=0.7)

            #

            PG_Duration = np.linspace(0, self.df['PG'].Duration, 100)
            PG_step = self.df['RF'].Amp * np.sinc(RF_Duration) > self.df['RF'].Amp / 4
            for i in range(-self.df['PG'].Amp, self.df['PG'].Amp, 1):
                self.axes_sequence_PG.plot(PG_Duration + self.df['PG'].Pos, (i * PG_step), color='blueviolet')

            self.axes_sequence_PG.plot(np.linspace(0, self.df['RO'].Pos + self.df['RO'].Duration, 2), np.zeros(shape=2),
                                       color='blueviolet',
                                       linewidth=0.7)

            #
            FG_Duration = np.linspace(0, self.df['FG'].Duration, 100)
            FG_step = self.df['RF'].Amp * np.sinc(RF_Duration) > self.df['RF'].Amp / 4
            self.axes_sequence_FG.plot(FG_Duration + self.df['FG'].Pos, (self.df['FG'].Amp * FG_step), color='orange')
            self.axes_sequence_FG.plot(np.linspace(0, self.df['RO'].Pos + self.df['RO'].Duration, 2), np.zeros(shape=2),
                                       color='orange', linewidth=0.7)

            #
            RO_Duration = np.linspace(0, self.df['RO'].Duration, 100)
            RO_step = self.df['RF'].Amp * np.sinc(RF_Duration) > self.df['RF'].Amp / 4
            self.axes_sequence_RO.plot(RO_Duration + self.df['RO'].Pos, (self.df['RO'].Amp * RO_step), color='red')
            self.axes_sequence_RO.plot(np.linspace(0, self.df['RO'].Pos + self.df['RO'].Duration, 2), np.zeros(shape=2), color='red',
                                       linewidth=0.7)

            #
            self.canvas_sequence.draw()

    def custom_sequence(self):
        print("Entered")
        df_custom = self.df.copy()
        print(df_custom)
        df_custom['RF'].Amp = self.spinBox_RF.value()
        # df_cpy['PG'].Amp = self.spinBox_Gradient.value()

        print(df_custom['RF'].Amp)
        print(df_custom)

        RF_Duration = np.linspace(-df_custom['RF'].Duration / 2, df_custom['RF'].Duration / 2, 100)
        self.axes_sequence_custom_RF.plot(RF_Duration + df_custom['RF'].Duration / 2, df_custom['RF'].Amp * np.sinc(2 * RF_Duration),
                                   color='b')
        self.axes_sequence_custom_RF.plot(np.linspace(0, df_custom['RO'].Pos + df_custom['RO'].Duration, 2), np.zeros(shape=2), color='b',
                                   linewidth=0.7)

        SS_Duration = np.linspace(0, df_custom['SS'].Duration, 100)
        SS_step = df_custom['RF'].Amp * np.sinc(RF_Duration) > df_custom['RF'].Amp / 4
        # print(SS_step)
        self.axes_sequence_custom_SS.set_ylabel("SS")
        self.axes_sequence_custom_SS.plot(SS_Duration, (df_custom['SS'].Amp * SS_step), color='g')
        self.axes_sequence_custom_SS.plot(np.linspace(0, df_custom['RO'].Pos + df_custom['RO'].Duration, 2), np.zeros(shape=2), color='g',
                                   linewidth=0.7)

        #

        PG_Duration = np.linspace(0, df_custom['PG'].Duration, 100)
        PG_step = df_custom['RF'].Amp * np.sinc(RF_Duration) > df_custom['RF'].Amp / 4
        for i in range(-df_custom['PG'].Amp, df_custom['PG'].Amp, 1):
            self.axes_sequence_custom_PG.plot(PG_Duration + df_custom['PG'].Pos, (i * PG_step), color='blueviolet')

        self.axes_sequence_custom_PG.plot(np.linspace(0, df_custom['RO'].Pos + df_custom['RO'].Duration, 2), np.zeros(shape=2),
                                   color='blueviolet',
                                   linewidth=0.7)

        #
        FG_Duration = np.linspace(0, df_custom['FG'].Duration, 100)
        FG_step = df_custom['RF'].Amp * np.sinc(RF_Duration) > df_custom['RF'].Amp / 4
        self.axes_sequence_custom_FG.plot(FG_Duration + df_custom['FG'].Pos, (df_custom['FG'].Amp * FG_step), color='orange')
        self.axes_sequence_custom_FG.plot(np.linspace(0, df_custom['RO'].Pos + df_custom['RO'].Duration, 2), np.zeros(shape=2),
                                   color='orange', linewidth=0.7)

        #
        RO_Duration = np.linspace(0, df_custom['RO'].Duration, 100)
        RO_step = df_custom['RF'].Amp * np.sinc(RF_Duration) > df_custom['RF'].Amp / 4
        self.axes_sequence_custom_RO.plot(RO_Duration + df_custom['RO'].Pos, (df_custom['RO'].Amp * RO_step), color='red')
        self.axes_sequence_custom_RO.plot(np.linspace(0, df_custom['RO'].Pos + df_custom['RO'].Duration, 2), np.zeros(shape=2), color='red',
                                   linewidth=0.7)

        #
        self.canvas_sequence_custom.draw()



    def phantom_read(self):
        initial_dir = "\src\docs\phantom images"
        abs_initial_dir = os.path.abspath(initial_dir)
        phantom_path = QFileDialog.getOpenFileName(self, directory=abs_initial_dir, filter="Images files (*.jpg *.jpgs *.png)")[0]
        if phantom_path == "":
            pass
        else:
            img = cv2.imread(phantom_path)
            img = cv2.resize(img,(558, 282), interpolation=cv2.INTER_AREA)
            self.axes_Orig_Spat.imshow(img, cmap='gray')
            self.canvas_Orig_Spat.draw()
