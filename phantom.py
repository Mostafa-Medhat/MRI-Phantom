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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import norm


class Phantom(qtw.QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("src/ui/Phantom.ui", self)

        self.figure_sequence = Figure(dpi=75)
        self.axes_sequence_RF = self.figure_sequence.add_subplot(5, 1, 1)
        self.axes_sequence_RF.set_ylabel("RF")
        self.axes_sequence_RF.set_frame_on(False)
        self.axes_sequence_RF.axes.get_xaxis().set_visible(False)
        self.axes_sequence_RF.axes.set_title("Sequence")

        self.axes_sequence_SS = self.figure_sequence.add_subplot(5, 1, 2)
        self.axes_sequence_SS.set_ylabel("SS")
        self.axes_sequence_SS.set_frame_on(False)
        self.axes_sequence_SS.axes.get_xaxis().set_visible(False)

        self.axes_sequence_PG = self.figure_sequence.add_subplot(5, 1, 3)
        self.axes_sequence_PG.set_ylabel("PG")
        self.axes_sequence_PG.set_frame_on(False)
        self.axes_sequence_PG.axes.get_xaxis().set_visible(False)

        self.axes_sequence_FG = self.figure_sequence.add_subplot(5, 1, 4)
        self.axes_sequence_FG.set_ylabel("FG")
        self.axes_sequence_FG.set_frame_on(False)
        self.axes_sequence_FG.axes.get_xaxis().set_visible(False)

        self.axes_sequence_RO = self.figure_sequence.add_subplot(5, 1, 5)
        self.axes_sequence_RO.set_ylabel("RO")
        self.axes_sequence_RO.set_frame_on(False)

        self.canvas_sequence = FigureCanvas(self.figure_sequence)
        self.figure_sequence.subplots_adjust(hspace=0.5)
        # self.canvas_sequence.figure.tight_layout()
        self.verticalLayout_4.addWidget(self.canvas_sequence)


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

        self.Json_Read()

    def Json_Read(self):
        dictionary = json.load(open('seq.json'))
        # print(type(dictionary))
        df = pd.DataFrame(dictionary)
        print(df['PG'].Pos)

        # fig, axs = plt.subplots(5, sharex=True)
        # # Remove horizontal space between axes
        # fig.subplots_adjust(hspace=0.5)
        # self.axes_sequence.subplots(sharex=True)

        RF_Duration = np.linspace(-df['RF'].Duration / 2, df['RF'].Duration / 2, 100)
        self.axes_sequence_RF.plot(RF_Duration + df['RF'].Duration / 2, df['RF'].Amp * np.sinc(2 * RF_Duration), color='b')
        self.axes_sequence_RF.plot(np.linspace(0, df['RO'].Pos + df['RO'].Duration, 2), np.zeros(shape=2), color='b',
                                   linewidth=0.7)



        SS_Duration = np.linspace(0, df['SS'].Duration, 100)
        SS_step = df['RF'].Amp * np.sinc(RF_Duration) > df['RF'].Amp / 6
        # print(SS_step)
        self.axes_sequence_SS.set_ylabel("SS")
        self.axes_sequence_SS.plot(SS_Duration, (df['SS'].Amp * SS_step), color='g')
        self.axes_sequence_SS.plot(np.linspace(0, df['RO'].Pos + df['RO'].Duration, 2), np.zeros(shape=2), color='g', linewidth=0.7)

        #


        PG_Duration = np.linspace(0, df['PG'].Duration, 100)
        PG_step = df['RF'].Amp * np.sinc(RF_Duration) > df['RF'].Amp / 6
        for i in range(-df['PG'].Amp, df['PG'].Amp, 1):
            self.axes_sequence_PG.plot(PG_Duration + df['PG'].Pos, (i * PG_step), color='blueviolet')

        self.axes_sequence_PG.plot(np.linspace(0, df['RO'].Pos + df['RO'].Duration, 2), np.zeros(shape=2), color='blueviolet',
                    linewidth=0.7)

        #


        FG_Duration = np.linspace(0, df['FG'].Duration, 100)
        FG_step = df['RF'].Amp * np.sinc(RF_Duration) > df['RF'].Amp / 6
        self.axes_sequence_FG.plot(FG_Duration + df['FG'].Pos, (df['FG'].Amp * FG_step), color='orange')
        self.axes_sequence_FG.plot(np.linspace(0, df['RO'].Pos + df['RO'].Duration, 2), np.zeros(shape=2), color='orange', linewidth=0.7)

        #
        RO_Duration = np.linspace(0, df['RO'].Duration, 100)
        RO_step = df['RF'].Amp * np.sinc(RF_Duration) > df['RF'].Amp / 6
        self.axes_sequence_RO.plot(RO_Duration + df['RO'].Pos, (df['RO'].Amp * RO_step), color='red')
        self.axes_sequence_RO.plot(np.linspace(0, df['RO'].Pos + df['RO'].Duration, 2), np.zeros(shape=2), color='red', linewidth=0.7)

        #
        self.canvas_sequence.draw()
