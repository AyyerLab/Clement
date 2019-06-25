#!/usr/bin/env python

import sys
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PIL import Image
import pyqtgraph as pg

warnings.simplefilter('ignore', category=FutureWarning)

class GUI(QtGui.QMainWindow):
    def __init__(self, flist):
        super(GUI, self).__init__()
        self.flist = flist
        self.data = [np.array(Image.open(fname)) for fname in flist]
        self.clicked_points = []
        self.grid_box = None
        self._init_ui()

    def _init_ui(self):
        self.resize(800,600)
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QHBoxLayout()
        widget.setLayout(layout)

        #self.imview = pg.ImageView(view=pg.PlotItem())
        self.imview = pg.ImageView()
        self.imview.ui.roiBtn.hide()
        self.imview.ui.menuBtn.hide()
        self.imview.setImage(self.data[0])
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)
        layout.addWidget(self.imview, stretch=1)
        
        vbox = QtWidgets.QVBoxLayout()
        layout.addLayout(vbox)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        cbox = QtWidgets.QComboBox()
        cbox.addItems(self.flist)
        cbox.currentIndexChanged.connect(self._file_changed)
        line.addWidget(cbox)
        
        self.define_btn = QtWidgets.QPushButton('Define Grid', self)
        self.define_btn.setCheckable(True)
        self.define_btn.toggled.connect(self._define_toggled)
        vbox.addWidget(self.define_btn)

        vbox.addStretch(1)
        
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        line.addWidget(button)
        line.addStretch(1)

        self.show()

    def _imview_clicked(self, event):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
        if self.define_btn.isChecked():
            roi = pg.CircleROI(self.imview.getImageItem().mapFromScene(event.pos()),
                               5, 
                               parent=self.imview.getImageItem(),
                               movable=False)
            roi.removeHandle(0)
            self.imview.addItem(roi)
            self.clicked_points.append(roi)
        else:
            pass

    def _define_toggled(self, checked):
        if checked:
            print('Defining grid: Click on corners')
            if self.grid_box is not None:
                self.imview.removeItem(self.grid_box)
                self.grid_box = None
        else:
            print('Done defining grid: Manually adjust fine positions')
            self.grid_box = pg.PolyLineROI([c.pos() for c in self.clicked_points], closed=True, movable=False)
            self.imview.addItem(self.grid_box)
            [self.imview.removeItem(roi) for roi in self.clicked_points]
            self.clicked_points = []

    def _file_changed(self, index):
        self.imview.setImage(self.data[index], levels=(self.data[index].min(), self.data[index].mean()*5))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CLEM GUI')
    parser.add_argument('files', help='List of files to process', nargs='+')
    args = parser.parse_args()
    print(args.files)
    
    app = QtWidgets.QApplication([])
    gui = GUI(args.files)
    sys.exit(app.exec_())
