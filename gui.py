#!/usr/bin/env python

import sys
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PIL import Image
import pyqtgraph as pg
import assemble
import align_fm

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
        self.resize(1000,400)
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        hbox = QtWidgets.QHBoxLayout()
        vbox1 = QtWidgets.QVBoxLayout()
        vbox2 = QtWidgets.QVBoxLayout()

        #self.imview = pg.ImageView(view=pg.PlotItem())
        self.imview = pg.ImageView()
        self.imview.ui.roiBtn.hide()
        self.imview.ui.menuBtn.hide()
        self.imview.setImage(self.data[0])
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)
        vbox1.addWidget(self.imview,stretch=1)
        
        self.imview2 = pg.ImageView()
        self.imview2.ui.roiBtn.hide()
        self.imview2.ui.menuBtn.hide()
        vbox2.addWidget(self.imview2,stretch=1)
        	
        self.fselector = QtWidgets.QComboBox()
        self.fselector.addItems(self.flist)
        self.fselector.currentIndexChanged.connect(self._file_changed)
        vbox1.addWidget(self.fselector)
        
        self.define_btn = QtWidgets.QPushButton('Define Grid', self)
        self.define_btn.setCheckable(True)
        self.define_btn.toggled.connect(self._define_toggled)
        vbox1.addWidget(self.define_btn)
        
        self.align_btn = QtWidgets.QPushButton('Align color channels', self)
        self.align_btn.clicked.connect(self._calc_shift)
        vbox1.addWidget(self.align_btn)
        
        self.assemble_btn = QtWidgets.QPushButton('Assemble EM grid', self)
        self.assemble_btn.clicked.connect(self._load_assemble)
        #self.assemble_btn.toggled.connect(self._assemble_toggled)
        vbox2.addWidget(self.assemble_btn)
     
        #vbox2.addStretch(1)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        vbox2.addWidget(button)
        #vbox2.addStretch(1)
        
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        widget.setLayout(hbox)

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
        vr = self.imview.getImageItem().getViewBox().targetRect()
        self.imview.setImage(self.data[index], levels=(self.data[index].min(), self.data[index].mean()*5))
        self.imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _next_file(self):
        ind = (self.fselector.currentIndex() + 1) % self.fselector.count()
        self.fselector.setCurrentIndex(ind)

    def _prev_file(self):
        ind = (self.fselector.currentIndex() - 1 + self.fselector.count()) % self.fselector.count()
        self.fselector.setCurrentIndex(ind)

    def keyPressEvent(self, event):
        key = event.key()
        mod = int(event.modifiers())

        if QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+P'):
            self._prev_file()
        elif QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+N'):
            self._next_file()
        elif QtGui.QKeySequence(mod+key) == QtGui.QKeySequence('Ctrl+W'):
            self.close()
        else:
            event.ignore()
    
    def _load_assemble(self):
        print('Assemble .mrc file')
        img = assemble.assemble()
        print('Done')
        #self.imview.setImage(img)[::10,::10]
       
        self.imview2.setImage(img[::10,::10], levels=(img.min(), img.mean()*5))
        #vr2 = self.imview2.getImageItem().getViewBox().targetRect()
        #self.imview2.getImageItem().getViewBox().setRange(vr2, padding=0)

        if img is not None:
            self.assemble_btn.setEnabled(False)
        #    self.assemble_btn = QtWidgets.QPushButton('Show assembled EM image', self)
        #    self.assemble_btn.setCheckable(False)
        #    self.assemble_btn.clicked.connect(self._file_changed)
        #    #self.assemble_btn.toggled.connect(self._assemble_toggled)
        #    vbox.addWidget(self.assemble_btn) 


    def _calc_shift(self):
        print('Align color channels')
        #ref,shift1,shift2,coordinates = align_fm.calc_shift(data)
    
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CLEM GUI')
    parser.add_argument('files', help='List of files to process', nargs='+')
    args = parser.parse_args()
    print(args.files)
    
    app = QtWidgets.QApplication([])
    gui = GUI(args.files)
    sys.exit(app.exec_())
