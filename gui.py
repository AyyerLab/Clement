#!/usr/bin/env python

import sys
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PIL import Image
import pyqtgraph as pg
import assemble
import align_fm
import os

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
           
        splitter_images = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        hbox = QtWidgets.QHBoxLayout()
        vbox1 = QtWidgets.QVBoxLayout()
        vbox2 = QtWidgets.QVBoxLayout()
        hbox2 = QtWidgets.QHBoxLayout()
        hbox_lower = QtWidgets.QHBoxLayout()

        self.imview = pg.ImageView()
        self.imview.ui.roiBtn.hide()
        self.imview.ui.menuBtn.hide()
        self.imview.setImage(self.data[0])
        self.imview.scene.sigMouseClicked.connect(self._imview_clicked)
        splitter_images.addWidget(self.imview)
      
        self.imview2 = pg.ImageView()
        self.imview2.ui.roiBtn.hide()
        self.imview2.ui.menuBtn.hide()
        splitter_images.addWidget(self.imview2)
        
        splitter_im_text = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter_im_text.setSizes([100,200])
        splitter_im_text.addWidget(splitter_images)
        
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
        
        self.fliph = QtWidgets.QCheckBox('Flip horizontally', self)
        self.fliph.stateChanged.connect(self._fliph)
        hbox2.addWidget(self.fliph) 
        self.flipv = QtWidgets.QCheckBox('Flip vertically', self)
        self.flipv.stateChanged.connect(self._flipv)
        hbox2.addWidget(self.flipv)
        self.transpose = QtWidgets.QCheckBox('Transpose',self)
        self.transpose.stateChanged.connect(self._transpose)
        hbox2.addWidget(self.transpose)
        self.rotate = QtWidgets.QCheckBox('Rotate 90°',self)
        self.rotate.stateChanged.connect(self._rotate)
        hbox2.addWidget(self.rotate)
    
        vbox1.addLayout(hbox2)       
    
        self.assemble_btn = QtWidgets.QPushButton('Select and assemble .mrc file', self)
        self.assemble_btn.clicked.connect(self._load_assemble)
        
        vbox2.addWidget(self.assemble_btn)
        
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        vbox2.addWidget(button)     
        
        hbox_lower.addLayout(vbox1)
        hbox_lower.addLayout(vbox2)
        lower_widget = QtWidgets.QWidget()
        lower_widget.setLayout(hbox_lower) 

        splitter_im_text.addWidget(lower_widget)
        hbox.addWidget(splitter_im_text)
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
    
    def _fliph(self,state):
        vr = self.imview.getImageItem().getViewBox().targetRect()
        img = self.imview.getImageItem().image
        if state == QtCore.Qt.Checked:
            self.imview.setImage(np.flipud(img),levels=(img.min(),img.mean()*5))
        else:
            self.imview.setImage(np.flipud(img),levels=(img.min(),img.mean()*5))
        self.imview.getImageItem().getViewBox().setRange(vr, padding=0)
 
    def _flipv(self,state):
        vr = self.imview.getImageItem().getViewBox().targetRect()
        img = self.imview.getImageItem().image
        if state == QtCore.Qt.Checked:
            self.imview.setImage(np.fliplr(img),levels=(img.min(),img.mean()*5))
        else:
            self.imview.setImage(np.fliplr(img),levels=(img.min(),img.mean()*5))       
        self.imview.getImageItem().getViewBox().setRange(vr, padding=0)
    
    def _transpose(self,state):
        vr = self.imview.getImageItem().getViewBox().targetRect()
        img = self.imview.getImageItem().image
        if state == QtCore.Qt.Checked:
            self.imview.setImage(img.T,levels=(img.min(),img.mean()*5))
        else:
            self.imview.setImage(img.T,levels=(img.min(),img.mean()*5))
        self.imview.getImageItem().getViewBox().setRange(vr, padding=0)
    
    def _rotate(self,state):
        vr = self.imview.getImageItem().getViewBox().targetRect()
        img = self.imview.getImageItem().image
        if state == QtCore.Qt.Checked:
            self.imview.setImage(np.rot90(img,k=1,axes=(0,1)),levels=(img.min(),img.mean()*5))
        else:
            self.imview.setImage(np.rot90(img,k=1,axes=(1,0)),levels=(img.min(),img.mean()*5))    
        self.imview.getImageItem().getViewBox().setRange(vr, padding=0)
    
    def _load_assemble(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select .mrc file', os.getcwd() , '*.mrc')
        print(fileName)
        print('Assemble .mrc file')
        img = assemble.assemble(fileName)
        print('Done')
        self.imview2.setImage(img, levels=(img.min(), img.mean()*5))

        if img is not None:
            self.assemble_btn.setEnabled(False)

    def _calc_shift(self):
        print('Align color channels')
        new_list = align_fm.calc_shift(self.flist)
        
        self.fselector.addItems(new_list)
        self.fselector.currentIndexChanged.connect(self._file_changed)
        
        data_shifted = [np.array(Image.open(fname)) for fname in new_list]
        self.data = np.concatenate((self.data,data_shifted),axis=0)
    
        self.align_btn.setEnabled(False)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CLEM GUI')
    parser.add_argument('files', help='List of files to process', nargs='+')
    args = parser.parse_args()
    print(args.files)
    
    app = QtWidgets.QApplication([])
    gui = GUI(args.files)
    sys.exit(app.exec_())
