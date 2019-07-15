#!/usr/bin/env python

import sys
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from PIL import Image
import pyqtgraph as pg
import assemble
import align_fm
#import affine_transform
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
        self.resize(1000,800)
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout()
        widget.setLayout(layout)

        # Image views
        splitter_images = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter_images, stretch=1)

        # -- FM Image view
        self.fm_imview = pg.ImageView()
        self.fm_imview.ui.roiBtn.hide()
        self.fm_imview.ui.menuBtn.hide()
        self.fm_imview.setImage(self.data[0])
        self.fm_imview.scene.sigMouseClicked.connect(self._imview_clicked)
        splitter_images.addWidget(self.fm_imview)

        # -- EM Image view
        self.em_imview = pg.ImageView()
        self.em_imview.ui.roiBtn.hide()
        self.em_imview.ui.menuBtn.hide()
        splitter_images.addWidget(self.em_imview)

        # Options
        options = QtWidgets.QHBoxLayout()
        layout.addLayout(options)

        # -- FM options
        vbox = QtWidgets.QVBoxLayout()
        options.addLayout(vbox)

        # ---- Select file
        self.fselector = QtWidgets.QComboBox()
        self.fselector.addItems(self.flist)
        self.fselector.currentIndexChanged.connect(self._file_changed)
        vbox.addWidget(self.fselector)

        # ---- Define and align to grid
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.define_btn = QtWidgets.QPushButton('Define Grid', self)
        self.define_btn.setCheckable(True)
        self.define_btn.toggled.connect(self._define_toggled)
        line.addWidget(self.define_btn)
        self.transform_btn = QtWidgets.QPushButton('Transform image', self)
        self.transform_btn.clicked.connect(self._affine_transform)
        line.addWidget(self.transform_btn)

        # ---- Align colors
        self.align_btn = QtWidgets.QPushButton('Align color channels', self)
        self.align_btn.clicked.connect(self._calc_shift)
        vbox.addWidget(self.align_btn)

        # ---- Flips and rotates
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        self.fliph = QtWidgets.QCheckBox('Flip horizontally', self)
        self.fliph.stateChanged.connect(self._fliph)
        line.addWidget(self.fliph)
        self.flipv = QtWidgets.QCheckBox('Flip vertically', self)
        self.flipv.stateChanged.connect(self._flipv)
        line.addWidget(self.flipv)
        self.transpose = QtWidgets.QCheckBox('Transpose',self)
        self.transpose.stateChanged.connect(self._transpose)
        line.addWidget(self.transpose)
        self.rotate = QtWidgets.QCheckBox('Rotate 90°',self)
        self.rotate.stateChanged.connect(self._rotate)
        line.addWidget(self.rotate)
        vbox.addStretch(1)

        # -- EM options
        vbox = QtWidgets.QVBoxLayout()
        options.addLayout(vbox)

        # ---- Assemble montage
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        label = QtWidgets.QLabel('EM Montage:', self)
        line.addWidget(label)
        self.mrc_fname = QtWidgets.QLineEdit(self)
        self.mrc_fname.returnPressed.connect(self._assemble_mrc)
        line.addWidget(self.mrc_fname, stretch=1)
        button = QtWidgets.QPushButton('Browse', self)
        button.clicked.connect(self._browse_mrc)
        line.addWidget(button)
        button = QtWidgets.QPushButton('Assemble', self)
        button.clicked.connect(self._assemble_mrc)
        line.addWidget(button)

        # ---- Quit button
        vbox.addStretch(1)

        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)
        line.addStretch(1)
        button = QtWidgets.QPushButton('Quit', self)
        button.clicked.connect(self.close)
        line.addWidget(button)

        self.show()

    def _imview_clicked(self, event):
        if event.button() == QtCore.Qt.RightButton:
            event.ignore()
        if self.define_btn.isChecked():
            roi = pg.CircleROI(self.fm_imview.getImageItem().mapFromScene(event.pos()),
                               5,
                               parent=self.fm_imview.getImageItem(),
                               movable=False)
            roi.removeHandle(0)
            self.fm_imview.addItem(roi)
            self.clicked_points.append(roi)
        else:
            pass

    def _define_toggled(self, checked):
        if checked:
            print('Defining grid: Click on corners')
            if self.grid_box is not None:
                self.fm_imview.removeItem(self.grid_box)
                self.grid_box = None
        else:
            print('Done defining grid: Manually adjust fine positions')
            self.grid_box = pg.PolyLineROI([c.pos() for c in self.clicked_points], closed=True, movable=False)
            self.fm_imview.addItem(self.grid_box)
            print(self.grid_box)
            [self.fm_imview.removeItem(roi) for roi in self.clicked_points]
            self.clicked_points = []

    def _file_changed(self, index):
        vr = self.fm_imview.getImageItem().getViewBox().targetRect()
        self.fm_imview.setImage(self.data[index], levels=(self.data[index].min(), self.data[index].mean()*5))
        self.fm_imview.getImageItem().getViewBox().setRange(vr, padding=0)

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
        vr = self.fm_imview.getImageItem().getViewBox().targetRect()
        img = self.fm_imview.getImageItem().image
        if state == QtCore.Qt.Checked:
            self.fm_imview.setImage(np.flipud(img),levels=(img.min(),img.mean()*5))
        else:
            self.fm_imview.setImage(np.flipud(img),levels=(img.min(),img.mean()*5))
        self.fm_imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _flipv(self,state):
        vr = self.fm_imview.getImageItem().getViewBox().targetRect()
        img = self.fm_imview.getImageItem().image
        if state == QtCore.Qt.Checked:
            self.fm_imview.setImage(np.fliplr(img),levels=(img.min(),img.mean()*5))
        else:
            self.fm_imview.setImage(np.fliplr(img),levels=(img.min(),img.mean()*5))
        self.fm_imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _transpose(self,state):
        vr = self.fm_imview.getImageItem().getViewBox().targetRect()
        img = self.fm_imview.getImageItem().image
        if state == QtCore.Qt.Checked:
            self.fm_imview.setImage(img.T,levels=(img.min(),img.mean()*5))
        else:
            self.fm_imview.setImage(img.T,levels=(img.min(),img.mean()*5))
        self.fm_imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _rotate(self,state):
        vr = self.fm_imview.getImageItem().getViewBox().targetRect()
        img = self.fm_imview.getImageItem().image
        if state == QtCore.Qt.Checked:
            self.fm_imview.setImage(np.rot90(img,k=1,axes=(0,1)),levels=(img.min(),img.mean()*5))
        else:
            self.fm_imview.setImage(np.rot90(img,k=1,axes=(1,0)),levels=(img.min(),img.mean()*5))
        self.fm_imview.getImageItem().getViewBox().setRange(vr, padding=0)

    def _browse_mrc(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select .mrc file', os.getcwd() , '*.mrc')
        print(fileName)
        if fileName is not '':
            self.mrc_fname.setText(fileName)

    def _assemble_mrc(self):
        img = assemble.assemble(self.mrc_fname.text())
        print('Done')
        self.em_imview.setImage(img, levels=(img.min(), img.mean()*5))

    def _calc_shift(self):
        print('Align color channels')
        new_list = align_fm.calc_shift(self.flist)

        self.fselector.addItems(new_list)
        self.fselector.currentIndexChanged.connect(self._file_changed)

        data_shifted = [np.array(Image.open(fname)) for fname in new_list]
        self.data = np.concatenate((self.data,data_shifted),axis=0)

        self.align_btn.setEnabled(False)

    def _affine_transform(self):
        if self.grid_box is not None:
            print('Perform affine transformation')
            print(self.grid_box.getState())
            points = self.grid_box.getState()['points']
            my_points = [list((point[0],point[1])) for point in points]
            transformed_img = affine_transform.calc_transform(my_points)

        else:
            print('Define grid box first!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CLEM GUI')
    parser.add_argument('files', help='List of files to process', nargs='+')
    args = parser.parse_args()
    print(args.files)

    app = QtWidgets.QApplication([])
    gui = GUI(args.files)
    sys.exit(app.exec_())
