#!/usr/bin/env python

import sys
import os
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from em_controls import EMControls
from fm_controls import FMControls
import fm_operations # To kill javabridge on exit
import align_fm

warnings.simplefilter('ignore', category=FutureWarning)

class GUI(QtGui.QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        self.ind = 0
         
        self.colors = ['#ff0000', '#00ff00', '#0000ff', '#808080']
        self.color_data = None
        self.overlay = True
         
        self.box_points = []
        self.tr_box_points = []
        self.box_coordinate = None
        
        self.refine = False
        self.settings = QtCore.QSettings('MPSD-CNI', 'CLEMGui', self)
        self.colors = self.settings.value('channel_colors', defaultValue=['#ff0000', '#00ff00', '#0000ff', '#808080'])
        self._init_ui()

    def _init_ui(self):
        geom = self.settings.value('geometry')
        if geom is None:
            self.resize(1600, 800)
        else:
            self.setGeometry(geom)
        self.theme = self.settings.value('theme')
        if self.theme is None:
            self.theme = 'none'
        self._set_theme(self.theme)

        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        # Image views
        splitter_images = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter_images, stretch=1)

        # -- FM Image view
        self.fm_imview = pg.ImageView()
        self.fm_imview.ui.roiBtn.hide()
        self.fm_imview.ui.menuBtn.hide()
        splitter_images.addWidget(self.fm_imview)

        # -- EM Image view
        self.em_imview = pg.ImageView()
        self.em_imview.ui.roiBtn.hide()
        self.em_imview.ui.menuBtn.hide()
        splitter_images.addWidget(self.em_imview)

        # Options
        options = QtWidgets.QHBoxLayout()
        options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(options)

        self.fmcontrols = FMControls(self.fm_imview, self.colors)
        options.addWidget(self.fmcontrols)
        self.emcontrols = EMControls(self.em_imview)
        options.addWidget(self.emcontrols)

        # Connect controllers
        self.emcontrols.quit_button.clicked.connect(self.close)
        self.emcontrols.other = self.fmcontrols
        self.fmcontrols.other = self.emcontrols

        # Menu Bar
        self._init_menubar()
        self.show()

    def _init_menubar(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        # -- File menu
        filemenu = menubar.addMenu('&File')
        action = QtWidgets.QAction('Load &FM image(s)', self)
        action.triggered.connect(self.fmcontrols._load_fm_images)
        filemenu.addAction(action)
        action = QtWidgets.QAction('Load &EM montage', self)
        action.triggered.connect(self.emcontrols._load_mrc)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Save binned montage', self)
        action.triggered.connect(self.emcontrols._save_mrc_montage)
        filemenu.addAction(action)
        action = QtWidgets.QAction('&Quit', self)
        action.triggered.connect(self.close)
        filemenu.addAction(action)

        # -- Theme menu
        thememenu = menubar.addMenu('&Theme')
        agroup = QtWidgets.QActionGroup(self)
        agroup.setExclusive(True)
        action = QtWidgets.QAction('None', self)
        action.triggered.connect(lambda: self._set_theme('none'))
        thememenu.addAction(action)
        agroup.addAction(action)
        action = QtWidgets.QAction('Dark', self)
        action.triggered.connect(lambda: self._set_theme('dark'))
        thememenu.addAction(action)
        agroup.addAction(action)

        self.show()

    def _set_theme(self, name):
        if name == 'none':
            self.setStyleSheet('')
        else:
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'styles/%s.css'%name), 'r') as f:
                self.setStyleSheet(f.read())
        self.settings.setValue('theme', name)

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

    def closeEvent(self, event):
        fm_operations.javabridge.kill_vm()
        self.settings.setValue('channel_colors', self.colors)
        self.settings.setValue('geometry', self.geometry())
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    gui = GUI()
    sys.exit(app.exec_())
