#!/usr/bin/env python

import sys
import os
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from . import res_styles
from .em_controls import EMControls
from .fm_controls import FMControls
from .popup import Merge

warnings.simplefilter('ignore', category=FutureWarning)

def resource_path(rel_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, rel_path)

class GUI(QtGui.QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        self.settings = QtCore.QSettings('MPSD-CNI', 'CLEMGui', self)
        self.colors = self.settings.value('channel_colors', defaultValue=['#ff0000', '#00ff00', '#0000ff', '#808080'])
        self._init_ui()

    def _init_ui(self):
        geom = self.settings.value('geometry')
        if geom is None:
            self.resize(1600, 800)
        else:
            self.setGeometry(geom)

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
        self.fmcontrols.curr_folder = self.settings.value('fm_folder', defaultValue=os.getcwd())
        options.addWidget(self.fmcontrols)
        self.emcontrols = EMControls(self.em_imview)
        self.emcontrols.curr_folder = self.settings.value('em_folder', defaultValue=os.getcwd())
        options.addWidget(self.emcontrols)

        # Connect controllers
        self.emcontrols.quit_button.clicked.connect(self.close)
        self.emcontrols.other = self.fmcontrols
        self.fmcontrols.other = self.emcontrols
        self.fmcontrols.merge_btn.clicked.connect(self.merge)

        # Menu Bar
        self._init_menubar()

        self.theme = self.settings.value('theme')
        if self.theme is None:
            self.theme = 'none'
        self._set_theme(self.theme)
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
        action = QtWidgets.QAction('Solarized', self)
        action.triggered.connect(lambda: self._set_theme('solarized'))
        thememenu.addAction(action)
        agroup.addAction(action)

        self.show()
   
    def merge(self):
        self.fm = self.fmcontrols.ops
        self.em = self.emcontrols.ops
        if self.fm is not None and self.em is not None:
            if self.fm._tf_data is not None and (self.em._tf_data is not None or self.em.tf_region is not None):
                self.fm.calc_merge_matrix(self.em.data, self.em.points)
                self.popup = Merge(self)
                self.popup.show()
            else:
                print('Transform FM and EM data first!')
        else:
            print('Select FM and EM data first!')

    def _set_theme(self, name):
        if name == 'none':
            self.setStyleSheet('')
        else:
            self.setStyleSheet('')
            with open(resource_path('styles/%s.qss'%name), 'r') as f:
                self.setStyleSheet(f.read())
            if name == 'solarized':
                c = (203, 76, 22, 80)
                bc = '#002b36'
            else:
                c = (0, 0, 255, 80)
                bc = (0, 0, 0)

            for imview in [self.em_imview, self.fm_imview]:
                imview.view.setBackgroundColor(bc)
                hwidget = imview.getHistogramWidget()
                hwidget.setBackground(bc)
                hwidget.item.region.setBrush(c)
                hwidget.item.fillHistogram(color=c)
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
        self.settings.setValue('channel_colors', self.colors)
        self.settings.setValue('geometry', self.geometry())
        self.settings.setValue('fm_folder', self.fmcontrols.curr_folder)
        self.settings.setValue('em_folder', self.emcontrols.curr_folder)
        event.accept()

def main():
    app = QtWidgets.QApplication([])
    gui = GUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
