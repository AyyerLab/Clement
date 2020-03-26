#!/usr/bin/env python

import sys
import os
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from . import res_styles
from .em_controls import EMControls
from .fm_controls import FMControls
from .fib_controls import FIBControls
from .project import Project
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
        self.em_imview = QtWidgets.QStackedWidget()
        self.tem_imview = pg.ImageView()
        self.tem_imview.ui.roiBtn.hide()
        self.tem_imview.ui.menuBtn.hide()
        self.fib_imview = pg.ImageView()
        self.fib_imview.ui.roiBtn.hide()
        self.fib_imview.ui.menuBtn.hide()
        self.em_imview.addWidget(self.tem_imview)
        self.em_imview.addWidget(self.fib_imview)
        splitter_images.addWidget(self.em_imview)

        # Options
        options = QtWidgets.QHBoxLayout()
        options.setContentsMargins(4, 0, 4, 4)
        layout.addLayout(options)

        self.fmcontrols = FMControls(self.fm_imview, self.colors)
        self.fmcontrols.curr_folder = self.settings.value('fm_folder', defaultValue=os.getcwd())
        options.addWidget(self.fmcontrols)

        vbox = QtWidgets.QVBoxLayout()
        options.addLayout(vbox)
        self.tabs = QtWidgets.QTabWidget()
        self.tab_2d = QtWidgets.QWidget()
        self.tab_3d = QtWidgets.QWidget()
        self.tabs.resize(300, 200)
        self.tabs.addTab(self.tab_2d, 'TEM/SEM')
        self.tabs.addTab(self.tab_3d, 'FIB')
        vbox.addWidget(self.tabs)

        vbox_2d = QtWidgets.QVBoxLayout()
        vbox_3d = QtWidgets.QVBoxLayout()
        self.tab_2d.setLayout(vbox_2d)
        self.tab_3d.setLayout(vbox_3d)

        self.emcontrols = EMControls(self.tem_imview, vbox_2d)
        self.emcontrols.curr_folder = self.settings.value('em_folder', defaultValue=os.getcwd())
        vbox_2d.addWidget(self.emcontrols)
        self.fibcontrols = FIBControls(self.fib_imview, vbox_3d, self.emcontrols.ops)
        self.fibcontrols.curr_folder = self.settings.value('em_folder', defaultValue=os.getcwd())
        vbox_3d.addWidget(self.fibcontrols)

        self.tabs.currentChanged.connect(self.select_tab)
        # Connect controllers
        self.emcontrols.quit_button.clicked.connect(self.close)
        self.emcontrols.other = self.fmcontrols
        self.fibcontrols.quit_button.clicked.connect(self.close)
        self.fibcontrols.other = self.fmcontrols
        self.fmcontrols.other = self.emcontrols
        self.fmcontrols.merge_btn.clicked.connect(self.merge)

        self.popup = None
        self.project = Project(self.fmcontrols, self.emcontrols, self)
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
        action = QtWidgets.QAction('Load project', self)
        action.triggered.connect(self._load_p)
        filemenu.addAction(action)
        action = QtWidgets.QAction('Save project', self)
        action.triggered.connect(self._save_p)
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

    def _save_p(self):
        self.project._save_project()
    
    def _load_p(self):
        self.project._load_project()

    def select_tab(self, idx):
        if idx == 0:
            self.em_imview.setCurrentIndex(0)
            self.emcontrols._update_imview()
            self.fmcontrols.other = self.emcontrols
        else:
            self.em_imview.setCurrentIndex(1)
            self.fibcontrols._update_imview()
            self.fibcontrols.sem_ops = self.emcontrols.ops
            self.fmcontrols.other = self.fibcontrols
            if self.emcontrols.ops is not None:
                if self.emcontrols.ops.points is not None:
                    self.fibcontrols.enable_buttons(enable=True)
                else:
                    self.fibcontrols.enable_buttons(enable=False)

    def merge(self,project=None):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.fm = self.fmcontrols.ops
        self.em = self.emcontrols.ops 
        
        if self.fm is not None and self.em is not None:
            if self.fm._tf_points is not None and (self.em._tf_points is not None or self.em._tf_points_region is not None):
                self.fm.calc_merge_matrix(self.em.data, self.em.points)
                if self.popup is not None:
                    self.popup.close()
                self.popup = Merge(self)
                self.project.merged = True
                self.project.popup = self.popup
                if self.project.load_merge:
                    self.project._load_merge(project)
                    self.project.load_merge = False
                QtWidgets.QApplication.restoreOverrideCursor()
                self.popup.show()
            else:
                print('You have to transform both images first!')
                QtWidgets.QApplication.restoreOverrideCursor()
        else:
            print('Select FM and EM data first!')
            QtWidgets.QApplication.restoreOverrideCursor()

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

            for imview in [self.fib_imview, self.tem_imview, self.fm_imview]:
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
        self.settings.setValue('fib_folder', self.fibcontrols.curr_folder)
        event.accept()

def main():
    app = QtWidgets.QApplication([])
    gui = GUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
