from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
from collections.abc import Iterable
import time
import copy
from datetime import datetime
import os
import traceback
import time

def wait_cursor(printer=None):
    def wait(func):
        def wrapper(self, *args, **kwargs):
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            printer = None
            if getattr(self, 'print') is not None:
                printer = getattr(self, 'print')
                printer('Run ', func.__name__)
            #print_done = False
            try:
                #start = time.time()
                func(self, *args, **kwargs)
                #end = time.time()
                #if (end-start) > 1 and printer is not None:
                #    printer('Done')
            except Exception as e:
                QtWidgets.QApplication.restoreOverrideCursor()
                if printer is not None:
                    printer(traceback.format_exc())
                raise
            QtWidgets.QApplication.restoreOverrideCursor()
        return wrapper
    return wait

def add_montage_line(parent, vbox, type_str, downsampling=False):
    line = QtWidgets.QHBoxLayout()
    vbox.addLayout(line)
    button = QtWidgets.QPushButton('Load %s Image:'%type_str, parent)
    button.clicked.connect(parent._load_mrc)
    line.addWidget(button)
    parent.mrc_fname = QtWidgets.QLabel(parent)
    line.addWidget(parent.mrc_fname, stretch=1)

    if downsampling:
        line = QtWidgets.QHBoxLayout()
        vbox.addLayout(line)

        label = QtWidgets.QLabel('Filtering:')
        line.addWidget(label)
        step_label = QtWidgets.QLabel(parent)
        step_label.setText('Downsampling:')
        parent.step_box = QtWidgets.QLineEdit(parent)
        parent.step_box.setMaximumWidth(30)
        parent.step_box.setText('10')
        parent.step_box.setEnabled(False)
        parent._downsampling = parent.step_box.text()
        line.addWidget(step_label)
        line.addWidget(parent.step_box)

        filter_label = QtWidgets.QLabel(parent)
        filter_label.setText('Filter:')
        parent.sl_filter = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        parent.sl_filter.setRange(1,10)
        parent.sl_filter.setFocusPolicy(QtCore.Qt.NoFocus)
        parent.sl_filter.valueChanged.connect(lambda state, param=0: _set_filter(parent, param))
        parent.sl_box = QtWidgets.QSpinBox(parent)
        parent.sl_box.setRange(1,10)
        parent.sl_box.editingFinished.connect(lambda param=1: _set_filter(parent, param))
        parent.sl_box.setValue(3)
        line.addWidget(filter_label)
        line.addWidget(parent.sl_filter)
        line.addWidget(parent.sl_box)

        line.addStretch(1)
        parent.assemble_btn = QtWidgets.QPushButton('Assemble', parent)
        parent.assemble_btn.clicked.connect(parent._assemble_mrc)
        parent.assemble_btn.setEnabled(False)
        line.addWidget(parent.assemble_btn)
    parent.transp_btn = QtWidgets.QCheckBox('Transpose', parent)
    parent.transp_btn.clicked.connect(parent._transpose)
    parent.transp_btn.setEnabled(False)
    line.addWidget(parent.transp_btn)

def _set_filter(parent, param, state=None):
    if parent.tag is not 'SEM':
        return
    if param == 0:
        value = parent.sl_filter.value()
        parent.sl_box.blockSignals(True)
        parent.sl_box.setValue(value)
        parent.sl_box.blockSignals(False)
    else:
        value = parent.sl_box.value()
        parent.sl_filter.blockSignals(True)
        parent.sl_filter.setValue(value)
        parent.sl_filter.blockSignals(False)
        parent.sl_box.clearFocus()
    parent.ops._update_data(filter=value)
    parent._update_imview()

def add_fmpeaks_line(parent, vbox):
    line = QtWidgets.QHBoxLayout()
    vbox.addLayout(line)
    label = QtWidgets.QLabel('Peaks:')
    line.addWidget(label)
    parent.show_peaks_btn = QtWidgets.QCheckBox('Show FM peaks',parent)
    parent.show_peaks_btn.setEnabled(True)
    parent.show_peaks_btn.setChecked(False)
    #parent.show_peaks_btn.stateChanged.connect(parent._show_FM_peaks)
    parent.show_peaks_btn.toggled.connect(parent._show_FM_peaks)
    parent.show_peaks_btn.setEnabled(False)
    line.addWidget(parent.show_peaks_btn)
    label = QtWidgets.QLabel('Translation:', parent)
    line.addWidget(label)
    parent.translate_peaks_btn = QtWidgets.QPushButton('Collective', parent)
    parent.translate_peaks_btn.setCheckable(True)
    parent.translate_peaks_btn.setChecked(False)
    parent.translate_peaks_btn.toggled.connect(parent._translate_peaks)
    parent.translate_peaks_btn.setEnabled(False)
    line.addWidget(parent.translate_peaks_btn)
    parent.refine_peaks_btn = QtWidgets.QPushButton('Individual', parent)
    parent.refine_peaks_btn.setCheckable(True)
    parent.refine_peaks_btn.setChecked(False)
    parent.refine_peaks_btn.toggled.connect(parent._refine_peaks)
    parent.refine_peaks_btn.setEnabled(False)
    line.addWidget(parent.refine_peaks_btn)
    line.addStretch(1)

    size_label = QtWidgets.QLabel(parent)
    size_label.setText('Bead size [\u03BCm]:')
    parent.size_box = QtWidgets.QLineEdit(parent)
    parent.size_box.setText('1')
    parent.size_box.setEnabled(False)
    parent._bead_size = parent.size_box.text()
    parent.size_box.setMaximumWidth(30)
    line.addWidget(size_label)
    line.addWidget(parent.size_box)
    parent.auto_opt_btn = QtWidgets.QPushButton('Fit beads', parent)
    parent.auto_opt_btn.setEnabled(False)
    parent.auto_opt_btn.clicked.connect(parent.fit_circles)
    line.addWidget(parent.auto_opt_btn)
    line.addStretch(1)

def add_define_grid_line(parent, vbox):
    line = QtWidgets.QHBoxLayout()
    vbox.addLayout(line)
    label = QtWidgets.QLabel('Grid:', parent)
    line.addWidget(label)
    parent.define_btn = QtWidgets.QPushButton('Define grid square', parent)
    parent.define_btn.setCheckable(True)
    parent.define_btn.toggled.connect(parent._define_grid_toggled)
    parent.define_btn.setEnabled(False)
    line.addWidget(parent.define_btn)
    parent.show_grid_btn = QtWidgets.QCheckBox('Show grid square', parent)
    parent.show_grid_btn.setEnabled(False)
    parent.show_grid_btn.setChecked(False)
    parent.show_grid_btn.stateChanged.connect(parent._show_grid)
    line.addWidget(parent.show_grid_btn)
    line.addStretch(1)

def add_transform_grid_line(parent, vbox, show_original=True):
    line = QtWidgets.QHBoxLayout()
    vbox.addLayout(line)
    label = QtWidgets.QLabel('Transformations:', parent)
    line.addWidget(label)
    parent.transform_btn = QtWidgets.QPushButton('Transform image', parent)
    parent.transform_btn.clicked.connect(parent._affine_transform)
    parent.transform_btn.setEnabled(False)
    line.addWidget(parent.transform_btn)
    #parent.rot_transform_btn = QtWidgets.QCheckBox('Disable Shearing', parent)
    #parent.rot_transform_btn.setEnabled(False)
    #line.addWidget(parent.rot_transform_btn)
    if show_original:
        parent.show_btn = QtWidgets.QCheckBox('Show original data', parent)
        parent.show_btn.setEnabled(False)
        parent.show_btn.setChecked(True)
        parent.show_btn.stateChanged.connect(parent._show_original)
        line.addWidget(parent.show_btn)
        line.addStretch(1)
    return line

class PrintGUI(QtCore.QObject):
    def __init__(self, label):
        super(PrintGUI, self).__init__()
        self.label = label
        self.string = ''
        self.log_string = None
        self.log_file = None

    def print(self, *args):
        self.parse(*args)
        self.parse_log(*args)

    def log(self, *args):
        self.parse_log(*args)

    def parse(self, *args):
        QtCore.QCoreApplication.processEvents()
        full_string = ' '
        s_list = []
        for s in args:
            string = self.convert(s)
            s_list.append(string)
        self.label.setText(full_string.join(s_list))

    def parse_log(self,*args):
        full_string = ' '
        s_list = []
        for s in args:
            string = self.convert(s)
            s_list.append(string)
        self.log_file.write(full_string.join(s_list) + '\n')

    def convert(self, s):
        string = None
        if isinstance(s, str):
            string = s
        elif isinstance(s, Iterable):
            conv = np.array(s).tolist()
            string = ' '
            string = string.join([str(elem) for elem in conv])
        else:
            string = str(s)
        return string

    def run(self):
        print("Let's go!")
        dir_name = 'log'
        tot_path = os.path.join(os.getcwd(), dir_name)
        if not os.path.isdir(tot_path):
            os.mkdir(tot_path)

        dtime = datetime.now()
        fname = dtime.strftime('%Y%m%d_%H%M%S.txt')
        self.log_file = open(os.path.join(tot_path,fname), 'a')

        print(tot_path)