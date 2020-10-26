# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ImageViewTemplate.ui'
#
# Created: Thu May  1 15:20:42 2014
#      by: pyside-uic 0.2.15 running on PySide 1.2.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui
import numpy as np


import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems import ImageItem
from pyqtgraph.graphicsItems import GraphicsWidget
from pyqtgraph.graphicsItems import ViewBox
from pyqtgraph.graphicsItems.GradientEditorItem import *
from pyqtgraph.graphicsItems import LinearRegionItem
from pyqtgraph.graphicsItems.PlotDataItem import *
from pyqtgraph.graphicsItems import PlotCurveItem
from pyqtgraph.graphicsItems import VTickGroup
from pyqtgraph.graphicsItems import InfiniteLine
from pyqtgraph.graphicsItems import ROI
from pyqtgraph.graphicsItems import AxisItem
from pyqtgraph.graphicsItems.GridItem import *
from pyqtgraph.Point import Point
from pyqtgraph.graphicsItems import ROI
from pyqtgraph.SignalProxy import SignalProxy

from pyqtgraph.widgets.GraphicsView import GraphicsView
from pyqtgraph.graphicsItems.HistogramLUTItem import HistogramLUTItem
from pyqtgraph.graphicsItems import *
import weakref
import copy
from pyqtgraph.imageview.ImageViewTemplate_pyqt5 import *


def rescaleData(data, scale, offset, dtype=None, clip=None):
    """Return data rescaled and optionally cast to a new dtype::

        data => (data-offset) * scale

    """
    if dtype is None:
        dtype = data.dtype
    else:
        dtype = np.dtype(dtype)

    try:
        if not getConfigOption('useWeave'):
            raise Exception('Weave is disabled; falling back to slower version.')
        try:
            import scipy.weave
        except ImportError:
            raise Exception('scipy.weave is not importable; falling back to slower version.')

        ## require native dtype when using weave
        if not data.dtype.isnative:
            data = data.astype(data.dtype.newbyteorder('='))
        if not dtype.isnative:
            weaveDtype = dtype.newbyteorder('=')
        else:
            weaveDtype = dtype

        newData = np.empty((data.size,), dtype=weaveDtype)
        flat = np.ascontiguousarray(data).reshape(data.size)
        size = data.size

        code = """
        double sc = (double)scale;
        double off = (double)offset;
        for( int i=0; i<size; i++ ) {
            newData[i] = ((double)flat[i] - off) * sc;
        }
        """
        scipy.weave.inline(code, ['flat', 'newData', 'size', 'offset', 'scale'], compiler='gcc')
        if dtype != weaveDtype:
            newData = newData.astype(dtype)
        data = newData.reshape(data.shape)
    except:
        if getConfigOption('useWeave'):
            if getConfigOption('weaveDebug'):
                debug.printExc("Error; disabling weave.")
            setConfigOptions(useWeave=False)

        # p = np.poly1d([scale, -offset*scale])
        # d2 = p(data)
        d2 = data - float(offset)
        d2 *= scale

        # Clip before converting dtype to avoid overflow
        if dtype.kind in 'ui':
            lim = np.iinfo(dtype)
            if clip is None:
                # don't let rescale cause integer overflow
                d2 = np.clip(d2, lim.min, lim.max)
            else:
                d2 = np.clip(d2, max(clip[0], lim.min), min(clip[1], lim.max))
        else:
            if clip is not None:
                d2 = np.clip(d2, *clip)
        data = d2.astype(dtype)
    return data


def makeARGB(data, lut=None, levels=None, scale=None, useRGBA=False):
    """
    Convert an array of values into an ARGB array suitable for building QImages,
    OpenGL textures, etc.

    Returns the ARGB array (unsigned byte) and a boolean indicating whether
    there is alpha channel data. This is a two stage process:

        1) Rescale the data based on the values in the *levels* argument (min, max).
        2) Determine the final output by passing the rescaled values through a
           lookup table.

    Both stages are optional.

    ============== ==================================================================================
    **Arguments:**
    data           numpy array of int/float types. If
    levels         List [min, max]; optionally rescale data before converting through the
                   lookup table. The data is rescaled such that min->0 and max->*scale*::

                      rescaled = (clip(data, min, max) - min) * (*scale* / (max - min))

                   It is also possible to use a 2D (N,2) array of values for levels. In this case,
                   it is assumed that each pair of min,max values in the levels array should be
                   applied to a different subset of the input data (for example, the input data may
                   already have RGB values and the levels are used to independently scale each
                   channel). The use of this feature requires that levels.shape[0] == data.shape[-1].
    scale          The maximum value to which data will be rescaled before being passed through the
                   lookup table (or returned if there is no lookup table). By default this will
                   be set to the length of the lookup table, or 255 if no lookup table is provided.
    lut            Optional lookup table (array with dtype=ubyte).
                   Values in data will be converted to color by indexing directly from lut.
                   The output data shape will be input.shape + lut.shape[1:].
                   Lookup tables can be built using ColorMap or GradientWidget.
    useRGBA        If True, the data is returned in RGBA order (useful for building OpenGL textures).
                   The default is False, which returns in ARGB order for use with QImage
                   (Note that 'ARGB' is a term used by the Qt documentation; the *actual* order
                   is BGRA).
    ============== ==================================================================================
    """

    print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
    profile = debug.Profiler()
    #levels = levels[:data.shape[-1],:]
    if data.ndim not in (2, 3):
        raise TypeError("data must be 2D or 3D")
    if data.ndim == 3 and data.shape[2] > 4:
        raise TypeError("data.shape[2] must be <= 4")

    if lut is not None and not isinstance(lut, np.ndarray):
        lut = np.array(lut)

    if levels is None:
        # automatically decide levels based on data dtype
        if data.dtype.kind == 'u':
            levels = np.array([0, 2 ** (data.itemsize * 8) - 1])
        elif data.dtype.kind == 'i':
            s = 2 ** (data.itemsize * 8 - 1)
            levels = np.array([-s, s - 1])
        elif data.dtype.kind == 'b':
            levels = np.array([0, 1])
        else:
            raise Exception('levels argument is required for float input types')
    if not isinstance(levels, np.ndarray):
        levels = np.array(levels)
    levels = levels.astype(np.float)
    if levels.ndim == 1:
        if levels.shape[0] != 2:
            raise Exception('levels argument must have length 2')
    elif levels.ndim == 2:
        if lut is not None and lut.ndim > 1:
            raise Exception('Cannot make ARGB data when both levels and lut have ndim > 2')
        #if levels.shape != (data.shape[-1], 2):
        #    raise Exception('levels must have shape (data.shape[-1], 2)')
    else:
        raise Exception("levels argument must be 1D or 2D (got shape=%s)." % repr(levels.shape))

    profile()

    # Decide on maximum scaled value
    if scale is None:
        if lut is not None:
            scale = lut.shape[0]
        else:
            scale = 255.

    # Decide on the dtype we want after scaling
    if lut is None:
        dtype = np.ubyte
    else:
        dtype = np.min_scalar_type(lut.shape[0] - 1)

    # awkward, but fastest numpy native nan evaluation
    #
    nanMask = None
    if data.dtype.kind == 'f' and np.isnan(data.min()):
        nanMask = np.isnan(data)
    # Apply levels if given
    if levels is not None:
        if isinstance(levels, np.ndarray) and levels.ndim == 2:
            # we are going to rescale each channel independently
            #if levels.shape[0] != data.shape[-1]:
            #    raise Exception(
            #        "When rescaling multi-channel data, there must be the same number of levels as channels (data.shape[-1] == levels.shape[0])")
            newData = np.empty(data.shape, dtype=int)
            for i in range(data.shape[-1]):
                minVal, maxVal = levels[i]
                if minVal == maxVal:
                    maxVal = np.nextafter(maxVal, 2 * maxVal)
                rng = maxVal - minVal
                rng = 1 if rng == 0 else rng
                newData[..., i] = rescaleData(data[..., i], scale / rng, minVal, dtype=dtype)
            data = newData
        else:
            # Apply level scaling unless it would have no effect on the data
            minVal, maxVal = levels
            if minVal != 0 or maxVal != scale:
                if minVal == maxVal:
                    maxVal = np.nextafter(maxVal, 2 * maxVal)
                data = rescaleData(data, scale / (maxVal - minVal), minVal, dtype=dtype)

    profile()
    # apply LUT if given
    if lut is not None:
        data = applyLookupTable(data, lut)
    else:
        if data.dtype is not np.ubyte:
            data = np.clip(data, 0, 255).astype(np.ubyte)

    profile()

    # this will be the final image array
    imgData = np.empty(data.shape[:2] + (4,), dtype=np.ubyte)

    profile()

    # decide channel order
    if useRGBA:
        order = [0, 1, 2, 3]  # array comes out RGBA
    else:
        order = [2, 1, 0, 3]  # for some reason, the colors line up as BGR in the final image.

    # copy data into image array
    if data.ndim == 2:
        # This is tempting:
        #   imgData[..., :3] = data[..., np.newaxis]
        # ..but it turns out this is faster:
        for i in range(3):
            imgData[..., i] = data
    elif data.shape[2] == 1:
        for i in range(3):
            imgData[..., i] = data[..., 0]
    else:
        for i in range(0, data.shape[2]):
            imgData[..., i] = data[..., order[i]]

    profile()

    # add opaque alpha channel if needed
    if data.ndim == 2 or data.shape[2] == 3:
        alpha = False
        imgData[..., 3] = 255
    else:
        alpha = True

    # apply nan mask through alpha channel
    if nanMask is not None:
        alpha = True
        imgData[nanMask, 3] = 0

    profile()
    return imgData, alpha


class PlotROI(ROI.ROI):
    def __init__(self, size):
        ROI.ROI.__init__(self, pos=[0,0], size=size) #, scaleSnap=True, translateSnap=True)
        self.addScaleHandle([1, 1], [0, 0])
        self.addRotateHandle([0, 0], [0.5, 0.5])


class Imview(pg.ImageView):
    def __init__(self, levelMode='mono'):
        super(Imview, self).__init__(levelMode=levelMode)

    def init_again(self, parent=None, name="ImageView", view=None, imageItem=None,
                 levelMode='mono', *args):
        QtGui.QWidget.__init__(self, parent, *args)
        self._imageLevels = None  # [(min, max), ...] per channel image metrics
        self.levelMin = None  # min / max levels across all channels
        self.levelMax = None

        self.name = name
        self.image = None
        self.axes = {}
        self.imageDisp = None
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.ui.histogram.setParent(None)
        #self.ui.gridLayout.removeWidget(self.ui.histogram)
        self.hist = HistogramWidget(self.ui.layoutWidget)
        self.ui.histogram = self.hist
        self.ui.histogram.setObjectName("histogram")
        self.ui.gridLayout.addWidget(self.ui.histogram, 0, 1, 1, 2)
        self.ui.histogram.item.setImageItem(self.getImageItem())

        self.scene = self.ui.graphicsView.scene()
        self.ui.histogram.setLevelMode(levelMode)

        self.ignorePlaying = False

        if view is None:
            self.view = ViewBox.ViewBox()
        else:
            self.view = view
        self.ui.graphicsView.setCentralItem(self.view)
        self.view.setAspectLocked(True)
        self.view.invertY()

        if imageItem is None:
            #self.imageItem = ImageItem.ImageItem()
            self.imageItem = Item()
        else:
            self.imageItem = imageItem
        self.view.addItem(self.imageItem)
        self.currentIndex = 0

        self.ui.histogram.setImageItem(self.imageItem)

        self.menu = None

        self.ui.normGroup.hide()

        self.roi = PlotROI(10)
        self.roi.setZValue(20)
        self.view.addItem(self.roi)
        self.roi.hide()
        self.normRoi = PlotROI(10)
        self.normRoi.setPen('y')
        self.normRoi.setZValue(20)
        self.view.addItem(self.normRoi)
        self.normRoi.hide()
        self.roiCurves = []
        self.timeLine = InfiniteLine.InfiniteLine(0, movable=True, markers=[('^', 0), ('v', 1)])
        self.timeLine.setPen((255, 255, 0, 200))
        self.timeLine.setZValue(1)
        self.ui.roiPlot.addItem(self.timeLine)
        self.ui.splitter.setSizes([self.height() - 35, 35])
        self.ui.roiPlot.hideAxis('left')
        self.frameTicks = VTickGroup.VTickGroup(yrange=[0.8, 1], pen=0.4)
        self.ui.roiPlot.addItem(self.frameTicks, ignoreBounds=True)

        self.keysPressed = {}
        self.playTimer = QtCore.QTimer()
        self.playRate = 0
        self.lastPlayTime = 0

        self.normRgn = LinearRegionItem.LinearRegionItem()
        self.normRgn.setZValue(0)
        self.ui.roiPlot.addItem(self.normRgn)
        self.normRgn.hide()

        ## wrap functions from view box
        for fn in ['addItem', 'removeItem']:
            setattr(self, fn, getattr(self.view, fn))

        ## wrap functions from histogram
        for fn in ['setHistogramRange', 'autoHistogramRange', 'getLookupTable', 'getLevels']:
            setattr(self, fn, getattr(self.ui.histogram, fn))

        self.timeLine.sigPositionChanged.connect(self.timeLineChanged)
        self.ui.roiBtn.clicked.connect(self.roiClicked)
        self.roi.sigRegionChanged.connect(self.roiChanged)
        # self.ui.normBtn.toggled.connect(self.normToggled)
        self.ui.menuBtn.clicked.connect(self.menuClicked)
        self.ui.normDivideRadio.clicked.connect(self.normRadioChanged)
        self.ui.normSubtractRadio.clicked.connect(self.normRadioChanged)
        self.ui.normOffRadio.clicked.connect(self.normRadioChanged)
        self.ui.normROICheck.clicked.connect(self.updateNorm)
        self.ui.normFrameCheck.clicked.connect(self.updateNorm)
        self.ui.normTimeRangeCheck.clicked.connect(self.updateNorm)
        self.playTimer.timeout.connect(self.timeout)

        self.normProxy = SignalProxy(self.normRgn.sigRegionChanged, slot=self.updateNorm)
        self.normRoi.sigRegionChangeFinished.connect(self.updateNorm)

        self.ui.roiPlot.registerPlot(self.name + '_ROI')
        self.view.register(self.name)

        self.noRepeatKeys = [QtCore.Qt.Key_Right, QtCore.Qt.Key_Left, QtCore.Qt.Key_Up, QtCore.Qt.Key_Down,
                             QtCore.Qt.Key_PageUp, QtCore.Qt.Key_PageDown]

        self.roiClicked()  ## initialize roi plot to correct shape / visibility

    def setImage(self, img, autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None,
                 transform=None, autoHistogramRange=True, levelMode=None):
        print('uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
        if hasattr(img, 'implements') and img.implements('MetaArray'):
            img = img.asarray()

        if not isinstance(img, np.ndarray):
            required = ['dtype', 'max', 'min', 'ndim', 'shape', 'size']
            if not all([hasattr(img, attr) for attr in required]):
                raise TypeError("Image must be NumPy array or any object "
                                "that provides compatible attributes/methods:\n"
                                "  %s" % str(required))

        self.image = img
        self.imageDisp = None
        if levelMode is not None:
            self.ui.histogram.setLevelMode(levelMode)

        if axes is None:
            x, y = (0, 1) if self.imageItem.axisOrder == 'col-major' else (1, 0)

            if img.ndim == 2:
                self.axes = {'t': None, 'x': x, 'y': y, 'c': None}
            elif img.ndim == 3:
                # Ambiguous case; make a guess
                if img.shape[2] <= 4:
                    self.axes = {'t': None, 'x': x, 'y': y, 'c': 2}
                else:
                    self.axes = {'t': 0, 'x': x + 1, 'y': y + 1, 'c': None}
            elif img.ndim == 4:
                # Even more ambiguous; just assume the default
                self.axes = {'t': 0, 'x': x + 1, 'y': y + 1, 'c': 3}
            else:
                raise Exception("Can not interpret image with dimensions %s" % (str(img.shape)))
        elif isinstance(axes, dict):
            self.axes = axes.copy()
        elif isinstance(axes, list) or isinstance(axes, tuple):
            self.axes = {}
            for i in range(len(axes)):
                self.axes[axes[i]] = i
        else:
            raise Exception(
                "Can not interpret axis specification %s. Must be like {'t': 2, 'x': 0, 'y': 1} or ('t', 'x', 'y', 'c')" % (
                    str(axes)))

        for x in ['t', 'x', 'y', 'c']:
            self.axes[x] = self.axes.get(x, None)
        axes = self.axes

        if xvals is not None:
            self.tVals = xvals
        elif axes['t'] is not None:
            if hasattr(img, 'xvals'):
                try:
                    self.tVals = img.xvals(axes['t'])
                except:
                    self.tVals = np.arange(img.shape[axes['t']])
            else:
                self.tVals = np.arange(img.shape[axes['t']])


        self.currentIndex = 0
        self.updateImage(autoHistogramRange=autoHistogramRange)
        if levels is None and autoLevels:
            self.autoLevels()
        if levels is not None:  ## this does nothing since getProcessedImage sets these values again.
            self.setLevels(*levels)

        if self.ui.roiBtn.isChecked():
            self.roiChanged()


        if self.axes['t'] is not None:
            self.ui.roiPlot.setXRange(self.tVals.min(), self.tVals.max())
            self.frameTicks.setXVals(self.tVals)
            self.timeLine.setValue(0)
            if len(self.tVals) > 1:
                start = self.tVals.min()
                stop = self.tVals.max() + abs(self.tVals[-1] - self.tVals[0]) * 0.02
            elif len(self.tVals) == 1:
                start = self.tVals[0] - 0.5
                stop = self.tVals[0] + 0.5
            else:
                start = 0
                stop = 1
            for s in [self.timeLine, self.normRgn]:
                s.setBounds([start, stop])


        self.imageItem.resetTransform()
        if scale is not None:
            self.imageItem.scale(*scale)
        if pos is not None:
            self.imageItem.setPos(*pos)
        if transform is not None:
            self.imageItem.setTransform(transform)


        if autoRange:
            self.autoRange()
        self.roiClicked()


class Item(ImageItem.ImageItem):
    def __init__(self, image=None, **kwargs):
        super(Item, self).__init__(image)

    def render(self):
        # Convert data to QImage for display.

        profile = debug.Profiler()
        if self.image is None or self.image.size == 0:
            return

        # Request a lookup table if this image has only one channel
        if self.image.ndim == 2 or self.image.shape[2] == 1:
            if isinstance(self.lut, Callable):
                lut = self.lut(self.image)
            else:
                lut = self.lut
        else:
            lut = None

        if self.autoDownsample:
            # reduce dimensions of image based on screen resolution
            o = self.mapToDevice(QtCore.QPointF(0,0))
            x = self.mapToDevice(QtCore.QPointF(1,0))
            y = self.mapToDevice(QtCore.QPointF(0,1))

            # Check if graphics view is too small to render anything
            if o is None or x is None or y is None:
                return

            w = Point(x-o).length()
            h = Point(y-o).length()
            if w == 0 or h == 0:
                self.qimage = None
                return
            xds = max(1, int(1.0 / w))
            yds = max(1, int(1.0 / h))
            axes = [1, 0] if self.axisOrder == 'row-major' else [0, 1]
            image = fn.downsample(self.image, xds, axis=axes[0])
            image = fn.downsample(image, yds, axis=axes[1])
            self._lastDownsample = (xds, yds)

            # Check if downsampling reduced the image size to zero due to inf values.
            if image.size == 0:
                return
        else:
            image = self.image

        # if the image data is a small int, then we can combine levels + lut
        # into a single lut for better performance
        levels = self.levels
        if levels is not None and levels.ndim == 1 and image.dtype in (np.ubyte, np.uint16):
            if self._effectiveLut is None:
                eflsize = 2**(image.itemsize*8)
                ind = np.arange(eflsize)
                minlev, maxlev = levels
                levdiff = maxlev - minlev
                levdiff = 1 if levdiff == 0 else levdiff  # don't allow division by 0
                if lut is None:
                    efflut = fn.rescaleData(ind, scale=255./levdiff,
                                            offset=minlev, dtype=np.ubyte)
                else:
                    lutdtype = np.min_scalar_type(lut.shape[0]-1)
                    efflut = fn.rescaleData(ind, scale=(lut.shape[0]-1)/levdiff,
                                            offset=minlev, dtype=lutdtype, clip=(0, lut.shape[0]-1))
                    efflut = lut[efflut]

                self._effectiveLut = efflut
            lut = self._effectiveLut
            levels = None

        # Convert single-channel image to 2D array
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        # Assume images are in column-major order for backward compatibility
        # (most images are in row-major order)
        if self.axisOrder == 'col-major':
            image = image.transpose((1, 0, 2)[:image.ndim])

        #argb, alpha = fn.makeARGB(image, lut=lut, levels=levels)
        argb, alpha = makeARGB(image, lut=lut, levels=levels)
        self.qimage = fn.makeQImage(argb, alpha, transpose=False)


class HistogramWidget(GraphicsView):
    def __init__(self, parent=None, *args, **kargs):
        background = kargs.pop('background', 'default')
        GraphicsView.__init__(self, parent, useOpenGL=False, background=background)
        self.item = Histogram(*args, **kargs)
        self.setCentralItem(self.item)
        self.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.setMinimumWidth(95)

    def sizeHint(self):
        return QtCore.QSize(115, 200)

    def __getattr__(self, attr):
        return getattr(self.item, attr)

#__all__ = ['HistogramLUTItem']


class Histogram(GraphicsWidget.GraphicsWidget):
    """
    This is a graphicsWidget which provides controls for adjusting the display of an image.

    Includes:

    - Image histogram
    - Movable region over histogram to select black/white levels
    - Gradient editor to define color lookup table for single-channel images

    Parameters
    ----------
    image : ImageItem or None
        If *image* is provided, then the control will be automatically linked to
        the image and changes to the control will be immediately reflected in
        the image's appearance.
    fillHistogram : bool
        By default, the histogram is rendered with a fill.
        For performance, set *fillHistogram* = False.
    rgbHistogram : bool
        Sets whether the histogram is computed once over all channels of the
        image, or once per channel.
    levelMode : 'mono' or 'rgba'
        If 'mono', then only a single set of black/whilte level lines is drawn,
        and the levels apply to all channels in the image. If 'rgba', then one
        set of levels is drawn for each channel.
    """

    sigLookupTableChanged = QtCore.Signal(object)
    sigLevelsChanged = QtCore.Signal(object)
    sigLevelChangeFinished = QtCore.Signal(object)

    def __init__(self, image=None, fillHistogram=True, rgbHistogram=False, levelMode='mono', num_channels=4):
        GraphicsWidget.GraphicsWidget.__init__(self)

        self.colors = None
        self.lut = None
        self.imageItem = lambda: None   # fake a dead weakref
        self.levelMode = levelMode
        self.num_channels = num_channels
        # self.levelMode = 'rgba'
        self.rgbHistogram = rgbHistogram

        self.layout = QtGui.QGraphicsGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(1, 1, 1, 1)
        self.layout.setSpacing(0)
        self.vb = ViewBox.ViewBox(parent=self)
        self.vb.setMaximumWidth(152)
        self.vb.setMinimumWidth(45)
        self.vb.setMouseEnabled(x=False, y=True)
        self.gradient = GradientEditorItem.GradientEditorItem()
        self.gradient.setOrientation('right')
        self.gradient.loadPreset('grey')


        self.regions = [LinearRegionItem.LinearRegionItem([0, 1], 'horizontal', swapMode='block')]
        self.region = self.regions[0]

        self.plots = [PlotCurveItem.PlotCurveItem(pen=(200, 200, 200, 100))]
        self.plot = self.plots[0]

        for plot in self.plots:
            plot.rotate(90)

        self.axis = AxisItem.AxisItem('left', linkView=self.vb, maxTickLength=-10, parent=self)
        self.layout.addItem(self.axis, 0, 0)
        self.layout.addItem(self.vb, 0, 1)
        self.layout.addItem(self.gradient, 0, 2)
        self.range = None
        self.gradient.setFlag(self.gradient.ItemStacksBehindParent)
        self.vb.setFlag(self.gradient.ItemStacksBehindParent)

        self.gradient.sigGradientChanged.connect(self.gradientChanged)
        self.vb.sigRangeChanged.connect(self.viewRangeChanged)
        self.add = QtGui.QPainter.CompositionMode_Plus

        if image is not None:
            self.setImageItem(image)

    def init_hist(self, colors, num_channels):
        self.num_channels = num_channels
        if self.colors is None:
            self.colors = copy.copy(colors)

        counter = 0
        while len(self.regions) <= self.num_channels+1:
            self.regions.append(LinearRegionItem.LinearRegionItem([0, 1], 'horizontal', swapMode='block', pen=self.colors[counter],
                             brush=fn.mkBrush(self.colors[counter]), span=(1 / 3., 2 / 3.)))
            self.plots.append(PlotCurveItem.PlotCurveItem(pen=self.colors[counter], compositionMode=self.add))
            counter += 1

        for region in self.regions:
            region.setZValue(1000)
            self.vb.addItem(region)
            region.lines[0].addMarker('<|', 0.5)
            region.lines[1].addMarker('|>', 0.5)
            region.sigRegionChanged.connect(self.regionChanging)
            region.sigRegionChangeFinished.connect(self.regionChanged)

        for plot in self.plots:
            plot.rotate(90)
            self.vb.addItem(plot)

        self._showRegions()

        self.vb.addItem(self.plot)
        self.autoHistogramRange()

    def changeColors(self, colors):
        if self.levelMode == 'mono':
            return

        if colors != self.colors:
            if colors is not None:
                self.colors = copy.copy(colors)
            for i in range(len(self.regions)-1):
                self.vb.removeItem(self.regions[i+1])
                self.vb.removeItem(self.plots[i+1])
            self.regions = [self.regions[0]]
            self.plots = [self.plots[0]]

            counter = 0
            while len(self.regions) <= self.num_channels+1:
                self.regions.append(LinearRegionItem.LinearRegionItem([0, 1], 'horizontal', swapMode='block', pen=colors[counter],
                                                                      brush=fn.mkBrush(colors[counter]),
                                                                      span=(1 / 3., 2 / 3.)))
                self.plots.append(PlotCurveItem.PlotCurveItem(pen=colors[counter], compositionMode=self.add))
                counter += 1

            for i in range(len(self.regions)-1):
                region = self.regions[i+1]
                region.setZValue(1000)
                self.vb.addItem(region)
                region.lines[0].addMarker('<|', 0.5)
                region.lines[1].addMarker('|>', 0.5)
                region.sigRegionChanged.connect(self.regionChanging)
                region.sigRegionChangeFinished.connect(self.regionChanged)
                plot = self.plots[i+1]
                plot.rotate(90)
                self.vb.addItem(plot)

            #lut = self.getLookupTable(self.imageItem())
            #raw_colors = [color.lstrip('#') for color in colors]
            #rgb_colors = [tuple(int(color[i:i+2], 16) for i in (0,2,4)) for color in raw_colors]
            #print(rgb_colors)
            #self.gradient.restoreState({'mode': 'rgb',
            #                            'ticks': [(0.0, rgb_colors[0]),
            #                                      (0.25, rgb_colors[1]),
            #                                      (0.5, rgb_colors[2]),
            #                                      (0.75, rgb_colors[3])]})

            #self.gradient.loadPreset('thermal')
            #self.imageItem().setLookupTable(self.getLookupTable)  ## send function pointer, not the result

            #lut2 = self.getLookupTable(self.imageItem())

            #self.sigLookupTableChanged.emit(self)

    def fillHistogram(self, fill=True, level=0.0, color=(100, 100, 200)):
        if self.colors is None:
            colors = [color, (255, 0, 0, 50), (0, 255, 0, 50), (0, 0, 255, 50), (255, 255, 255, 50), (255, 255, 255, 50), (255, 255, 255, 50)]
        else:
            colors = []
            for i in range(self.num_channels):
                colors.append(self.colors[i])
        for i in range(len(self.plots)-1):
            plot = self.plots[i]
            if fill:
                plot.setFillLevel(level)
                plot.setBrush(colors[i])
            else:
                plot.setFillLevel(None)

    def paint(self, p, *args):
        if self.levelMode != 'mono':
            return

        pen = self.region.lines[0].pen
        rgn = self.getLevels()
        p1 = self.vb.mapFromViewToItem(self, Point(self.vb.viewRect().center().x(), rgn[0]))
        p2 = self.vb.mapFromViewToItem(self, Point(self.vb.viewRect().center().x(), rgn[1]))
        gradRect = self.gradient.mapRectToParent(self.gradient.gradRect.rect())
        for pen in [fn.mkPen((0, 0, 0, 100), width=3), pen]:
            p.setPen(pen)
            p.drawLine(p1 + Point(0, 5), gradRect.bottomLeft())
            p.drawLine(p2 - Point(0, 5), gradRect.topLeft())
            p.drawLine(gradRect.topLeft(), gradRect.topRight())
            p.drawLine(gradRect.bottomLeft(), gradRect.bottomRight())

    def setHistogramRange(self, mn, mx, padding=0.1):
        """Set the Y range on the histogram plot. This disables auto-scaling."""
        self.vb.enableAutoRange(self.vb.YAxis, False)
        self.vb.setYRange(mn, mx, padding)

    def autoHistogramRange(self):
        """Enable auto-scaling on the histogram plot."""
        self.vb.enableAutoRange(self.vb.XYAxes)

    def setImageItem(self, img):
        """Set an ImageItem to have its levels and LUT automatically controlled
        by this HistogramLUTItem.
        """
        self.imageItem = weakref.ref(img)
        img.sigImageChanged.connect(self.imageChanged)
        img.setLookupTable(self.getLookupTable)  ## send function pointer, not the result
        self.regionChanged()
        self.imageChanged(autoLevel=True)

    def viewRangeChanged(self):
        self.update()

    def gradientChanged(self):
        if self.imageItem() is not None:
            if self.gradient.isLookupTrivial():
                self.imageItem().setLookupTable(None)  # lambda x: x.astype(np.uint8))
            else:
                self.imageItem().setLookupTable(self.getLookupTable)  ## send function pointer, not the result
        self.lut = None
        self.sigLookupTableChanged.emit(self)

    def getLookupTable(self, img=None, n=None, alpha=None):
        """Return a lookup table from the color gradient defined by this
        HistogramLUTItem.
        """
        if self.levelMode != 'mono':
            return None
        if n is None:
            #if img.dtype == np.uint8:
            #    n = 256
            #else:
            #    n = 512
            n = 512
        if self.lut is None:
            self.lut = self.gradient.getLookupTable(n, alpha=alpha)
        return self.lut

    def regionChanged(self):
        if self.imageItem() is not None:
            self.imageItem().setLevels(self.getLevels())
        #    pass
        #self.sigLevelChangeFinished.emit(self)

    def regionChanging(self):
        if self.imageItem() is not None:
            self.imageItem().setLevels(self.getLevels())
        #    pass
        #self.update()
        #self.sigLevelsChanged.emit(self)
        pass

    def imageChanged(self, autoLevel=False, autoRange=False):
        if self.imageItem() is None:
            return
        if self.levelMode == 'mono':
            for plt in self.plots[1:]:
                plt.setVisible(False)
            self.plots[0].setVisible(True)
            # plot one histogram for all image data
            profiler = debug.Profiler()
            h = self.imageItem().getHistogram()
            profiler('get histogram')
            if h[0] is None:
                return
            self.plot.setData(*h)
            profiler('set plot')
            if autoLevel:
                mn = h[0][0]
                mx = h[0][-1]
                self.region.setRegion([mn, mx])
                profiler('set region')
            else:
                mn, mx = self.imageItem().levels
                self.region.setRegion([mn, mx])
        elif self.levelMode == 'rgba':
            # plot one histogram for each channel
            self.plots[0].setVisible(False)
            ch = self.imageItem().getHistogram(perChannel=True)
            if ch[0] is None:
                return
            for i in range(1, self.num_channels+1):
                if len(ch) >= i:
                    h = ch[i - 1]
                    self.plots[i].setVisible(True)
                    self.plots[i].setData(*h)
                    if autoLevel:
                        mn = h[0][0]
                        mx = h[0][-1]
                        self.region[i].setRegion([mn, mx])
                else:
                    # hide channels not present in image data
                    self.plots[i].setVisible(False)
            # make sure we are displaying the correct number of channels
            self._showRegions()
        else:
            # plot one histogram for each channel
            self.plots[0].setVisible(False)
            ch = self.imageItem().getHistogram(perChannel=True)
            if ch[0] is None:
                return
            for i in range(self.num_channels):
                if len(ch) >= i:
                    h = ch[i - 1]
                    self.plots[i].setVisible(True)
                    self.plots[i].setData(*h)
                    if autoLevel:
                        mn = h[0][0]
                        mx = h[0][-1]
                        self.region[i].setRegion([mn, mx])
                else:
                    # hide channels not present in image data
                    self.plots[i].setVisible(False)
            # make sure we are displaying the correct number of channels
            self._showRegions()

    def getLevels(self):
        """Return the min and max levels.

        For rgba mode, this returns a list of the levels for each channel.
        """
        if self.levelMode == 'mono':
            return self.region.getRegion()
        else:
            #nch = self.imageItem().channels()
            nch = self.num_channels
            return [r.getRegion() for r in self.regions[1:nch + 1]]

    def setLevels(self, min=None, max=None, rgba=None):
        """Set the min/max (bright and dark) levels.

        Arguments may be *min* and *max* for single-channel data, or
        *rgba* = [(rmin, rmax), ...] for multi-channel data.
        """
        if self.levelMode == 'mono':
            if min is None:
                min, max = rgba[0]
            assert None not in (min, max)
            self.region.setRegion((min, max))
        else:
            if rgba is None:
                raise TypeError("Must specify rgba argument when levelMode != 'mono'.")
            for i, levels in enumerate(rgba):
                self.regions[i + 1].setRegion(levels)

    def setLevelMode(self, mode):
        """ Set the method of controlling the image levels offered to the user.
        Options are 'mono' or 'rgba'.
        """
        assert mode in ('mono', 'rgba', 'custom')

        if mode == self.levelMode:
            return

        oldLevels = self.getLevels()
        self.levelMode = mode
        self._showRegions()

        # do our best to preserve old levels
        if mode == 'mono':
            levels = np.array(oldLevels).mean(axis=0)
            self.setLevels(*levels)
        else:
            levels = [oldLevels] * self.num_channels
            self.setLevels(rgba=levels)

        # force this because calling self.setLevels might not set the imageItem
        # levels if there was no change to the region item
        self.imageItem().setLevels(self.getLevels())

        self.imageChanged()
        self.update()

    def _showRegions(self):
        for i in range(len(self.regions)):
            self.regions[i].setVisible(False)

        if self.levelMode == 'rgba':
            imax = 4
            if self.imageItem() is not None:
                # Only show rgb channels if connected image lacks alpha.
                nch = self.imageItem().channels()
                if nch is None:
                    nch = 3
            xdif = 1.0 / nch
            for i in range(1, nch + 1):
                self.regions[i].setVisible(True)
                self.regions[i].setSpan((i - 1) * xdif, i * xdif)
            self.gradient.hide()
        elif self.levelMode == 'custom':
            nch = self.num_channels
            xdif = 1.0 / nch
            for i in range(1, nch + 1):
                self.regions[i].setVisible(True)
                self.regions[i].setSpan((i - 1) * xdif, i * xdif)
            self.gradient.hide()
        elif self.levelMode == 'mono':
            self.regions[0].setVisible(True)
            self.gradient.show()
        else:
            raise ValueError("Unknown level mode %r" % self.levelMode)

    def saveState(self):
        return {
            'gradient': self.gradient.saveState(),
            'levels': self.getLevels(),
            'mode': self.levelMode,
        }

    def restoreState(self, state):
        self.setLevelMode(state['mode'])
        self.gradient.restoreState(state['gradient'])
        self.setLevels(*state['levels'])
