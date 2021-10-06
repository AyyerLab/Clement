from autoscript_sdb_microscope_client import SdbMicroscopeClient
from autoscript_sdb_microscope_client.enumerations import *
from autoscript_sdb_microscope_client.structures import *
import numpy as np
from matplotlib import pyplot as plt
plt.ion()
microscope = SdbMicroscopeClient()

microscope.connect('localhost')

hv = microscope.beams.electron_beam.high_voltage.value

microscope.imaging.set_active_view(2)
microscope.imaging.set_active_device(ImagingDevice.ION_BEAM)
microscope.beams.ion_beam.horizontal_field_width.value = 20e-6

img = microscope.imaging.get_image()
print(img.data.shape)
plt.figure()
plt.imshow(img.data)
plt.title('before milling')

microscope.patterning.set_default_beam_type(BeamType.ION)

#?????
microscope.patterning.set_default_application_file('Si')

circles = []
for i in range(3):
    for j in range(3):
        circ = microscope.patterning.create_circle(i*5e-6, j*5e-6, 1e-6, 0.8e-6, 1e-6)
        circ.gas_type = 'Pt dep' #???

microscope.beams.ion_beam.turn_on()

microscope.patterning.run()
microscope.patterning.stop()
microscope.imaging.grab_frame(GrabFrameSettings(resolution="1536x1024", dwell_time=1e-6))
img_milled = microscope.imaging.get_image()

plt.figure()
plt.imshow(img_milled.data)
plt.title('milled')
plt.show()
