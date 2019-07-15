import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage import data

src=data.camera()
c_in=0.5*np.array(src.shape)
c_out=np.array((256.0,256.0))
for i in range(0,7):
    a=i*15.0*np.pi/180.0
    transform=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    offset=c_in-c_out.dot(transform)
    dst=nd.interpolation.affine_transform(src,transform.T,order=2,offset=offset,output_shape=(512,512),cval=0.0,output=np.float32)
    plt.subplot(1,7,i+1)
    plt.imshow(dst,cmap='gray')
    
plt.show()
