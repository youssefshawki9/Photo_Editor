import cv2
import numpy as np
import scipy    # pip install scipy
from scipy import ndimage
import scipy.fftpack as fp
import matplotlib.pyplot as plt
from skimage.io import imread    # pip install scikit-image

#########################################
# SHOW PLOT FUNCTION
#########################################

def showPlot(img):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    # plt.show()


#########################################
# HIGH PASS FILTER
#########################################


#########################################
# MAIN
#########################################

# keep original image under imgOG
imgOG = cv2.imread('lena.png')
# keep filtered image under imgFiltered
imgFiltered = imgOG.copy()
imgFiltered = cv2.cvtColor(imgOG, cv2.COLOR_BGR2HSV)

imgOG = cv2.cvtColor(imgOG, cv2.COLOR_BGR2RGB)
# showPlot(imgOG)

imgValues = imgFiltered[:,:,2]

F1 = fp.fft2((imgValues).astype(float))
F2 = fp.fftshift(F1)
fftOG = (20*np.log10(0.1 + F2)).astype(int)
# showPlot(fftOG)

(w, h) = imgValues.shape
half_w, half_h = int(w/2), int(h/2)
# high pass filter
n = 25
F2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0 # select all but the first 50x50 (low) frequencies
fftHP = (20*np.log10(0.1 + F2)).astype(int)
(20*np.log10( 0.1 + F2)).astype(int)
# showPlot(fftHP)

imgValues = fp.ifft2(fp.ifftshift(F2)).real
imgFiltered[:,:,2] = imgValues
imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_HSV2RGB)
# showPlot(imgFiltered)

plt.subplot(221), plt.imshow(imgOG), plt.title('Original Image')
plt.axis('off')
plt.subplot(222), plt.imshow(fftOG, 'gray'), plt.title('original freq')
plt.axis('off')
plt.subplot(223), plt.imshow(fftHP, 'gray'), plt.title('High pass filter')
plt.axis('off')
plt.subplot(224), plt.imshow(imgFiltered), plt.title('Effect after filtering')
plt.axis('off')


plt.show()