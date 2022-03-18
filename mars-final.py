import cv2
import numpy as np
import scipy    # pip install scipy
from scipy import ndimage
import scipy.fftpack as fp
import matplotlib.pyplot as plt
from skimage.io import imread    # pip install scikit-image

#########################################
# MOVE IMAGE TO FREQUENCY DOMAIN
#########################################
def toFFT(img):
    F1 = fp.fft2((img).astype(float))
    F2 = fp.fftshift(F1)
    fft = (20*np.log10(0.1 + F2)).astype(int)
    return F1, F2, fft


#########################################
# HIGH-PASS FILTER
#########################################
def highPassFiltering(img, F2):
    (w, h) = img.shape
    half_w, half_h = int(w/2), int(h/2)
    n = 50    # high-pass size
    # select all but the first 50x50 frequencies
    F2[half_w-n:half_w+n+1, half_h-n:half_h+n+1] = 0
    fft = (20*np.log10(0.1 + F2)).astype(int)
    img = fp.ifft2(fp.ifftshift(F2)).real
    return img, fft, F2


#########################################
# LOW-PASS FILTER
#########################################
def lowPassFiltering(img, F2):
    (w, h) = img.shape
    half_w, half_h = int(w/2), int(h/2)
    n = 30    # low-pass size
    Fblank = np.zeros((w, h), np.uint8)
    # select the first 30x30 frequencies
    Fblank[half_w-n:half_w+n+1, half_h-n:half_h+n+1] = 1
    F2 = Fblank*F2
    fft = (20*np.log10(0.1 + F2)).astype(int)
    img = fp.ifft2(fp.ifftshift(F2)).real
    return img, fft, F2


#########################################
# CHOOSE FILTER
#########################################
def chooseFilter(imgValues, filter):
    if filter == 'lowpass':
        # move to frequency domain then apply lowpass filter
        F1, F2, fftOG = toFFT(imgValues)
        imgValues, fftFiltered, F2 = lowPassFiltering(imgValues, F2)

    elif filter == 'median':
        # median acts as a=low-pass filter ---- blurring effect
        # src : source file ---- ksize: int kernel size
        imgValues = cv2.medianBlur(imgValues, 21)
        _, _, fftFiltered = toFFT(imgValues)

    elif filter == 'highpass':
        # move to frequency domain then apply highpass filter
        F1, F2, fftOG = toFFT(imgValues)
        imgValues, fftFiltered, F2 = highPassFiltering(imgValues, F2)

    elif filter == 'laplacian':
        # laplacian acts as hig-pass filter ---- edge detector effect
        # src : source file ---- ddepth : depth of output image ---- ksize : blurring kernel size
        imgValues = cv2.GaussianBlur(imgValues, (3,3), 0)
        imgValues = cv2.Laplacian(imgValues, cv2.CV_64F, (21, 21))
        _, _, fftFiltered = toFFT(imgValues)

    return imgValues, fftFiltered


#########################################
# MAIN
#########################################

# keep original image under imgOG
imgOG = cv2.imread('rand.jpg')
# keep filtered image under imgFiltered
imgFiltered = imgOG.copy()
imgFiltered = cv2.cvtColor(imgOG, cv2.COLOR_BGR2HSV)
imgOG = cv2.cvtColor(imgOG, cv2.COLOR_BGR2RGB)

# get v-channel from HSV format to apply filter
imgValues = imgFiltered[:, :, 2]

_, _, fftOG = toFFT(imgValues)
plt.subplot(221), plt.imshow(imgOG), plt.title('Original Image')
plt.axis('off')
plt.subplot(222), plt.imshow(fftOG, 'gray'), plt.title('original freq')
plt.axis('off')


imgValues, fftFiltered = chooseFilter(imgValues, 'laplacian')
# overwrite v-vhannel in filtered image variable
imgFiltered[:, :, 2] = imgValues
imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_HSV2RGB)
plt.subplot(223), plt.imshow(imgFiltered), plt.title('Effect after filtering')
plt.axis('off')
plt.subplot(224), plt.imshow(fftFiltered, 'gray'), plt.title('filter frequency')
plt.axis('off')

# imgValues, fftFiltered = chooseFilter(imgValues, 'median')
# # overwrite v-vhannel in filtered image variable
# imgFiltered[:, :, 2] = imgValues
# imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_HSV2RGB)
# plt.subplot(223), plt.imshow(imgFiltered), plt.title('Effect after filtering')
# plt.axis('off')
# plt.subplot(224), plt.imshow(fftFiltered, 'gray'), plt.title('filter frequency')
# plt.axis('off')

# imgValues, fftFiltered = chooseFilter(imgValues, 'highpass')
# # overwrite v-vhannel in filtered image variable
# imgFiltered[:, :, 2] = imgValues
# imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_HSV2RGB)
# plt.subplot(223), plt.imshow(imgFiltered), plt.title('Effect after filtering')
# plt.axis('off')
# plt.subplot(224), plt.imshow(fftFiltered, 'gray'), plt.title('filter frequency')
# plt.axis('off')

# imgValues, fftFiltered = chooseFilter(imgValues, 'laplacian')
# # overwrite v-vhannel in filtered image variable
# imgFiltered[:, :, 2] = imgValues
# imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_HSV2RGB)
# plt.subplot(223), plt.imshow(imgFiltered), plt.title('Effect after filtering')
# plt.axis('off')
# plt.subplot(224), plt.imshow(fftFiltered, 'gray'), plt.title('filter frequency')
# plt.axis('off')


plt.show()