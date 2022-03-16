import cv2
import numpy as np
import scipy
from scipy import ndimage
import scipy.fftpack as fp
import matplotlib.pyplot as plt


#########################################
# HIGH PASS FILTER
#########################################

# Transfer parameters are Fourier transform spectrogram and filter size
def highPassFiltering(img, size):
    h, w = img.shape[0:2]  # Getting image properties
    # Find the center point of the Fourier spectrum
    h1, w1 = int(h/2), int(w/2)
    # Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 0
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0
    return img


#########################################
# LOW PASS FILTER
#########################################

# Transfer parameters are Fourier transform spectrogram and filter size
def lowPassFiltering(img, size):
    h, w = img.shape[0:2]  # Getting image properties
    # Find the center point of the Fourier spectrum
    h1, w1 = int(h/2), int(w/2)
    # Define a blank black image with the same size as the Fourier Transform Transfer
    img2 = np.zeros((h, w), np.uint8)
    # Center point plus or minus half of the filter size, forming a filter size that defines the size, then set to 1, preserving the low frequency part
    img2[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 1
    # A low-pass filter is obtained by multiplying the defined low-pass filter with the incoming Fourier spectrogram one-to-one.
    img3 = img2*img
    return img3


#########################################
# CHOOSE FILTER
#########################################

def RGBFilter(comp, filter):
    img_dft = np.fft.fft2(comp)
    # Move frequency domain from upper left to middle
    dft_shift = np.fft.fftshift(img_dft)

    if filter == 'hpf':
        dft_shift = highPassFiltering(dft_shift, 200)
    elif filter == 'lpf':
        dft_shift = lowPassFiltering(dft_shift, 200)

    res = np.log(np.abs(dft_shift))
    # Move the frequency domain from the middle to the upper left corner
    idft_shift = np.fft.ifftshift(dft_shift)
    ifimg = np.fft.ifft2(idft_shift)  # Fourier library function call
    ifimg = np.abs(ifimg)

    return res, ifimg


#########################################
# CHOOSE FILTER MAIN
#########################################

def spatialFiltering(imgFiltered, filter):
    imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_BGR2HSV)
    imgValues = imgFiltered[:, :, 2]

    if filter == 'lowpass':
        # src : source file ---- ddepth : depth of output image ---- ksize : blurring kernel size
        imgValues = cv2.boxFilter(imgValues, -1, (51, 51))

    elif filter == 'median':
        # median acts as a=low-pass filter ---- blurring effect
        # src : source file ---- ksize: int kernel size
        imgValues = cv2.medianBlur(imgValues, 9)

    elif filter == 'highpass':
        # kernel = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
        # imgValues = ndimage.convolve(imgValues, kernel)
        imgValues = cv2.boxFilter(imgValues, -1, (15, 15))
        imgValues = imgFiltered[:, :, 2] - imgValues

    elif filter == 'laplacian':
        # laplacian acts as hig-pass filter ---- edge detector
        # src : source file ---- ddepth : depth of output image ---- ksize : blurring kernel size
        imgValues = cv2.Laplacian(imgValues, -1, (11, 11))

    imgFiltered[:, :, 2] = imgValues
    imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_HSV2BGR)

    return imgFiltered, imgValues


#########################################
# MAIN
#########################################

# keep original image under imgOG
imgOG = cv2.imread('rand.jpg')
# keep filtered image under imgFiltered
imgFiltered = imgOG.copy()

imgFiltered, imgValues = spatialFiltering(imgFiltered, 'lowpass')


# # high-pass && low-pass filtering implementation

# imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_BGR2HSV)
# imgValues = imgFiltered[:,:,2]

# res, ifimg = RGBFilter(imgValues, 'hpf')

# imgFiltered[:,:,2] = ifimg
# imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_HSV2BGR)

# cv2.imshow('res', res)


cv2.imshow('OG', imgOG)
cv2.imshow('filtered', imgFiltered)

F1 = fp.fft2((imgValues).astype(float))
F2 = fp.fftshift(F1)

plt.figure(figsize=(10, 10))
plt.imshow((20*np.log10(0.1 + F2)).astype(int), cmap=plt.cm.gray)
plt.show()



cv2.waitKey(0)
cv2.destroyAllWindows()
