import imghdr
import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread("rand.jpg", cv2.IMREAD_COLOR)
img = cv2.imread('rand.jpg')
# cv2.imshow('wtv', img)

# print(img.shape)

imgR = img[:, :, 2]
imgG = img[:, :, 1]
imgB = img[:, :, 0]

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

# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# resGray, ifimgGray = RGBFilter(imgGray, 'pf')

resR, ifimgR = RGBFilter(imgR, 'lpf')
resG, ifimgG = RGBFilter(imgG, 'lpf')
resB, ifimgB = RGBFilter(imgB, 'lpf')

imgMerge = cv2.merge([ifimgB, ifimgG, ifimgR])

# print(np.shape(imgMerge))
# gray = cv2.cvtColor(imgMerge, cv2.COLOR_BGR2GRAY)
# cv2.imshow("grsu",gray)

# img_dft = np.fft.fft2(imgMerge)
# dft_shift = np.fft.fftshift(img_dft)
# res = np.log(np.abs(dft_shift))
# idft_shift = np.fft.ifftshift(dft_shift)


resMerge = cv2.merge([resB, resG, resR])


# cv2.imshow("oG", img)
# cv2.imshow("res", resMerge)
# cv2.imshow("filtered", np.int8(imgMerge))

cv2.imshow("oG", img)
cv2.imshow("res", resMerge)
cv2.imshow("filtered", np.int8(imgMerge))

cv2.waitKey(0)
cv2.destroyAllWindows()
