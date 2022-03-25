#Video Playlist: https://www.youtube.com/playlist?list=PLHae9ggVvqPgyRQQOtENr6hK0m1UquGaG

import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2
import numpy as np
import scipy    # pip install scipy
from scipy import ndimage
import scipy.fftpack as fp
import matplotlib.pyplot as plt
from skimage.io import imread    # pip install scikit-image


imgOG = cv2.imread('lena.png') # load an image

##################################
# USING V-CHANNEL HIGH-PASS
##################################
imgHSV = cv2.cvtColor(imgOG, cv2.COLOR_BGR2HSV)
imgValues = imgHSV[:,:,2]

dft = cv2.dft(np.float32(imgValues), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

rows, cols = imgValues.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.ones((rows, cols, 2), np.uint8)
r = 80
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 0

fshift = dft_shift * mask
fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

imgFiltered = imgHSV.copy()
imgFiltered[:,:,2] = img_back
imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_HSV2BGR)

imgOG = cv2.cvtColor(imgOG, cv2.COLOR_BGR2RGB)
imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_BGR2RGB)



imgFinal = imgOG.copy()

##################################
# USING BGR-CHANNELS HIGH-PASS
##################################
for i in range(0,3):
    channel = imgFinal[:,:,i]
    dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = channel.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.ones((rows, cols, 2), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0

    fshift = dft_shift * mask
    fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    f_ishift = np.fft.ifftshift(fshift)
    channelBack = cv2.idft(f_ishift)
    channelBack = cv2.magnitude(channelBack[:,:,0], channelBack[:,:,1])

    imgFinal[:,:,i] = channelBack

imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB)

##################################
# USING V-CHANNEL LOW-PASS
##################################
imgHSV = cv2.cvtColor(imgOG, cv2.COLOR_BGR2HSV)
imgValues = imgHSV[:,:,2]

dft = cv2.dft(np.float32(imgValues), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

rows, cols = imgValues.shape
crow, ccol = int(rows / 2), int(cols / 2)

mask = np.zeros((rows, cols, 2), np.uint8)
r = 100
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1

fshift = dft_shift * mask
fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

imgFiltered = imgHSV.copy()
imgFiltered[:,:,2] = img_back
imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_HSV2BGR)

# imgOG = cv2.cvtColor(imgOG, cv2.COLOR_BGR2RGB)
imgFiltered = cv2.cvtColor(imgFiltered, cv2.COLOR_BGR2RGB)




# imgFinal = imgOG.copy()

# ##################################
# # USING BGR-CHANNELS LOW-PASS
# ##################################
# for i in range(0,3):
#     channel = imgFinal[:,:,i]
#     dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
#     dft_shift = np.fft.fftshift(dft)
#     magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

#     rows, cols = channel.shape
#     crow, ccol = int(rows / 2), int(cols / 2)

#     mask = np.zeros((rows, cols, 2), np.uint8)
#     r = 100
#     center = [crow, ccol]
#     x, y = np.ogrid[:rows, :cols]
#     mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
#     mask[mask_area] = 1

#     fshift = dft_shift * mask
#     fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
#     f_ishift = np.fft.ifftshift(fshift)
#     channelBack = cv2.idft(f_ishift)
#     channelBack = cv2.magnitude(channelBack[:,:,0], channelBack[:,:,1])

#     imgFinal[:,:,i] = channelBack

# imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB)

# (imgR, imgG, imgB) = cv2.split(imgFinal)
# imgMerge = cv2.merge([imgR, imgG, imgB])
# plt.subplot(221), plt.imshow(imgR), plt.title('r channel'), plt.axis('off')
# plt.subplot(222), plt.imshow(imgG), plt.title('g channel'), plt.axis('off')
# plt.subplot(223), plt.imshow(imgB), plt.title('b channel'), plt.axis('off')
# plt.subplot(224), plt.imshow(imgMerge), plt.title('all channels'), plt.axis('off')

fig = plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(imgOG), plt.title('original image'), plt.axis('off')
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('original fft'), plt.axis('off')
plt.subplot(223), plt.imshow(imgFinal), plt.title('inverse fft'), plt.axis('off')
plt.subplot(224), plt.imshow(fshift_mask_mag, cmap='gray'), plt.title('fft + mask'), plt.axis('off')

plt.show()