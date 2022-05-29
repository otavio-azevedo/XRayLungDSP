import cv2
import numpy as np
from matplotlib import pyplot as plt

def showOtsu(img):
    # global thresholding
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

    for i in range(0, 3):
        plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

    plt.show()

def countBlackAndWhitePixels(img):
    number_of_white_pix = np.sum(img == 255)
    number_of_black_pix = np.sum(img == 0)
    print('Number of white pixels:', number_of_white_pix)
    print('Number of black pixels:', number_of_black_pix)

#img = cv2.imread("images\\healthy\\3.png", 0)
img = cv2.imread("images\\unhealthy\\1.png", 0)

# #Erode & Dilate to remove noises and improve the result of the next operation (threshold)
# kernel = np.ones((20,20),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1)
# dilation = cv2.dilate(img,kernel,iterations = 1)


##Threshold to segment the area of the lungs
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, img = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

# Destaca área dos pulmoes
for x in range(img.shape[1]):
    # Fill dark top pixels:
    if img[0, x] == 0:
        cv2.floodFill(img, None, seedPoint=(x, 0), newVal=255, loDiff=3, upDiff=3)  # Fill the background with white color

    # Fill dark bottom pixels:
    if img[-1, x] == 0:
        cv2.floodFill(img, None, seedPoint=(x, img.shape[0]-1), newVal=255, loDiff=3, upDiff=3)  # Fill the background with white color

for y in range(img.shape[0]):
    # Fill dark left side pixels:
    if img[y, 0] == 0:
        cv2.floodFill(img, None, seedPoint=(0, y), newVal=255, loDiff=3, upDiff=3)  # Fill the background with white color

    # Fill dark right side pixels:
    if img[y, -1] == 0:
        cv2.floodFill(img, None, seedPoint=(img.shape[1]-1, y), newVal=255, loDiff=3, upDiff=3)  # Fill the background with white color

# Remove pontos menores
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
lower_white = np.array([220, 220, 220], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)
img = cv2.inRange(img, lower_white, upper_white)  # could also use threshold
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))  # "erase" the small points in the resulting mask
img = cv2.bitwise_not(img)  # invert mask

# Cut the image in half
height, width = img.shape
width_cutoff = width // 2
right = img[:, :width_cutoff] #right lung
left = img[:, width_cutoff:] #left lung

## TODO: 
    # COMO IDENTIFICAR A PRESENÇA DO TUMOR (MANCHA PRETA NO BRANCO)?
    # INFORMAR EM QUAL DOS PULMÕES ESTÁ O TUMOR

#cv2.imshow('Resultado', right)
#cv2.imshow('Resultado', left)
cv2.imshow('Resultado', img)

cv2.waitKey()
cv2.destroyAllWindows()



# cv2.imshow("Teste",cv2.resize(thres, (400, 300)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()