import cv2
import numpy as np
from matplotlib import pyplot as plt

def showOtsu(img):
    # global thresholding
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [img, 0, th1,
            img, 0, th2,
            blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
            'Original Noisy Image','Histogram',"Otsu's Thresholding",
            'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

    for i in range(0,3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    
    plt.show()

def showBinarized(img):
    # #binarize img
    ret, imgResult = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    titles = ['Original Image','Binarized Image']
    images = [img, imgResult]

    for i in range(0,2):
         plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
         plt.title(titles[i])
         plt.xticks([]),plt.yticks([])

    plt.show()

def countBlackAndWhitePixels(img):
    number_of_white_pix = np.sum(img == 255) 
    number_of_black_pix = np.sum(img == 0) 
    print('Number of white pixels:', number_of_white_pix) 
    print('Number of black pixels:', number_of_black_pix)

img1 = cv2.imread("images\\unhealthy\\cancer.png", 0) 
#img = cv2.imread("images\\unhealthy\\metastase.png", 0) 
#img = cv2.imread("images\\unhealthy\\pneumonia.jpeg", 0) 
img2 = cv2.imread("images\\healthy\\IM-0001-0001.jpeg",0) 
#img = cv2.imread("images\\healthy\\IM-0007-0001.jpeg",0) 
#img = cv2.imread("images\\healthy\\IM-0011-0001.jpeg",0) 


#simple threshold 
#showBinarized(img)

#otsu method
showOtsu(img1)
showOtsu(img2)


