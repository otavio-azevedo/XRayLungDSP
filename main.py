import cv2
import numpy as np
from matplotlib import pyplot as plt

def contrastLungAreas(img):
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

# Carrega imagens para a aplicação
#img = cv2.imread("images\\healthy\\1.png", 0)
img = cv2.imread("images\\unhealthy\\2.png", 0)
img_colored = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Threshold to segment the area of the lungs
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, img = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

# Destaca área dos pulmoes
contrastLungAreas(img)

# Remove ruídos
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))  # "erase" the small points in the resulting mask

# Obtem os contornos da imagem
contours, hierarchy = cv2.findContours(
  cv2.bitwise_not(img),
  cv2.RETR_TREE,
  cv2.CHAIN_APPROX_SIMPLE)

# Ordena o vetor de contours
contoursSorted = sorted(contours,key=cv2.contourArea)

# Remove os dois elementos maiores (pulmoes)
#print(len(contoursSorted))
contoursSorted.remove(contoursSorted[-1])
contoursSorted.remove(contoursSorted[-1])
#print(len(contoursSorted))

# Desenha os contornos na imagem colorida
result = cv2.drawContours(img_colored, contoursSorted, -1,(0,0,255),3)

# Printa a posição e a area de cada contorno restante
for i in range(len(contoursSorted)):
    # Se área restante foi maior que 500, considera que existe a presença de possível tumor no CT
    if cv2.contourArea(contoursSorted[i]) > 500: 
        boolExistMass = True
        break
    else:
        boolExistMass = False
    
print(boolExistMass)

cv2.imshow('Detected contours', result)
cv2.waitKey()
cv2.destroyAllWindows()

## TODO: 
# INFORMAR EM QUAL DOS PULMÕES ESTÁ O TUMOR?
# # Cut the image in half
# height, width, channels = result.shape
# right = img_colored[:, :width // 2] #right lung
# left = img_colored[:, width // 2:] #left lung
# cv2.imshow('Resultado', right)
# cv2.imshow('Resultado', left)