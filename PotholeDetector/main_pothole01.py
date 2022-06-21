import cv2
import os
import numpy as np

with open('pothole.names', 'rt') as f:
    classesNomes = f.read().splitlines()
print(classesNomes)

# Classificador
classificador = cv2.CascadeClassifier()


def getPictureByName():
    paths = [os.path.join('potholes', p) for p in os.listdir('potholes')]
    # print(paths)
    potholes = []
    classes = []

    for picturePath in paths:
        # print(picturePath)
        pothole = cv2.imread(picturePath)  # Matrizes das imagens
        potholeGray = cv2.imread(picturePath, cv2.COLOR_BGR2GRAY)  # Matrizes das imagens em escala de cinza
        name = os.path.split(picturePath)[-1].split('_')[0]
        #print(name)
        if name == 'pothole':
            classes.append(1)
        elif name == 'waterpothole':
            classes.append(2)
        elif name == 'mixpothole':
            classes.append(3)
        potholes.append(potholeGray)

    return np.array(classes), potholes


classes, potholes = getPictureByName()
print(classes)
classificador(potholes, classes)
"""try:
    classificador.train(potholes, classes)
except:
    print("Erro")"""

