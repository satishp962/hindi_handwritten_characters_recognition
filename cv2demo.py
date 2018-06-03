import cv2
import os
import numpy as np
from imutils import paths
from matplotlib import pyplot as plt

classname = 0
folname = 0

for _ in range(0, 58):
    imagePaths = list(paths.list_images('./dataset_prepare/' + str(classname)))
    os.mkdir("./dataset_padded/" + str(folname))

    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label assuming
        # that our path has the following format:
        # /path/to/dataset/{class}/{image}.jpg  
        if os.path.exists(imagePath):
            # preprocess image for black and white
            image_gray = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image_gray, (32, 32))
            # imginvert = cv2.bitwise_not(image)
            BLACK = [0, 0, 0]
            constant = cv2.copyMakeBorder(image, 10 , 10, 10, 10, cv2.BORDER_CONSTANT, value=BLACK)
            
            cv2.imwrite("./dataset_padded/" + str(folname) + "/" + str(i) + ".png", constant)
    
    folname += 1
    classname += 1
