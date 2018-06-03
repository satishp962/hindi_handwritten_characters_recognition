import os
import cv2
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

img_height, img_width = 28, 28

def get_data(imagePaths, verbose=100):
    # initialize the list of features and labels
    data = []
    labels = []

    # loop over the input images
    for (i, imagePath) in enumerate(imagePaths):
        # load the image and extract the class label assuming
        # that our path has the following format:
        # /path/to/dataset/{class}/{image}.jpg  
        if os.path.exists(imagePath):

            # preprocess image for black and white
            image_gray = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            # imginvert = cv2.bitwise_not(image_gray)
            image = cv2.resize(image_gray, (img_height, img_width))
            
            # get image class
            label = imagePath.split(os.path.sep)[-2]

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

    # shuffle variables
    data_sh = []
    labels_sh = []

    from random import shuffle
    index_data = list(range(len(data)))
    
    # shuffle
    print("Shuffling data and labels list.")
    shuffle(index_data)

    for i in index_data:
        data_sh.append(data[i])
        labels_sh.append(labels[i])

    data = data_sh
    labels = labels_sh
    
    data = np.array(data)
    labels = np.array(labels)

    data = data.reshape((data.shape[0], img_height * img_width))

    # show some information on memory consumption of the images
    print("\n[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

    # encode the labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # return a tuple of the data and labels
    return data, labels

if __name__ == "__main__" :
    imagePaths = list(paths.list_images('./dataset_ka_kha'))
    data, labels = get_data(imagePaths, 5000)
        
    with open('dataset_pickles/dataset_10_classes.pickle', 'wb') as f:
        pickle.dump([data, labels], f)