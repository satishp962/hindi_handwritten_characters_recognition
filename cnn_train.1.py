import os, glob
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import time
import os.path as path
import keras
import keras.backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, Activation

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import cv2
import pickle
import numpy as np
from imutils import paths
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

np.random.seed(2017)

MODEL_NAME = 'hindi_convnet'
EPOCHS = 25
BATCH_SIZE = 16
CHANNELS = 1
num_classes = 10
img_height = 28
img_width = 28

def load_data(datasetPath):

    # load data from the pickle file
    with open(datasetPath, 'rb') as f:
        data, labels = pickle.load(f)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    print("\n[INFO] splitting dataset into train and validation sets")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    print("\n[INFO] reshaping images")
    x_train = x_train.reshape(x_train.shape[0], img_height, img_width, CHANNELS)
    x_test = x_test.reshape(x_test.shape[0], img_height, img_width, CHANNELS)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def build_model():
    print("\n[INFO] creating model")
    model = Sequential()

    model.add(Conv2D(20, (5, 5), padding="same", input_shape=(img_height, img_width, CHANNELS)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, (5, 5), padding="same", input_shape=(img_height, img_width, CHANNELS)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    model.add(Dense(num_classes, activation='softmax'))

    # chanDim  = 1
    # # first CONV => RELU => CONV => RELU => POOL layer set
    # model.add(Conv2D(32, (5, 5), padding="same", input_shape=(28, 28, CHANNELS)))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Conv2D(32, (5, 5), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(64, (5, 5), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(Conv2D(64, (5, 5), padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(axis=chanDim))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))

    # # first (and only) set of FC => RELU layers
    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    # # # softmax classifier
    # model.add(Dense(num_classes, activation='softmax'))

    # model.summary()
    return model

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)

def train(model, x_train, y_train, x_test, y_test):

    print("\n[INFO] model training starting\n")
    model.compile(loss=keras.losses.categorical_crossentropy, \
                  optimizer=keras.optimizers.Adam(), \
                  metrics=['accuracy'])

    # define data preparation
    datagen = ImageDataGenerator(
                # featurewise_center=True,
                # featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2)
                # horizontal_flip=True,
                # vertical_flip=True)

    # fit parameters from data
    datagen.fit(x_train)
    # configure batch size and retrieve one batch of images
    # for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
	# # create a grid of 3x3 images
    #     for i in range(0, 9):
    #         plt.subplot(330 + 1 + i)
    #         plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    #     # show the plot
    #     plt.show()
    #     break

    # start timer
    start = time.time()

    # checkpoint
    # outputFolder = './output-checkpoint'
    # if not os.path.exists(outputFolder):
    #     os.makedirs(outputFolder)

    # filepath = outputFolder + "/weights-{epoch:02d}-{val_acc:.2f}.h5py"

    # epoch_num = 50
    # file_ini = outputFolder + '/weights-' + str(epoch_num) + '*'
    # filename =  glob.glob(file_ini)

    # checkpoint creation for saving model
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, \
    #                         save_best_only=False, save_weights_only=True, \
    #                         mode='auto', period=10)

    # tensorboard logs for visualization
    # tbCallBack = TensorBoard(log_dir='./logs', batch_size=BATCH_SIZE, \
                # write_grads=True, histogram_freq=2, write_graph=True, write_images=True)

    # define early stopping callback
    # earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, \
    #                       verbose=1, mode='auto')

    # fits the model on batches with real-time data augmentation:
    model_info = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(x_train) / BATCH_SIZE, epochs=EPOCHS, 
                        validation_data=(x_test, y_test))
    
    # model_info = model.fit(x_train, y_train, \
    #         batch_size=BATCH_SIZE, \
    #         epochs=EPOCHS, \
    #         verbose=1, \
    #         validation_data=(x_test, y_test)) #, \
            # callbacks=[tbCallBack])

    # end timer and print the time taken to train model
    end = time.time()
    print("\nModel took %0.2f seconds to train"%(end - start))

    # display graph accuracy and loss
    plot_model_history(model_info)

    # compute test accuracy
    print("Accuracy on test data is: %0.2f"%accuracy(x_test, y_test, model))


def export_model(saver, model, input_node_names, output_node_name):
    # Save trained model in .h5py file
    print("\n[INFO] exporting model to .h5py file")
    model.save_weights('out/hhrc_nn.h5py')

    print("\n[INFO] exporting model to .pb file\n")
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
        
    print("\n[INFO] model saved")

def main():

    print("\n[info] checking for 'out' folder if exists")
    if not path.exists('out'):
        print("\n[info] out folder not available, creating folder 'out' in the root directory")
        os.mkdir('out')

    print("\n[INFO] loading data from the dataset into variables")
    x_train, y_train, x_test, y_test = load_data('dataset_pickles/dataset_10_classes.pickle')

    # create a grid of 3x3 images
    # for i in range(0, 9):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(x_test[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    # # show the plot
    # plt.show()

    # train model
    model = build_model()
    # model.load_weights('./output-checkpoint/hhrc_nn.h5py')
    train(model, x_train, y_train, x_test, y_test)

    # export model
    export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Softmax")

if __name__ == '__main__':
    main()
