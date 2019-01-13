
import numpy as np
import cv2

import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd 
import os
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.layers.core import Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import sys

warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

config.gpu_options.visible_device_list = "0" 
set_session(tf.Session(config=config))   

print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__)); del keras
print("tensorflow version {}".format(tf.__version__))

def read_train_image(start, end):
    training_images = []
    images_path = "./ADEChallengeData2016/images"
    for i in range(start, end):
        num = i+16858
        image_name = "/training/ADE_train_%08d.jpg" % num
        image_path = images_path + image_name
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        training_images.append(image)
    training_images = np.asarray(training_images)
    return training_images

def read_train_annotations(start, end):
    # training_annotations = []
    training_annotations_labels = []
    annotations_path = "./ADEChallengeData2016/annotations"
    for i in range(start, end):
        num = i+16858
        annotation_name = "/training/ADE_train_%08d.png" % num
        annotation_path = annotations_path + annotation_name
        annotation = cv2.imread(annotation_path)
        annotation = cv2.resize(annotation, (224, 224))
        annotation = annotation[:, :, 0]
        annotation_array = np.zeros((224, 224, 151))
        for class_object in range(151):
            annotation_array[:, :, class_object] = (annotation == class_object) * 1.0
        training_annotations_labels.append(annotation_array)
    training_annotations_labels = np.reshape(training_annotations_labels, (len(training_annotations_labels), 224*224, 151))
    training_annotations_labels = np.asarray(training_annotations_labels)
    return training_annotations_labels




def read_validation_image(start, end):
    validation_images = []
    images_path = "./ADEChallengeData2016/images"
    for i in range(start, end):
        num = i+1777
        image_name = "/validation/ADE_val_%08d.jpg" % num
        image_path = images_path + image_name
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        validation_images.append(image)
    validation_images = np.asarray(validation_images)
    return validation_images


def read_validation_annotations(start, end):
    validation_annotations_labels = []
    annotations_path = "./ADEChallengeData2016/annotations"
    for i in range(start, end):
        num = i+1777
        annotation_name = "/validation/ADE_val_%08d.png" % num
        annotation_path = annotations_path + annotation_name
        annotation = cv2.imread(annotation_path)
        annotation = cv2.resize(annotation, (224, 224))
        annotation = annotation[:, :, 0]
        annotation_array = np.zeros((224, 224, 151))
        for class_object in range(151):
            annotation_array[:, :, class_object] = (annotation == class_object) * 1.0
        validation_annotations_labels.append(annotation_array)
    validation_annotations_labels = np.reshape(validation_annotations_labels, (len(validation_annotations_labels), 224*224, 151))
    validation_annotations_labels = np.asarray(validation_annotations_labels)
    return validation_annotations_labels


def train_batch_generation(batch_size):
    batch_number = 2038 // batch_size
    while True:
        x_train = []
        y_train = []
        batch_count = random.randint(0, batch_number - 1)
        x_train = read_train_image(batch_count * batch_size, (1 + batch_count) * batch_size)
        y_train = read_train_annotations(batch_count * batch_size, (1 + batch_count) * batch_size)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        yield (x_train, y_train)

def val_batch_generation(batch_size):
    batch_number = 97 // batch_size

    while True:
        x_val = []
        y_val = []
        batch_count = random.randint(0, batch_number - 1)
        x_val = read_validation_image(batch_count * batch_size, (1 + batch_count) * batch_size)
        y_val = read_validation_annotations(batch_count * batch_size, (1 + batch_count) * batch_size)
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)
        yield (x_val, y_val)


def SegNet(input_shape=(224, 224, 3), classes=151):

    img_input = Input(shape=input_shape)
    net = img_input

    net = Conv2D(64, (3, 3), padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)

    net = Conv2D(128, (3, 3), padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)

    net = Conv2D(256, (3, 3), padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)

    net = Conv2D(512, (3, 3), padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)

    net = Conv2D(512, (3, 3), padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)

    net = UpSampling2D(size=(2, 2))(net)
    net = Conv2D(256, (3, 3), padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)

    net = UpSampling2D(size=(2, 2))(net)
    net = Conv2D(128, (3, 3), padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)

    net = UpSampling2D(size=(2, 2))(net)
    net = Conv2D(64, (3, 3), padding="same")(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)

    net = Conv2D(classes, (1, 1), padding="valid")(net)
    net = Reshape((input_shape[0] * input_shape[1], classes))(net)
    net = Activation("softmax")(net)
    model = Model(img_input, net)
    return model


model = SegNet()
model.summary()

ticks = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
main_path = "./"
log_dir_name = "small_ade_Segnet_log_%s" % ticks
log_dir_path = main_path+log_dir_name

checkpoint_dir_name = "small_ade_Segnet_saved_model_%s" % ticks
checkpoint_dir_path = main_path+checkpoint_dir_name

os.mkdir(log_dir_path)
os.mkdir(checkpoint_dir_path)

tensorboard = TensorBoard(log_dir=log_dir_path)
checkpoint = ModelCheckpoint(filepath=checkpoint_dir_path+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5',monitor='val_acc',mode='auto' ,save_best_only='True')

callback_lists=[tensorboard,checkpoint]


model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
epoch_num = sys.argv[1]
hist1 = model.fit_generator(generator=train_batch_generation(21), steps_per_epoch=97, epochs=int(epoch_num), verbose=1,
                            validation_data=val_batch_generation(20),validation_steps=4, 
                            max_queue_size=3, workers=1,callbacks=callback_lists)

model.save('small_ade_Segnet_model.h5')