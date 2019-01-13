#!/usr/bin/env python

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
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import sys
from create_dataset import *

create_dataset()
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
    images_path = "./cityscapes/new_dataset"
    for i in range(start, end):
        image_name = "/training/images/%06d.png" % i
        image_path = images_path + image_name
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        training_images.append(image)
    training_images = np.asarray(training_images)
    return training_images

def read_train_annotations(start, end):
    # training_annotations = []
    training_annotations_labels = []
    annotations_path = "./cityscapes/new_dataset"
    for i in range(start, end):
        annotation_name = "/training/annotations/%06d.png" % i
        annotation_path = annotations_path + annotation_name
        annotation = cv2.imread(annotation_path)
        annotation = cv2.resize(annotation, (224, 224))
        annotation = annotation[:, :, 0]
        annotation_array = np.zeros((224, 224, 34))
        for class_object in range(34):
            annotation_array[:, :, class_object] = (annotation == class_object) * 1.0
        # training_annotations.append(annotation)
        training_annotations_labels.append(annotation_array)
    # training_annotations = np.asarray(training_annotations)
    training_annotations_labels = np.asarray(training_annotations_labels)
    # np.save("training_annotations.npy", training_annotations)
    # np.save("training_annotations_labels.npy", training_annotations_labels)
    return training_annotations_labels


def change_to_label(data):
    print('1')
    data_labels = []
    for image in data:
        annotation_array = np.zeros((224, 224, 34))
        for class_object in range(34):
            annotation_array[:, :, class_object] = (image == class_object) * 1.0
        data_labels.append(annotation_array)
    data_labels = np.asarray(data_labels)
    return data_labels


def read_validation_image(start, end):
    validation_images = []
    images_path = "./cityscapes/new_dataset"
    for i in range(start + 1, end + 1):
        image_name = "/validation/images/%06d.png" % i
        image_path = images_path + image_name
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        validation_images.append(image)
    validation_images = np.asarray(validation_images)
    return validation_images


def read_validation_annotations(start, end):
    validation_annotations_labels = []
    annotations_path = "./cityscapes/new_dataset"
    for i in range(start, end ):
        annotation_name = "/validation/annotations/%06d.png" % i
        annotation_path = annotations_path + annotation_name
        annotation = cv2.imread(annotation_path)
        annotation = cv2.resize(annotation, (224, 224))
        annotation = annotation[:, :, 0]
        annotation_array = np.zeros((224, 224, 34))
        for class_object in range(34):
            annotation_array[:, :, class_object] = (annotation == class_object) * 1.0
        validation_annotations_labels.append(annotation_array)
    validation_annotations_labels = np.asarray(validation_annotations_labels)
    return validation_annotations_labels



def train_batch_generation(batch_size):
    batch_number = 2975 // batch_size
    #batch_count = 0
    while True:
        x_train = []
        y_train = []
        batch_count = random.randint(0, batch_number - 1)
        x_train = read_train_image(batch_count * batch_size, (1 + batch_count) * batch_size)
        y_train = read_train_annotations(batch_count * batch_size, (1 + batch_count) * batch_size)
        #batch_count = +1
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        yield (x_train, y_train)


def val_batch_generation(batch_size):
    batch_number = 500 // batch_size
    #batch_count = 0
    while True:
        x_val = []
        y_val = []
        batch_count = random.randint(0, batch_number - 1)
        x_val = read_validation_image(batch_count * batch_size, (1 + batch_count) * batch_size)
        y_val = read_validation_annotations(batch_count * batch_size, (1 + batch_count) * batch_size)
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)
        #batch_count += 1
        yield (x_val, y_val)

VGG_Weights_path = "./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"



def FCN8(class_num, input_height=224, input_width=224):
    input_img = Input(shape=(input_height, input_width, 3))

    net = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", name="layer1_conv1",
                 data_format="channels_last")(input_img)
    net = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", name="layer1_conv2",
                 data_format="channels_last")(net)
    net = MaxPooling2D((2, 2), strides=(2, 2), name="layer1_pool", data_format="channels_last")(net)
    conv1 = net

    net = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", name="layer2_conv1",
                 data_format="channels_last")(net)
    net = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", name="layer2_conv2",
                 data_format="channels_last")(net)
    net = MaxPooling2D((2, 2), strides=(2, 2), name="layer2_pool", data_format="channels_last")(net)
    conv2 = net

    net = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="layer3_conv1",
                 data_format="channels_last")(net)
    net = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="layer3_conv2",
                 data_format="channels_last")(net)
    net = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="layer3_conv3",
                 data_format="channels_last")(net)
    net = MaxPooling2D((2, 2), strides=(2, 2), name="layer3_pool", data_format="channels_last")(net)
    pool3 = net

    net = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="layer4_conv1",
                 data_format="channels_last")(net)
    net = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="layer4_conv2",
                 data_format="channels_last")(net)
    net = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="layer4_conv3",
                 data_format="channels_last")(net)
    net = MaxPooling2D((2, 2), strides=(2, 2), name="layer4_pool", data_format="channels_last")(net)
    pool4 = net

    net = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="layer5_conv1",
                 data_format="channels_last")(net)
    net = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="layer5_conv2",
                 data_format="channels_last")(net)
    net = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="layer5_conv3",
                 data_format="channels_last")(net)
    net = MaxPooling2D((2, 2), strides=(2, 2), name="layer5_pool", data_format="channels_last")(net)
    pool5 = net

    vgg = Model(input_img, pool5)
    vgg.load_weights(VGG_Weights_path)

    n = 4096
    net = (Conv2D(n, (7, 7), activation="relu", padding="same", name="conv6", data_format="channels_last"))(net)
    net = (Conv2D(n, (1, 1), activation="relu", padding="same", name="conv7", data_format="channels_last"))(net)
    conv7 = net

    conv7_four_time = Conv2DTranspose(class_num, kernel_size=(4, 4), strides=(4, 4), use_bias=False,
                                      data_format="channels_last")(net)

    pool_four_time_1 = (Conv2D(class_num, (1, 1), activation="relu", padding="same", name="pool_four_time_1",
                               data_format="channels_last"))(pool4)
    pool_four_time_2 = (
        Conv2DTranspose(class_num, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format="channels_last"))(
        pool_four_time_1)

    pool_three_time_1 = (Conv2D(class_num, (1, 1), activation="relu", padding="same", name="pool_three_time_1",
                                data_format="channels_last"))(pool3)

    out = Add(name="add")([pool_four_time_2, pool_three_time_1, conv7_four_time])
    out = Conv2DTranspose(class_num, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format="channels_last")(
        out)
    out = (Activation("softmax"))(out)

    model = Model(input_img, out)

    return model


model = FCN8(class_num=34,
             input_height=224,
             input_width=224)
model.summary()

ticks = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
main_path = "./"
log_dir_name = "cityspace_FCN_log_%s" % ticks
log_dir_path = main_path+log_dir_name

checkpoint_dir_name = "cityspace_FCN_saved_model_%s" % ticks
checkpoint_dir_path = main_path+checkpoint_dir_name

os.mkdir(log_dir_path)
os.mkdir(checkpoint_dir_path)

tensorboard = TensorBoard(log_dir=log_dir_path)
checkpoint = ModelCheckpoint(filepath=checkpoint_dir_path+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5',monitor='val_acc',mode='auto' ,save_best_only='True')

callback_lists=[tensorboard,checkpoint]

sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
epoch_num = sys.argv[1]
hist1 = model.fit_generator(generator=train_batch_generation(45), steps_per_epoch=66, epochs=int(epoch_num), verbose=1,
                            validation_data=val_batch_generation(50),validation_steps=10, 
                            max_queue_size=3, workers=1,callbacks=callback_lists)

model.save('cityspace_FCN_my_model.h5')