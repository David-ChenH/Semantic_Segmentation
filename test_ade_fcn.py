from scipy.io import loadmat
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd
import cv2, os
import numpy as np
import matplotlib.pyplot as plt

import random
import seaborn as sns

color_map = loadmat("./color150.mat")


def give_color_to_seg_img(seg, n_classes):

    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (color_map["colors"][c-1][0]))
        seg_img[:, :, 1] += (segc * (color_map["colors"][c-1][1]))
        seg_img[:, :, 2] += (segc * (color_map["colors"][c-1][2]))

    return seg_img.astype("uint8")

def test():
    print("Starting create prediction images. Please wait...")
    if not os.path.exists("./ADEChallengeData2016/prediction_fcn"):
        os.makedirs("./ADEChallengeData2016/prediction_fcn")

    images = []
    ground_truths = []
    all_IoUs = []
    all_mIoU = []
    class_number = 150
    my_model = keras.models.load_model("./small_ade_FCN_model.h5")
    count = 1784
    for i in range(10):
        img = cv2.imread("./ADEChallengeData2016/images/validation/ADE_val_%0.8d.jpg" % (count + i))
        img = cv2.resize(img, (224,224))

        annotation = cv2.imread("./ADEChallengeData2016/annotations/validation/ADE_val_%0.8d.png" % (count+i))
        annotation = cv2.resize(annotation, (224,224))
        img = np.asarray([img])

        predict = my_model.predict(img)
        predict = np.reshape(predict, (1, 224, 224, 151))
        predict = np.argmax(predict, axis=3)
        predict = predict[0]
        annotation = annotation[:, :, 0]

        color_predict = give_color_to_seg_img(predict, class_number)
        color_annotation = give_color_to_seg_img(annotation, class_number)

        color_predict = cv2.cvtColor(color_predict, cv2.COLOR_BGR2RGB)
        color_annotation = cv2.cvtColor(color_annotation, cv2.COLOR_BGR2RGB)

        cv2.imwrite("./ADEChallengeData2016/prediction_fcn/pre_" + str(i) + ".jpg", color_predict)
        cv2.imwrite("./ADEChallengeData2016/prediction_fcn/ann_" + str(i) + ".jpg", color_annotation)
        count += 1

    print("Now you can find the prediction image in ./ADEChallengeData2016/prediction_segnet")


if __name__ == '__main__':
    test()