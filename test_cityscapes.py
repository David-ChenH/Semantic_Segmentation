import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, warnings
from keras.models import *
from keras.layers import *
import pandas as pd
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from create_dataset import *

color_map = {
    '0': (0, 0, 0),
    '1': (0, 0, 0),
    '2': (0, 0, 0),
    '3': (0, 0, 0),
    '4': (0, 0, 0),
    '5': (111, 74, 0),
    '6': (81, 0, 81),
    '7': (128, 64, 128),
    '8': (244, 35, 232),
    '9': (250, 170, 160),
    '10': (230, 150, 140),
    '11': (70, 70, 70),
    '12': (102, 102, 156),
    '13': (190, 153, 153),
    '14': (180, 165, 180),
    '15': (150, 100, 100),
    '16': (150, 120, 90),
    '17': (153, 153, 153),
    '18': (153, 153, 153),
    '19': (250, 170, 30),
    '20': (220, 220, 0),
    '21': (107, 142, 35),
    '22': (152, 251, 152),
    '23': (70, 130, 180),
    '24': (220, 20, 60),
    '25': (255, 0, 0),
    '26': (0, 0, 142),
    '27': (0, 0, 70),
    '28': (0, 60, 100),
    '29': (0, 0, 90),
    '30': (0, 0, 110),
    '31': (0, 80, 100),
    '32': (0, 0, 230),
    '33': (119, 11, 32),
    '-1': (0, 0, 142),
}


def color_image(image, color_map, class_number):
    if len(image.shape) == 3:
        image = image[:, :, 0]
    seg_img = np.zeros((image.shape[0], image.shape[1], 3)).astype('float')

    for c in range(0, class_number):
        segc = (image == c)
        seg_img[:, :, 0] += (segc * (color_map["%d" % c][0]))
        seg_img[:, :, 1] += (segc * (color_map["%d" % c][1]))
        seg_img[:, :, 2] += (segc * (color_map["%d" % c][2]))

    return seg_img.astype("uint8")


def IoU(ground_truth, image_predict, class_number):
    IoUs = []
    for c in range(class_number):
        if np.sum(ground_truth == c) != 0:
            TP = np.sum((ground_truth == c) & (image_predict == c))
            FP = np.sum((ground_truth != c) & (image_predict == c))
            FN = np.sum((ground_truth == c) & (image_predict != c))
            IoU = TP / float(TP + FP + FN)
        else:
            IoU = -1
        IoUs.append(IoU)
    IoUs = np.asarray(IoUs)

    mIoU = np.sum(ground_truth == image_predict) / (224 * 224)
    return IoUs, mIoU


def test():
    print("Starting create prediction images. Please wait...")
    if not os.path.exists("./cityscapes/prediction"):
        os.makedirs("./cityscapes/prediction")

    images = []
    ground_truths = []
    all_IoUs = []
    all_mIoU = []
    class_number = 34
    my_model = keras.models.load_model("./cityspace_FCN_my_model.h5")
    count = 0
    for file in os.listdir("./cityscapes/new_dataset/validation/annotations"):
        img = cv2.imread("./cityscapes/new_dataset/validation/images/" + file)
        annotation = cv2.imread("./cityscapes/new_dataset/validation/annotations/" + file)
        img = np.asarray([img])

        predict = my_model.predict(img)
        predict = np.argmax(predict, axis=3)
        predict = predict[0]
        annotation = annotation[:, :, 0]

        tmp_IoUs, tmp_mIoU = IoU(annotation, predict, class_number)
        all_IoUs.append(tmp_IoUs)
        all_mIoU.append(tmp_mIoU)

        color_predict = color_image(predict, color_map, class_number)
        color_annotation = color_image(annotation, color_map, class_number)

        color_predict = cv2.cvtColor(color_predict, cv2.COLOR_BGR2RGB)
        color_annotation = cv2.cvtColor(color_annotation, cv2.COLOR_BGR2RGB)

        cv2.imwrite("./cityscapes/prediction/pre_" + file, color_predict)
        cv2.imwrite("./cityscapes/prediction/ann_" + file, color_annotation)

        np.save("./cityscapes/all_IoUs.npy", np.asarray(all_IoUs))
        np.save("./cityscapes/all_mIoU.npy", np.asarray(all_mIoU))

    print("Now you can find the prediction image in ./cityscapes/prediction/")


if __name__ == '__main__':
    test()
    all_mIoU = np.load("./cityscapes/all_mIoU.npy")
    print(np.mean(all_mIoU))
