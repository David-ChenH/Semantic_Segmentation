import cv2
import os


def create_dataset():

    print("Starting to generate new dataset from cityscapes dataset. This process may take five minutes.")
    print("Please wait.....")

    if not os.path.exists("./cityscapes/new_dataset/training/annotations"):
        os.makedirs("./cityscapes/new_dataset/training/annotations")

    if not os.path.exists("./cityscapes/new_dataset/training/images"):
        os.makedirs("./cityscapes/new_dataset/training/images")

    if not os.path.exists("./cityscapes/new_dataset/validation/images"):
        os.makedirs("./cityscapes/new_dataset/validation/images")

    if not os.path.exists("./cityscapes/new_dataset/validation/annotations"):
        os.makedirs("./cityscapes/new_dataset/validation/annotations")

    i = -1
    for folder in os.listdir("./cityscapes/leftImg8bit/train"):
        if folder != ".DS_Store":
            for file in os.listdir("./cityscapes/leftImg8bit/train/" + folder):
                i += 1
                file_name = "./cityscapes/leftImg8bit/train/" + folder + "/" + file
                image = cv2.imread(file_name)
                image = cv2.resize(image, (224,224))
                cv2.imwrite("./cityscapes/new_dataset/training/images/%0.6d.png" % i, image)

                part = file.split("_")
                annotation_name = "./cityscapes/gtFine_trainvaltest/gtFine/train/" + folder + "/" + part[0] + "_" + part[1] + "_" + part[2] + "_gtFine_labelIds.png"
                annotation = cv2.imread(annotation_name)
                annotation = cv2.resize(annotation, (224, 224))
                cv2.imwrite("./cityscapes/new_dataset/training/annotations/%0.6d.png" % i, annotation)

    i = -1
    for folder in os.listdir("./cityscapes/leftImg8bit/val"):
        if folder != ".DS_Store":
            for file in os.listdir("./cityscapes/leftImg8bit/val/" + folder):
                i += 1
                file_name = "./cityscapes/leftImg8bit/val/" + folder + "/" + file
                image = cv2.imread(file_name)
                image = cv2.resize(image, (224, 224))
                cv2.imwrite("./cityscapes/new_dataset/validation/images/%0.6d.png" % i, image)

                part = file.split("_")
                annotation_name = "./cityscapes/gtFine_trainvaltest/gtFine/val/" + folder + "/" + part[0] + "_" + part[1] + "_" + \
                                  part[2] + "_gtFine_labelIds.png"
                annotation = cv2.imread(annotation_name)
                annotation = cv2.resize(annotation, (224, 224))
                cv2.imwrite("./cityscapes/new_dataset/validation/annotations/%0.6d.png" % i, annotation)

    print("Successfully generate new dataset! You can see the new dataset on ./cityscapes/new_dataset.")

