Requirements
1. tensorflow 1.8.0
2. python 3.6.5
3. Opencv 3.4.4.19
4. Keras 2.2.4

Setup
1. Download ADEChallengeData2016 Dataset Scene Parsing from http://sceneparsing.csail.mit.edu/; 
   download CitySpace Dataset gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip from https://www.cityscapes-dataset.com/downloads/; 
   download vgg16 pretrain model vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 from https://github.com/fchollet/deep-learning-models/releases

2. Extract the data to data folder with structure like below:

    - vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    - cityscapes.py
    - small_ade.py
    - segnet.py
    - color150.mat
    - readme.txt
    - test_ade_fcn.py
    - test_cityscapes.py
    - test_segnet.py
    - ADEChallengeData2016
        - annotations
            - training 
            - validation
        - images
            - training 
            - validation
        - objectinfo150.txt
        - sceneCategories.txt
    - cityscapes
        - gtFine_trainvaltest
            - gtFine 
                - test
                - train
                - val
            - license.txt
            - README
        - leftImg8bit
            - test
            - train
            - val

Usage

1. Train 120 epochs street scene in ADEChallengeData2016 Dataset with FCN. The model, weight and tensorboard documents will be generated after training.
$ python small_ade.py 120

2. Train 200 epochs CitySpace Dataset with FCN.The model, weight and tensorboard documents will be generated after training.
$ python cityscapes.py 200 

3. Train 50 epochs street scene in ADEChallengeData2016 Dataset with Segnet.The model, weight and tensorboard documents will be generated after training.
$ python segnet.py 50

4. Test model and generate segmented image after training.
$ python test_segnet.py
$ python test_ade_fcn.py
$ python test_cityscapes.py
