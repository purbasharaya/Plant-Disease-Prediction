Plant Disease Prediction using CNN
==================================

[](https://github.com/purbasharaya/Plant-Disease-Prediction?tab=readme-ov-file#introduction)

Introduction
------------

[](https://github.com/purbasharaya/Plant-Disease-Prediction#introduction)

Welcome to the Plant Disease Prediction project! This project is designed to assist farmers and agronomists in identifying plant diseases using Convolutional Neural Networks (CNN) and TensorFlow. By harnessing the power of computer vision and deep learning, this system aims to provide early detection of diseases, helping to mitigate their impact on crops and improve overall agricultural productivity.

Dataset
-------

[](https://github.com/purbasharaya/Plant-Disease-Prediction#dataset)

The project utilizes a carefully curated dataset comprising images of various plants affected by different diseases. Each image in the dataset is labeled with the corresponding plant disease category. The dataset has been preprocessed and is ready for training the CNN model. As we wanted more data to increase the accuracy we used data augmentation using the ImageDataGenerator library from keras and added certain parameters for the same.
- Dataset Link - [Click Here](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

CNN Model Architecture
----------------------

[](https://github.com/purbasharaya/Plant-Disease-Prediction#cnn-model-architecture)

The Convolutional Neural Network (CNN) serves as the core of the Plant Disease Prediction system. The CNN model architecture is designed to extract relevant features from input images, enabling it to distinguish between healthy plants and those suffering from different diseases. The model includes the following key components:

-   Input Layer: The images are fed into the network for processing.

-   Convolutional Layers: A series of convolutional layers with activation functions extract relevant patterns and features from the images.

-   MaxPooling Layers: MaxPooling layers downsample the feature maps to reduce spatial dimensions and enhance computational efficiency.

-   Flatten Layer: The output from the last convolutional layer is flattened into a one-dimensional vector, ready for further processing.

-   Fully Connected Layers: Dense layers process the flattened vector, enabling the model to make predictions based on the learned features.

-   Output Layer: The output layer uses the softmax activation function to predict the probability of each disease class for a given input image.

Model Training and Prediction
-----------------------------

[](https://github.com/purbasharaya/Plant-Disease-Prediction#model-training-and-prediction)

To train the CNN model, the dataset is split into training and testing sets to evaluate its performance accurately. The model is trained on the training set and fine-tuned to minimize the categorical cross-entropy loss function. The performance is assessed on the testing set to ensure generalization and prevent overfitting.

For prediction, after loading a trained model, new images can be preprocessed and fed into the model. The model will then provide predictions indicating the probability of each disease class for the input image.

Contributing
------------

[](https://github.com/purbasharaya/Plant-Disease-Prediction#contributing)

Contributions to this project are highly encouraged! If you have any improvements, new features, or dataset suggestions to add, please feel free to submit a pull request. Together, we can enhance the effectiveness of the Plant Disease Prediction system and contribute to the advancement of agricultural technology.

Acknowledgements
----------------

[](https://github.com/purbasharaya/Plant-Disease-Prediction#acknowledgements)

We extend our heartfelt appreciation to the developers and contributors of the libraries and frameworks used in this Plant Disease Prediction project. Their invaluable work enables us to create and deploy a powerful and efficient system:

-   [NumPy](https://numpy.org/)
-   [OpenCV](https://opencv.org/)
-   [Keras](https://keras.io/)
-   [TensorFlow](https://www.tensorflow.org/)
-   [scikit-learn](https://scikit-learn.org/)
