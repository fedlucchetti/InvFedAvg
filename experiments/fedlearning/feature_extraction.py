from keras.models import Model
from keras.applications.vgg19        import VGG19
import time
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, AveragePooling2D, AveragePooling1D,GlobalAveragePooling2D
from keras.layers import Dense,Flatten,Dropout,Reshape, GlobalAveragePooling1D
from keras import backend as K

class FeatureExtraction(object):
    def __init__(self,args=None):
        if args!=None:
            self.options = args.options.dataset_type

    def create_model_vgg19(self,input_shape):
        vgg19 = VGG19(include_top = False, weights = "imagenet", input_shape = (input_shape))
        model = Sequential()
        for layer in vgg19.layers:
            model.add(layer)
        for layer in model.layers:
            layer.trainable = False
        model.add(Reshape((1,512)))
        model.add(AveragePooling1D(2,data_format='channels_first'))
        model.add(Flatten())
        return model
    
    def resize(self,img):
        numberofImage = img.shape[0]
        new_array = np.zeros((numberofImage,48,48,3))
        for i in range(numberofImage):
            new_array[i] = tf.image.resize(img[i],(48,48))
        return new_array

    def extract(self,inputs):
        inputs = self.resize(inputs)
        input_shape  = inputs.shape[1::]
        model = self.create_model_vgg19(input_shape)
        return model.predict(inputs)

if __name__ == '__main__':
    fe = FeatureExtraction()
    from keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    features = fe.extract(X_train)
    print(features.shape)


