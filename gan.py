import keras
import numpy as np
import matplotlib.pyplot as plot
from tqdm import tqdm
from keras.datasets import mnist
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import SGD

# Center the data around 0, and change values to be in range [-1,1]
def preprocess_data(data):
    data = data[..., np.newaxis]
    data = (data.astype(np.float32) - 127.5) / 127.5
    return data

def generator(z_dim):
    """
    Create the generator network
    :param z_dim: Dimensions of input noise vector, z
    """
        #Input tensor will be batches of z_dim-dimensional vectors
        input_x = Input(shape=(z_dim,))
        x = input_x
        
        x = Dense(1024)(x)
        x = Activation('tanh')(x)
        x = Dense(128*7*7)(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Reshape((7, 7, 128))(x)

        x = UpSampling2D(size=(2, 2))(x) # 14x14
        x = Convolution2D(64, 5, 5, border_mode='same')(x)
        x = Activation('tanh')(x)

        x = UpSampling2D(size=(2, 2))(x) # 28x28
        x = Convolution2D(1, 5, 5, border_mode='same')(x)
        x = Activation('tanh')(x)

        return Model(input=input_x, output=x)

def discriminator():
    """
    Create the discriminator network
    """
    #greyscale 28x28 images as input
    input_x = Input(shape=(28, 28, 1))
    x = input_x
                
    #TODO:

def main():
    #load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train, X_test = preprocess_data(x_train), preprocess_data(x_test)

    print(X_train[0])

if __name__ == "__main__":
    main()
