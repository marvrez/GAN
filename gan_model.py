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

def mse_4d_tf(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=(1,2,3))

class GAN(models.Sequential):
    def __init__(self, input_dim = 64):
        super().__init__()
        self.input_dim = input_dim

        #assemble both discriminator and generator
        self.generator = self.generator()
        self.discriminator = self.discriminator()

        self.add(self.generator)
        self.discriminator.trainable = False
        self.add(self.discriminator)
        
        # Print summary of networks
        generator.summary()
        discriminator.summary()
        self.summary()

        self.compile_all()

    def compile_all(self):
        # Compiling stage
        d_optimizer = SGD(lr = 0.0005, momentum = 0.9, nesterov = True)
        g_optimizer = SGD(lr = 0.0005, momentum = 0.9, nesterov = True)
        self.generator.compile(loss = mse_4d_tf, optimizer = "SGD")
        self.compile(loss = 'binary_crossentropy', optimizer = g_optimizer)
        self.discriminator.trainable = True
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = d_optimizer)

    def generator(self):
        """
        Create the generator network
        :param z_dim: Dimensions of input noise vector, z
        """
        #Input tensor will be batches of z_dim-dimensional vectors
        input_x = Input(shape = (self.input_dim,))
        x = input_x
        
        x = Dense(1024)(x)
        x = Activation('tanh')(x)
        x = Dense(128*7*7)(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        x = Reshape((7, 7, 128))(x)

        x = UpSampling2D(size = (2, 2))(x) # 14x14
        x = Convolution2D(64, 5, 5, border_mode = 'same')(x)
        x = Activation('tanh')(x)

        x = UpSampling2D(size = (2, 2))(x) # 28x28
        x = Convolution2D(1, 5, 5, border_mode = 'same')(x)
        x = Activation('tanh')(x)

        return Model(input=input_x, output=x)

    def discriminator():
        """
        Create the discriminator network
        """
        #greyscale 28x28 images as input
        input_x = Input(shape=(28, 28, 1))
        x = input_x

        x = Conv2D(64, 5, 5, border_mode = 'same')(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)

        x = Conv2D(128, 5, 5)(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('tanh')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(input = input_x, output = x)
                            
