import keras
import numpy as np
from keras.layers import *
from keras.models import Model, Sequential
from keras import models, layers
from keras.optimizers import SGD

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

    def discriminator(self):
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

    def get_z(self, shape_len):
        # Get sample minibatch of m noise samples from noise prior p(z)
        input_dim = self.input_dim
        return np.random.uniform(-1, 1, size = (shape_len, input_dim))

    def train_networks(self, real_x):
        # Run one iteraton of training for both the discriminator and generator
        # Returns the training loss for the discriminator and generator
        shape_len = real_x.shape[2]

        # First trial for training discriminator
        z = self.get_z(shape_len)
        fake_x = self.generator.predict(z)

        x_batch = np.concatenate((real_x, fake_x))
        y_batch = [1] * shape_len + [0] * shape_len # real_x has target probability of 1
        d_loss = self.discriminator.train_on_batch(x_batch, y_batch)

        # Second trial for training generator
        z = self.get_z(shape_len)
        self.discriminator.trainable = False
        g_loss = self.train_on_batch(z, [1] * shape_len)
        self.discriminator.trainable = True

        return d_loss, g_loss
