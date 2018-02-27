import keras
import numpy as np
from keras.layers import *
from keras.models import Model, Sequential
from keras import models, layers
from keras.optimizers import SGD
import tensorflow as tf

class GAN(models.Sequential):
    def __init__(self, input_dim = 64):
        super().__init__()
        self.input_dim = input_dim

        #assemble both discriminator and generator
        self.generator = self.generator()
        self.discriminator = self.discriminator()
        
        # Print summary of networks
        self.generator.summary()
        self.discriminator.summary()

        self.add(self.generator)
        self.discriminator.trainable = False
        self.add(self.discriminator)

        self.compile_all()

    def compile_all(self):
        # Compiling stage
        d_optimizer = SGD(lr = 0.0005, momentum = 0.9, nesterov = True)
        g_optimizer = SGD(lr = 0.0005, momentum = 0.9, nesterov = True)

        self.discriminator.trainable = False
        self.compile(loss = 'binary_crossentropy', optimizer = g_optimizer)
        self.discriminator.trainable = True
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = d_optimizer)

    def generator(self):
        """
        Create the generator network
        """
        #Input tensor will be batches of z_dim-dimensional vectors
        input_dim = self.input_dim

        model = models.Sequential()
        model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        model.add(layers.Dense(128 * 7 * 7, activation='tanh'))
        model.add(layers.BatchNormalization())
        model.add(layers.Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))

        model.add(layers.UpSampling2D(size=(2, 2))) #14x14
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))

        model.add(layers.UpSampling2D(size=(2, 2))) #28x28
        model.add(layers.Conv2D(1, (5, 5), padding='same', activation='tanh'))
        return model


    def discriminator(self):
        """
        Create the discriminator network
        """
        model = models.Sequential()
        #greyscale 28x28 images as input
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh',
                                input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (5, 5), activation='tanh'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def get_z(self, shape_len):
        # Get sample minibatch of m noise samples from noise prior p(z)
        input_dim = self.input_dim
        return np.random.uniform(-1.0, 1.0, size = (shape_len, input_dim))

    def train_networks(self, real_x):
        # Run one iteraton of training for both the discriminator and generator
        # Returns the training loss for the discriminator and generator
        shape_len = real_x.shape[0]

        # First trial for training discriminator
        z = self.get_z(shape_len) # noise vector
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
