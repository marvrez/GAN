import keras
import numpy as np
from keras.layers import *
from keras.models import Model, Sequential
from keras import models, layers
from keras.optimizers import SGD, Adam
import tensorflow as tf

class GAN():
    def __init__(self, input_dim = 100):
        self.input_dim = input_dim
        self.optimizer = Adam(lr=0.0002, beta_1 =0.5)

        #assemble both discriminator and generator
        self.generator = self.generator()
        self.discriminator = self.discriminator()
        
        # Print summary of networks
        self.generator.summary()
        self.discriminator.summary()

        # Create the combined network
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.input_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)

        self.gan = Model(inputs = gan_input, outputs = gan_output)
        self.gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    def generator(self):
        """
        Create the generator network
        """
        #Input tensor will be batches of z_dim-dimensional vectors
        input_dim = self.input_dim
        model = models.Sequential()

        model.add(layers.Dense(128*7*7, input_dim=input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Reshape((128, 7, 7)))

        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(64, kernel_size=(5, 5), padding='same'))

        model.add(layers.LeakyReLU(0.2))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))

        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        return model


    def discriminator(self):
        """
        Create the discriminator network
        """
        #greyscale 28x28 images as input
        model = models.Sequential()
        model.add(layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', 
                    input_shape=(1, 28, 28), 
                    kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        return model

    def get_z(self, shape_len):
        # Get sample minibatch of m noise samples from noise prior p(z)
        input_dim = self.input_dim
        return np.random.normal(0.0, 1.0, size = [shape_len, input_dim])

    def train_networks(self, real_x):
        # Run one iteraton of training for both the discriminator and generator
        # Returns the training loss for the discriminator and generator
        shape_len = real_x.shape[0]

        # First trial for training discriminator
        z = self.get_z(shape_len) # noise vector
        fake_x = self.generator.predict(z)

        x_batch = np.concatenate((real_x, fake_x))
        y_batch = [0.9] * shape_len + [0] * shape_len # real_x has high target probability
        self.discriminator.trainable = True
        d_loss = self.discriminator.train_on_batch(x_batch, y_batch)

        # Second trial for training generator
        z = self.get_z(shape_len)
        self.discriminator.trainable = False
        g_loss = self.gan.train_on_batch(z, [1] * shape_len)

        return d_loss, g_loss
