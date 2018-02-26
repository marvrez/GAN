import keras
import numpy as np
from keras.datasets import mnist
from gan_train import *
import argparse

def main():
    parser = argparse.ArgumentParser()

    # Setup arguments for parser
    parser.add_argument('--batch_size', type=int, default = 16,
        help='Batch size for the networks')
    parser.add_argument('--epochs', type=int, default = 1000,
        help='Epochs for the networks')
    parser.add_argument('--output_dir', type=str, default = 'GAN_OUT',
        help='Output directory to save the image results.')
    parser.add_argument('--visualize', type=str, default = 0,
        help='Set to 1 if you want to visualize the results while training.')
    parser.add_argument('--input_dim', type=int, default = 10,
        help='Input dimension for the generator.')
    parser.add_argument('--n_train', type=int, default=32,
        help='The number of training data.')

    args = parser.parse_args()

    #load MNIST dataset 
    (x_train, y_train), (_, _) = mnist.load_data()

    train_gan(args, x_train)

if __name__ == "__main__":
    main()
