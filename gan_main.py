import keras
import numpy as np
from keras.datasets import mnist
from gan_train import *
import argparse

# Center the data around 0, and change values to be in range [-1,1]
def preprocess_data(data):
    data = data[..., np.newaxis]
    data = (data.astype(np.float32) - 127.5) / 127.5
    return data

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
    parser.add_argument('--input_dim', type=int, default = 100,
        help='Input dimension for the generator.')

    args = parser.parse_args()

    #load MNIST dataset and preprocess data
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = preprocess_data(x_train)

    train(args, data)

if __name__ == "__main__":
    main()
