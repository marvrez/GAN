import keras
import numpy as np
import matplotlib.pyplot as plot
from tqdm import tqdm
from keras.datasets import mnist
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import SGD
from gan_model import *
from gan_train import *
import argparse

def main():
    parser = argparse.ArgumentParser()

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
