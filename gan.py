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

def main():
    #load MNIST dataset
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    X_train, X_test = preprocess_data(X_train), preprocess_data(X_test)

    print(X_train[0])


if __name__ == "__main__":
    main()
