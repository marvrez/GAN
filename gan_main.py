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

def main():
    #load MNIST dataset
    (x_train, y_train), (_, _) = mnist.load_data()
    
    x_train = preprocess_data(x_train)

if __name__ == "__main__":
    main()
