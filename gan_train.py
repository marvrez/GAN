import math
import os
import numpy as np
import matplotlib.pyplot as plot
from tqdm import tqdm
from gan_model import GAN
from PIL import Image

# Center the data around 0, and change values to be in range [-1,1]
def preprocess_data(data, n_train):
    data = (data.astype(np.float32) - 127.5) / 127.5
    return data[:n_train]

# Get x from the image distribution
def get_x(x_train, BATCH_SIZE):
    return x_train[np.random.randint(0, x_train.shape[0], size = BATCH_SIZE)]

def combine_images(images):
    samples = images.shape[0]
    width   = int(math.sqrt(samples))
    height  = int(math.ceil(float(samples) / width))
    shape   = images.shape[2:]
    image   = np.zeros((height * shape[0], width * shape[1]), dtype=images.dtype)

    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0],
              j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image

def save_images(generated_images, output_dir, epoch, index):
    image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        output_dir + '/' + str(epoch) + "_" + str(index) + ".png")

def train_gan(args, training_images):
    BATCH_SIZE  = args.batch_size
    epochs      = args.epochs
    output_dir  = args.output_dir
    visualize   = args.visualize
    n_train     = args.n_train
    input_dim   = args.input_dim

    training_images = preprocess_data(training_images, n_train)
    training_images = training_images[:, np.newaxis, :, :]

    loss_fig = plot.figure()
    loss_ax = loss_fig.add_subplot(111)

    if visualize:
        # Show generated images
        output_fig = plot.figure()
        output_ax = output_fig.add_subplot(111)
        output_ax.axis('off')
        output_fig.show()

        # Show loss graph
        loss_fig.show()

    os.makedirs(output_dir, exist_ok=True)
    print('Output directory for images is', output_dir)

    # Stores average loss per epoch
    d_loss_epoch, g_loss_epoch = [], []

    gan = GAN(input_dim)

    BATCHES_PER_EPOCH = training_images.shape[0] // BATCH_SIZE
    print("Batches: %d, Batches per epoch: %d" % (BATCH_SIZE, BATCHES_PER_EPOCH))
    for epoch in tqdm(range(epochs), desc = "Training"):
        for idx in tqdm(range(BATCHES_PER_EPOCH), desc = 'Epoch {}'.format(epoch), leave = False):
            real_x = get_x(training_images, BATCH_SIZE) #get the image batch
            d_loss, g_loss = gan.train_networks(real_x)

        d_loss_epoch.append(d_loss)
        g_loss_epoch.append(g_loss)

        if epoch % 1 == 0 or epoch == epochs - 1:
            z = gan.get_z(10**2)
            generated_images = gan.generator.predict(z)
            save_images(generated_images, output_dir, epoch, 0)

            loss_ax.clear()
            loss_ax.plot(np.arange(len(d_loss_epoch)), d_loss_epoch, label='d_loss')
            loss_ax.plot(np.arange(len(g_loss_epoch)), g_loss_epoch, label='g_loss')
            loss_ax.legend()
            loss_fig.canvas.draw()
            loss_fig.savefig(output_dir + '/' + 'loss_graph.png')

            if visualize:
                output_ax.imshow(combine_images(generated_images), cmap='gray')
                output_fig.canvas.draw()


    #save weights when done training
    gan.generator.save_weights(output_dir + '/' + 'generator_weights', True)
    gan.discriminator.save_weights(output_dir + '/' + 'discriminator_weights', True)

    loss_fig.savefig(output_dir + '/' + 'loss_graph.png')

    np.savetxt(output_dir + '/' + 'd_loss', d_loss_epoch)
    np.savetxt(output_dir + '/' + 'g_loss', g_loss_epoch)
