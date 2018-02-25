import numpy as np
import matplotlib.pyplot as plot
from tqdm import tqdm
from gan_model import GAN
import os

# Get x from the image distribution
def get_x(x_train, index, BATCH_SIZE):
    return x_train[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]

def combine_images(images):
    num = images.shape[0]
    width, height = int(math.sqrt(num)), int(math.ceil(float(num) / width))
    shape = images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                      dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0],
              j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image

def create_image_grid(images):
    n = int(round(np.sqrt(len(images))))
    images = images.reshape(images.shape[:-1])
    i_h, i_w  = images.shape[1], images.shape[2]
    image_grid = np.zeros((n * i_h, n * i_w))
    for r in range(n):
        for c in range(n):
            image_grid[r*i_h:(r+1)*i_h, c*i_w:(c+1)*i_w] = images[r*n+c, ...]
    return image_grid

def save_images(generated_images, output_dir, epoch, index):
    image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        output_dir + '/' + str(epoch) + "_" + str(index) + ".png")

def train_gan(args, training_images):
    BATCH_SIZE = args.batch_size
    epochs = args.epochs
    output_dir = args.output_dir
    visualize = args.visualize
    input_dim = args.input_dim

    if(visualize):
        output_fig = plt.figure()
        output_ax = output_fig.add_subplot(111)
        output_ax.axis('off')
        output_fig.show()

        loss_fig = plt.figure()
        loss_ax = loss_fig.add_subplot(111)
        loss_fig.show()

    os.makedirs(output_dir, exist_ok=True)
    print('Output directory for images is', output_dir)

    # Stores average loss per epoch
    d_loss_epoch, g_loss_epoch = [], []

    gan = GAN(input_dim)

    BATCHES_PER_EPOCH = training_images.shape[0] // BATCH_SIZE
    print("Batches: %d, Batches per epoch: %d" % (BATCH_SIZE, BATCHES_PER_EPOCH))
    for epoch in tqdm(range(epochs), desc = "Training"):
        d_losses, g_losses= [], []
        for idx in tqdm(range(BATCHES_PER_EPOCH), desc = 'Epoch {}'.format(epoch), leave = False):
            real_x = get_x(training_images, idx, BATCH_SIZE)

            d_loss, g_loss = gan.train_networks(real_x)

            d_losses.append(d_loss)
            g_losses.append(g_loss)

        d_loss_epoch.append(sum(d_losses) / len(d_losses))
        g_loss_epoch.append(sum(g_losses) / len(g_losses))

    #save weights when done training
    gan.generator.save_weights(output_fold + '/' + 'generator_weights', True)
    gan.discriminator.save_weights(output_fold + '/' + 'discriminator_weights', True)

    np.savetxt(output_fold + '/' + 'd_loss', d_loss_ll)
    np.savetxt(output_fold + '/' + 'g_loss', g_loss_ll)
