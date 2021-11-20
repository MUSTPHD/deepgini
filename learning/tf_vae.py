import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
import tqdm

# bug fixed for: tensorflow.python.framework.errors_impl.InternalError: Blas GEMM launch failed
physical_devices = tf.config.list_physical_devices('GPU') 
print('-----', len(physical_devices))
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

train_size = 60000
batch_size = 32
test_size = 10000




def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


def load_data(train=True, test=False):
    (train_images, y_train), (test_images, y_test) = tf.keras.datasets.mnist.load_data()
    if train:
        train_images = preprocess_images(train_images)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size)
    if test:
        test_images = preprocess_images(test_images)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size)
    return {
        "train": train_dataset if train else None, 
        "y_train": y_train if train else None,
        "test": test_dataset if test else None,
        "y_test": y_test if test else None
    }


print('---', 'load data success')

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim=2):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker
        ]

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(loss)
        return {
            "loss": loss  # model.total_loss_tracker.result()
        }


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


# def compute_loss(model, x):
#     mean, logvar = model.encode(x)
#     z = model.reparameterize(mean, logvar)
#     x_logit = model.decode(z)
#     cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
#         logits=x_logit, labels=x)
#     logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#     logpz = log_normal_pdf(z, 0., 0.)
#     logqz_x = log_normal_pdf(z, mean, logvar)
#     return -tf.reduce_mean(logpx_z + logpz - logqz_x)


# @tf.function
# def train_step(model, x, optimizer):
#     """Executes one training step and returns the loss.

#     This function computes the loss and gradients, and uses the latter to
#     update the model's parameters.
#     """
#     with tf.GradientTape() as tape:
#         loss = compute_loss(model, x)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     model.total_loss_tracker.update_state(loss)
# #   tf.print('--loss', loss)
#     return {
#         "loss": loss  # model.total_loss_tracker.result()
#     }



if __name__ == '__main__':
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model = CVAE()
    weight_path = './vae_model/vae'
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
    train_dataset = load_data()["train"]
    iterator = iter(train_dataset)
    # ckpt = tf.train.Checkpoint(
    #     step=tf.Variable(1), optimizer=optimizer, net=model, iterator=iterator
    # )
    # manager = tf.train.CheckpointManager(ckpt, "./tf_ckpts", max_to_keep=3)

    checkpoint_filepath = './vae_model/vae'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True,
        verbose=1)

    epochs = 10


    model.compile(optimizer=optimizer, run_eagerly=True)
    model.fit(train_dataset, epochs=epochs, batch_size=batch_size, callbacks=[model_checkpoint_callback])


    # for epoch in range(1, epochs + 1):
    #     start_time = time.time()
    #     for train_x in tqdm.tqdm(train_dataset):
    #         ret = train_step(model, train_x, optimizer)
    #     # train_x = next(iterator)
    #     # ret = train_step(model, train_x, optimizer)
    #     end_time = time.time()
    #     loss = ret['loss']
    #     # ckpt.step.assign_add(1)
    #     # if int(ckpt.step) % 10 == 0:
    #     #     save_path = manager.save()
    #     #     print("Saved checkpoint for step {}: {}".format(
    #     #         int(ckpt.step), save_path))
    #     #     print("loss {:1.2f}".format(loss.numpy()))
    #     print('--loss', ret['loss'])
    #   print('--', end_time - start_time)
