import pickle
import os
from datetime import datetime

from keras.models import model_from_json

import keras
from keras import backend as K

"""# Exceptions"""

class UserError(Exception):
    """Base class for user errors."""
    pass

"""# Save and load tensorflow models"""

def save_model(model, path_to_folder):

    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
  
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(path_to_folder, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(path_to_folder, "weights.h5"))

    print(f"Model saved to {path_to_folder}/ with:\n - model's name: {model.name} \n - input shape: {model.input.shape} \n - output shape: {model.output.shape}")
    

def load_model(path_to_folder, custom_objects):

    ts = os.path.getmtime(os.path.join(path_to_folder, "model.json"))
    last_modified = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    json_file = open(os.path.join(path_to_folder, "model.json"), 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json, custom_objects=custom_objects)
    # load weights into new model
    model.load_weights(os.path.join(path_to_folder, "weights.h5"))

    print(f"Model loaded from {path_to_folder}/ with:\n - model's name: {model.name} \n - input shape: {model.input.shape} \n - output shape: {model.output.shape}\n - last modified: {last_modified}")

    return model

"""# Save and load objects as pickle files"""

def save_object(obj, path_to_file):
    with open(path_to_file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(path_to_file):
    with open(path_to_file, 'rb') as f:
        return pickle.load(f)

"""# Plots"""

# Helper function used for visualization in the following examples
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : Dict[str, Axes]
        Mapping between the title / label and the Axes.

    fontsize : int, optional
        How big the label should be
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)

"""# Custom variational layer"""

# construct a custom layer to calculate the loss
class CustomVariationalLayer(keras.layers.Layer):

    def __init__(self, beta, **kwargs):
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.beta = beta

    def vae_loss(self, x, z_decoded, z_mu, z_log_sigma):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss (xent)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        # Reconstruction loss (mse)
        mse_loss = keras.metrics.mean_squared_error(x, z_decoded)
        # KL divergence
        kl_loss = -self.beta * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        # KL divergence (corrected with the square of sigma)
        correct_kl_loss = -self.beta * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma)**2, axis=-1)
        # KL divergence (reversed)
        rev_kl_loss = -self.beta * K.mean(1 - z_log_sigma - K.square(z_mu)/K.exp(z_log_sigma) - 1/K.exp(z_log_sigma), axis=-1)
        return K.mean(mse_loss + kl_loss)

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        z_mu = inputs[2]
        z_log_sigma = inputs[3]
        loss = self.vae_loss(x, z_decoded, z_mu, z_log_sigma)
        self.add_loss(loss, inputs=inputs)
        return x

    # makes model saving more reliable
    def get_config(self):
        config = {'beta': self.beta}
        base_config = super(CustomVariationalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # There's actually no need to define `from_config` here, since returning
    # `cls(**config)` is the default behavior.
    @classmethod
    def from_config(cls, config):
        return cls(**config)

