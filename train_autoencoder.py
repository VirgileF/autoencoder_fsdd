import utils.utils as utils
import json
import os
import shutil
import numpy as np
import math as M

import keras
from keras import backend as K

import utils.model_training as model_training

import matplotlib.pyplot as plt

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        raise utils.UserError(f"Illegal command. Please type \'>> python <python_script.py> <params_file.json>\'.")

    # Load parameters from json file
    with open(sys.argv[1]) as f:
        TRAINING_PARAMS = json.load(f)

    # Retrieve paths from parameters
    data_dir = TRAINING_PARAMS['data_dir']
    save_model_to = TRAINING_PARAMS['save_model_to']

    # Check source directory.
    if not os.path.exists(data_dir):
        raise utils.UserError(
            f"Parameter data_dir={data_dir} not allowed. The path does not exists.")

    # Check target directory and confirm overwriting if exists.
    if os.path.exists(save_model_to):
        confirm = input(f"Folder \'{save_model_to}\' already exists. Do you want to overwrite it? (y/n): ")
        if confirm == 'y':
            shutil.rmtree(save_model_to)
        else:
            raise utils.UserError('User interrupted execution. Answer \'y\' to run script.')
    os.makedirs(save_model_to)

    # Load metadata
    with open(os.path.join(data_dir, 'metadata.json')) as f:
        metadata = json.load(f)

    # Load data
    X = np.load(os.path.join(data_dir, 'X_processed.npy'))
    source_data_dir = metadata['preprocessing_params']['source_dir']
    y_digits = np.load(os.path.join(source_data_dir, 'y_digits.npy'))
    y_speakers = np.load(os.path.join(source_data_dir, 'y_speakers.npy'))

    # Split train and test
    n_train = M.floor(TRAINING_PARAMS['training_ratio']*X.shape[0])
    train_index = np.random.choice(np.arange(X.shape[0]), n_train, replace=False)
    test_index = np.array([i for i in np.arange(X.shape[0]) if not i in train_index])
    X_train, X_test = X[train_index], X[test_index]
    y_digits_train, y_digits_test = y_digits[train_index], y_digits[test_index]
    y_speakers_train, y_speakers_test = y_speakers[train_index], y_speakers[test_index]

    # Normalize data
    input_shape = metadata['preprocessing_params']['input_shape']
    X_train, _ = model_training.normalize(X_train, TRAINING_PARAMS['normalize_type'], input_shape)
    X_test, _ = model_training.normalize(X_test, TRAINING_PARAMS['normalize_type'], input_shape)

    # Define loss
    loss_type = TRAINING_PARAMS['loss_type']

    def custom_loss(yTrue, yPred):
        x = K.flatten(yTrue)
        z_decoded = K.flatten(yPred)
        # Reconstruction loss (MSE loss)
        if loss_type == 'mse':
            loss = keras.metrics.mean_squared_error(x, z_decoded)
        # Reconstruction loss (MAE loss)
        elif loss_type == 'mae':
            loss = keras.metrics.mean_absolute_error(x, z_decoded)
        else:
            raise utils.UserError(f'Parameter loss_type={loss_type} is not allowed.')
        return K.mean(loss)

    # Build model
    print('===========================')
    print('... Building models ...')
    autoencoder, encoder, decoder = model_training.build_autoencoder(
        input_shape,
        TRAINING_PARAMS['encoder_type'],
        TRAINING_PARAMS['encoder_output'],
        TRAINING_PARAMS['latent_dim'],
        TRAINING_PARAMS['decoder_type'],
        TRAINING_PARAMS['decoder_output'],
        TRAINING_PARAMS['use_custom_loss'],
        custom_loss,
        TRAINING_PARAMS['learning_rate'],
        TRAINING_PARAMS['decay'],
        TRAINING_PARAMS['is_variational'],
        TRAINING_PARAMS['beta']
    )
    count_trainable = np.sum(
        [autoencoder.trainable_weights[i].numpy().size for i in range(len(autoencoder.trainable_weights))])
    count_non_trainable = np.sum(
        [autoencoder.non_trainable_weights[i].numpy().size for i in range(len(autoencoder.non_trainable_weights))])
    print(
        f'Autoencoder built with: \n - input shape: {autoencoder.input.shape}\n - output shape: {autoencoder.output.shape}\n - bottleneck shape: {encoder.output.shape}\n - complexity: \n   - trainable: {count_trainable} \n   - non-trainable: {count_non_trainable}')

    # Train model
    print('===========================')
    print('... Training models ...')
    autoencoder_train = model_training.train_autoencoder(
        autoencoder,
        TRAINING_PARAMS['n_epochs'],
        TRAINING_PARAMS['batch_size'],
        X_train,
        X_test)

    # Save model
    print('===========================')
    print('... Saving models ...')
    utils.save_model(autoencoder, os.path.join(TRAINING_PARAMS['save_model_to'], "autoencoder/"))
    utils.save_model(encoder, os.path.join(TRAINING_PARAMS['save_model_to'], "encoder/"))
    utils.save_model(decoder, os.path.join(TRAINING_PARAMS['save_model_to'], "decoder/"))
    print('Models saved successfully.')

    # Add training params to metadata
    metadata['training_params'] = TRAINING_PARAMS

    # Add training history to metadata
    metadata['training_history'] = autoencoder_train.history

    # Save metadata
    with open(os.path.join(TRAINING_PARAMS['save_model_to'], 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp, sort_keys=True, indent=4)

    # Save train_index
    np.save(os.path.join(TRAINING_PARAMS['save_model_to'], 'train_index.npy'), train_index)

    # Loss analysis
    print('\n==============================================')
    print('=============== LOSS ANALYSIS ================')
    print('==============================================')

    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs_list = np.arange(0, len(loss), 1)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, loss, 'bo', label='Training loss')
    plt.plot(epochs_list, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss ({})'.format(TRAINING_PARAMS['loss_type']))
    plt.title('Training and validation loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, np.log10(loss), 'bo', label='Training loss (log scale)')
    plt.plot(epochs_list, np.log10(val_loss), 'b', label='Validation loss(log scale)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss ({})'.format(TRAINING_PARAMS['loss_type']))
    plt.title('Training and validation loss (log scale)')
    plt.legend()

    plt.savefig(os.path.join(TRAINING_PARAMS['save_model_to'], 'loss_plot.png'))
    plt.show()