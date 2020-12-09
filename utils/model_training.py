import os
import time as T
import numpy as np
import keras
from keras import layers
from keras.models import Model
from keras import backend as K

import tensorflow as tf

import utils.data_preprocessing as data_preprocessing
import utils.utils as utils

"""# Load and process data

## Load and concatenate data
"""

# def load_data(features_list):
#
#     data_dict = {}
#
#     if 'stft' in features_list:
#         X_stft = np.load(os.path.join(root_path, 'datasets', 'X_stft.npy'))
#         X_stft = X_stft.reshape(X_stft.shape[0], -1)
#         data_dict['stft'] = X_stft
#         del X_stft
#
#     if 'wav' in features_list:
#         X_wav = np.load(os.path.join(root_path, 'datasets', 'X_wav.npy'))
#         data_dict['wav'] = X_wav
#         del X_wav
#
#     if 'mel' in features_list:
#         X_mel = np.load(os.path.join(root_path, 'datasets', 'X_mel.npy'))
#         X_mel = X_mel.reshape(X_mel.shape[0], -1)
#         data_dict['mel'] = X_mel
#         del X_mel
#
#     if 'mfcc' in features_list:
#         X_mfcc = np.load(os.path.join(root_path, 'datasets', 'X_mfcc.npy'))
#         X_mfcc = X_mfcc.reshape(X_mfcc.shape[0], -1)
#         data_dict['mfcc'] = X_mfcc
#         del X_mfcc
#
#     dimensions = dict([ (name, feature.shape[1]) for (name, feature) in data_dict.items()])
#     main_feature = features_list[0]
#     main_value = data_dict.pop(main_feature)
#     # Concatenate features with the main feature at the beginning
#     X = np.concatenate( [main_value] + list(data_dict.values()), axis=1)
#
#     y_digits = np.load(os.path.join(root_path, 'datasets', 'y_digits.npy'))
#     y_speakers = np.load(os.path.join(root_path, 'datasets', 'y_speakers.npy'))
#     train_index = np.load(os.path.join(root_path, 'datasets', 'train_index.npy'))
#
#     return X, dimensions, y_digits, y_speakers, train_index

# """## Load audio preprocessing parameters"""
#
# def load_librosa_params(preprocessing_parameters, features_list):
#
#     sr = preprocessing_parameters['SR']
#
#     if 'mel' in features_list or 'mfcc' in features_list:
#         librosa_params = preprocessing_parameters[features_list[0]]
#
#     if features_list == ['mel']:
#         input_shape = (librosa_params['n_mels'], preprocessing_parameters['SR']*preprocessing_parameters['STOP_TIME']//librosa_params['hop_length']+1)
#
#     elif features_list == ['mfcc']:
#         input_shape = (librosa_params['n_mfcc'], preprocessing_parameters['SR']*preprocessing_parameters['STOP_TIME']//librosa_params['hop_length']+1)
#
#     elif features_list == ['wav']:
#         input_shape = (1, preprocessing_parameters['SR']*preprocessing_parameters['STOP_TIME'])
#         librosa_params = None
#
#     else:
#         raise utils.UserError(f"Parameter features_list={features_list} not allowed. Expected one of {[['mel'], ['mfcc']]}.")
#
#     return librosa_params, input_shape, sr

"""## Split train and test sets"""

# def split_train_test(X, y_digits, y_speakers, train_index=None, n_test=None):
#
#     if train_index is None:
#         assert n_test != None
#         train_index = np.random.choice(np.arange(X.shape[0]), X.shape[0]-n_test, replace=False)
#
#     test_index = np.array([i for i in np.arange(X.shape[0]) if not i in train_index])
#
#     X_train, X_test = X[train_index], X[test_index]
#     y_digits_train, y_digits_test = y_digits[train_index], y_digits[test_index]
#     y_speakers_train, y_speakers_test = y_speakers[train_index], y_speakers[test_index]
#
#     return X_train, X_test, y_digits_train, y_digits_test, y_speakers_train, y_speakers_test, train_index

"""## Normalize data"""

def normalize(X, normalize_type, input_shape=None):

    X_norm = X.copy()
    norm_params = None

    if normalize_type is not None:

        # Min-max rescaling
        if normalize_type == 'min_max': 
            min = np.min(X_norm, axis=1).reshape(-1, 1)
            max = np.max(X_norm, axis=1).reshape(-1, 1)
            X_norm = (X_norm - min) \
                / (max - min)
            norm_params = {'min': min, 'max': max}

        # Avg_std rescaling
        elif normalize_type == 'avg_std':
            avg = np.mean(X_norm, axis=1).reshape(-1, 1)
            std = np.std(X_norm, axis=1).reshape(-1, 1)
            X_norm = (X_norm - avg) \
                / std
            norm_params = {'avg': avg, 'std': std}
            
        # Avg_std rescaling (f-band-wise split)
        elif normalize_type == 'avg_std_fband_split':
            assert input_shape is not None
            X_norm = X_norm.reshape(X_norm.shape[0], *input_shape)
            avg = np.mean(X_norm, axis=2)
            std = np.std(X_norm, axis=2)
            std[std==0] = 1
            avg = avg.reshape(*avg.shape, 1)
            std = std.reshape(*std.shape, 1)
            X_norm = (X_norm - avg) \
                / std
            X_norm = X_norm.reshape(X_norm.shape[0], -1)
            norm_params = {'avg': avg, 'std': std}

        # Avg_std rescaling (t-band-wise split)
        elif normalize_type == 'avg_std_tband_split':
            assert input_shape is not None
            X_norm = X_norm.reshape(X_norm.shape[0], *input_shape)
            avg = np.mean(X_norm, axis=1)
            std = np.std(X_norm, axis=1)
            std[std==0] = 1
            avg = avg.reshape(avg.shape[0], 1, avg.shape[1])
            std = std.reshape(std.shape[0], 1, std.shape[1])
            X_norm = (X_norm - avg) \
                / std
            X_norm = X_norm.reshape(X_norm.shape[0], -1)
            norm_params = {'avg': avg, 'std': std}

        else:
            raise utils.UserError(f'Parameter normalize_type={normalize_type} is not allowed.')

    return X_norm, norm_params

def inv_normalize(X_norm, normalize_type, norm_params, input_shape=None):

    X = X_norm.copy()

    if normalize_type is not None:

        # Min-max denormalization
        if normalize_type == 'min_max':
            assert list(norm_params.keys()) == ['min', 'max']
            min, max = norm_params['min'], norm_params['max']
            X = min + X*(max - min)

        # Avg_std denormalization
        elif normalize_type == 'avg_std':
            assert list(norm_params.keys()) == ['avg', 'std']
            avg, std = norm_params['avg'], norm_params['std']
            X = avg + X*std
            
        # Avg_std denormalization (f-band-wise split)
        elif normalize_type == 'avg_std_fband_split':
            assert input_shape is not None
            assert list(norm_params.keys()) == ['avg', 'std']
            avg, std = norm_params['avg'], norm_params['std']
            X = X.reshape(X.shape[0], *input_shape)
            X = avg + X*std
            X = X.reshape(X.shape[0], -1)

        # Avg_std denormalization (t-band-wise split)
        elif normalize_type == 'avg_std_tband_split':
            assert input_shape is not None
            assert list(norm_params.keys()) == ['avg', 'std']
            avg, std = norm_params['avg'], norm_params['std']
            X = X.reshape(X.shape[0], *input_shape)
            X = avg + X*std
            X = X.reshape(X.shape[0], -1)

        else:
            raise utils.UserError(f'Parameter normalize_type={normalize_type} is not allowed.')

    return X

"""# Autoencoder architecture

## Encoder network
"""

def get_classifier_model(input_shape, num_classes, classifier_type):
    
    model = keras.Sequential()

    if classifier_type == 'cnn_1':

        model.add(layers.Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(48, kernel_size=(2, 2), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(120, kernel_size=(2, 2), activation='relu'))
        model.add(layers.BatchNormalization())

        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())

        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    elif classifier_type == 'cnn_2':

        model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(layers.Activation('relu'))
        
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.Activation('relu'))
        
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.Activation('relu'))
        
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Conv2D(128, (3, 3), padding='same'))
        model.add(layers.Activation('relu'))
        
        # model.add(layers.Conv2D(128, (3, 3)))
        # model.add(layers.Activation('relu'))
        
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(512))
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        adam = tf.keras.optimizers.RMSprop(learning_rate = 0.001)
        model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    else:

        raise utils.UserError(f'Parameter classifier_type={classifier_type} not allowed.')

    return model

def flatten_model(model_nested):
    layers_flat = []
    for layer in model_nested.layers:
        try:
            layers_flat.extend(layer.layers)
        except AttributeError:
            layers_flat.append(layer)
    model_flat = keras.models.Sequential(layers_flat, name=model_nested.name)
    return model_flat
        
def build_encoder(encoder_type, encoder_output, input_shape, latent_dim):
    
    encoder = keras.Sequential(name=f'encoder_{encoder_type}')

    # FC encoder 1
    if encoder_type == 'fc_1':
        encoder.add(layers.Dense(4000, activation=tf.nn.leaky_relu, name='dense_leaky_relu_1'))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(.25))
        encoder.add(layers.Dense(1024, activation=tf.nn.leaky_relu, name='dense_leaky_relu_2'))
        encoder.add(layers.Dense(512 , activation=tf.nn.leaky_relu, name='dense_leaky_relu_3'))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(.25))
        encoder.add(layers.Dense(128 , activation=tf.nn.leaky_relu, name='dense_leaky_relu_4'))
        encoder.add(layers.Dense(32  , activation=tf.nn.leaky_relu, name='dense_leaky_relu_5'))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(.25))
        encoder.add(layers.Dense(latent_dim, activation=tf.nn.leaky_relu, name='dense_leaky_relu_0'))

    # FC encoder 2
    elif encoder_type == 'fc_2':
        encoder.add(layers.Dense(1024, activation=tf.nn.leaky_relu))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(.25))
        encoder.add(layers.Dense(512, activation=tf.nn.leaky_relu))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(.25))
        encoder.add(layers.Dense(256, activation=tf.nn.leaky_relu))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(.25))
        encoder.add(layers.Dense(latent_dim, activation=tf.nn.leaky_relu))

    # FC encoder 3
    elif encoder_type == 'fc_3':
        encoder.add(layers.Dense(2048, activation=tf.nn.leaky_relu))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(.25))
        encoder.add(layers.Dense(1024, activation=tf.nn.leaky_relu))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(.25))
        encoder.add(layers.Dense(512, activation=tf.nn.leaky_relu))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(.25))
        encoder.add(layers.Dense(256, activation=tf.nn.leaky_relu))
        encoder.add(layers.BatchNormalization())
        encoder.add(layers.Dropout(.25))
        encoder.add(layers.Dense(latent_dim, activation=tf.nn.leaky_relu))

    # CNN encoder 1
    elif encoder_type[:3] == 'cnn':

        encoder.add(layers.Reshape((*input_shape, 1)))
        loaded_classifier  = get_classifier_model((*input_shape, 1), latent_dim, encoder_type)

        for layer in loaded_classifier.layers[:-1]:
            encoder.add(layer)

    else:
        raise utils.UserError(f'Parameter encoder_type={encoder_type} is not allowed.')
    
    if encoder_output == 'sin':
        encoder.add(layers.Dense(2, activation='linear', name='dense_linear'))
        encoder.add(layers.Lambda(K.sin, name='lambda_sin'))

    elif encoder_output == 'tanh':
        encoder.add(layers.Dense(latent_dim, activation=tf.nn.tanh, name='dense_tanh'))

    elif encoder_output == 'leaky_relu':
        encoder.add(layers.Dense(latent_dim, activation=tf.nn.leaky_relu, name='dense_leakyReLU_00'))

    elif encoder_output == 'linear':
        encoder.add(layers.Dense(latent_dim, activation='linear', name='dense_linear'))
    
    else:
        raise utils.UserError(f'Parameter encoder_output={encoder_output} is not allowed.')

    encoder = flatten_model(encoder)

    return encoder

"""## Decoder network"""

def build_decoder(decoder_type, decoder_output, input_shape, latent_dim):

    # decoder_input = layers.Input(K.int_shape(z)[1:])
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')

    if decoder_type == 'fc_1':
        x = layers.Dense(32  , activation=tf.nn.leaky_relu, name='dense_leaky_relu_5')(decoder_input)
        x = layers.Dense(128 , activation=tf.nn.leaky_relu, name='dense_leaky_relu_4')(x)
        x = layers.Dense(512 , activation=tf.nn.leaky_relu, name='dense_leaky_relu_3')(x)
        x = layers.Dense(1024, activation=tf.nn.leaky_relu, name='dense_leaky_relu_2')(x)
        x = layers.Dense(4000, activation=tf.nn.leaky_relu, name='dense_leaky_relu_1')(x)
    
    elif decoder_type == 'fc_2':
        x = layers.Dense(32  , activation=tf.nn.leaky_relu, name='dense_leaky_relu_5')(decoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.5)(x)
        x = layers.Dense(128 , activation=tf.nn.leaky_relu, name='dense_leaky_relu_4')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.25)(x)
        x = layers.Dense(512 , activation=tf.nn.leaky_relu, name='dense_leaky_relu_3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.5)(x)
        x = layers.Dense(2048, activation=tf.nn.leaky_relu, name='dense_leaky_relu_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.25)(x)
        x = layers.Dense(8192, activation=tf.nn.leaky_relu, name='dense_leaky_relu_1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.5)(x)

    elif decoder_type == 'fc_3':
        x = layers.Dense(32  , activation=tf.nn.leaky_relu)(decoder_input)
        x = layers.Dense(128 , activation=tf.nn.leaky_relu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.25)(x)
        x = layers.Dense(512 , activation=tf.nn.leaky_relu)(x)
        x = layers.Dense(1024, activation=tf.nn.leaky_relu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.5)(x)
        x = layers.Dense(2048, activation=tf.nn.leaky_relu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.5)(x)
        x = layers.Dense(4000, activation=tf.nn.leaky_relu)(x)

    elif decoder_type == 'fc_4':
        x = layers.Dense(256, activation=tf.nn.leaky_relu)(decoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.25)(x)
        x = layers.Dense(512, activation=tf.nn.leaky_relu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.5)(x)
        x = layers.Dense(1024, activation=tf.nn.leaky_relu)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(.25)(x)
        x = layers.Dense(2048, activation=tf.nn.leaky_relu)(x)

    else:
        raise utils.UserError(f'Parameter decoder_type={decoder_type} is not allowed.')

    input_length = np.prod(input_shape)

    if decoder_output == 'leaky_relu':
        x = layers.Dense(input_length, activation=tf.nn.leaky_relu, name='dense_leaky_relu_0')(x)
    elif decoder_output == 'tanh':
        x = layers.Dense(input_length, activation=tf.nn.tanh, name='dense_tanh')(x)
    elif decoder_output == 'linear':
        x = layers.Dense(input_length, activation='linear', name='dense_linear')(x)
    elif decoder_output == 'sigmoid':
        x = layers.Dense(input_length, activation='sigmoid', name='dense_sigmoid')(x)
    else:
        raise utils.UserError(f'Parameter decoder_type={decoder_type} is not allowed.')

    # decoder model statement
    decoder = Model(decoder_input, x, name='decoder_{}'.format(decoder_type))

    return decoder

# def define_custom_loss(loss_type):

#     def custom_loss(yTrue, yPred):
#         x = K.flatten(yTrue)
#         z_decoded = K.flatten(yPred)
#         # Reconstruction loss (binary cross-entropy)
#         if loss_type == 'xent_only_main':
#             loss = keras.metrics.binary_crossentropy(x[:dim_main], z_decoded[:dim_main])
#         # Reconstruction loss (MSE loss)
#         elif loss_type == 'mse_only_main':
#             loss = keras.metrics.mean_squared_error(x[:dim_main], z_decoded[:dim_main])
#         # Reconstruction loss (MSE loss)
#         elif loss_type == 'mse_full':
#             loss = keras.metrics.mean_squared_error(x, z_decoded)
#         elif loss_type == 'mse_mfcc19':
#             assert dim_main == 4020
#             loss = keras.metrics.mean_squared_error(x[201:dim_main], z_decoded[201:dim_main])
#         # Reconstruction loss (MAE loss)
#         elif loss_type == 'mae':
#             loss = keras.metrics.mean_absolute_error(x, z_decoded)
#         return K.mean(loss)

#     print(f"Custom loss defined under variable 'custom_loss' with:\n - loss_type: {loss_type}")

#     return custom_loss

#LOSS_TYPE = 'mse_only_main'

    
# # construct a custom layer to calculate the loss
# class CustomLossLayer(keras.layers.Layer):

#     # def __init__(self, name, loss_type):
#     #     super(CustomVariationalLayer, self).__init__(name=name)
#     #     # self.name = name
#     #     self.loss_type = loss_type

#     def autoencoder_loss(self, x, z_decoded):
#         x = K.flatten(x)
#         z_decoded = K.flatten(z_decoded)
#         # Reconstruction loss (binary cross-entropy)
#         if loss_type == 'xent':
#             loss = keras.metrics.binary_crossentropy(x, z_decoded)
#         # Reconstruction loss (MSE loss)
#         elif loss_type == 'mse_only_main':
#             loss = keras.metrics.mean_squared_error(x[:8000], z_decoded[:8000])
#         # Reconstruction loss (MSE loss)
#         elif loss_type == 'mse_full':
#             loss = keras.metrics.mean_squared_error(x, z_decoded)
#             #loss = tf.keras.metrics.mean_squared_error(x, z_decoded)
#         # Reconstruction loss (MAE loss)
#         elif loss_type == 'mae':
#             loss = keras.metrics.mean_absolute_error(x, z_decoded)
#         # KL divergence
#         # kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
#         # Rescaling loss
#         #rescaling_loss = 5e-4 * K.mean(K.square(z), axis=-1)
#         #return K.mean(xent_loss + kl_loss)
#         return K.mean(loss)

#     # adds the custom loss to the class
#     def call(self, inputs):
#         x = inputs[0]
#         z_decoded = inputs[1]
#         loss = self.autoencoder_loss(x, z_decoded)
#         self.add_loss(loss, inputs=inputs)
#         return x

#     # makes model saving more reliable
#     def get_config(self):
#         config = super(CustomVariationalLayer, self).get_config()
#         # config.update({"units": self.units})
#         return config

#     # There's actually no need to define `from_config` here, since returning
#     # `cls(**config)` is the default behavior.
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

"""## Variational loss

"""

# # construct a custom layer to calculate the loss
# class CustomVariationalLayer(keras.layers.Layer):

#     def vae_loss(self, x, z_decoded, z_mu, z_log_sigma):
#         x = K.flatten(x)
#         z_decoded = K.flatten(z_decoded)
#         # Reconstruction loss
#         xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
#         # KL divergence
#         kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
#         # KL divergence (corrected with the square of sigma)
#         correct_kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma)**2, axis=-1)
#         # KL divergence (reversed)
#         rev_kl_loss = -5e-4 * K.mean(1 - z_log_sigma - K.square(z_mu)/K.exp(z_log_sigma) - 1/K.exp(z_log_sigma), axis=-1)
#         return K.mean(xent_loss + correct_kl_loss)

#     # adds the custom loss to the class
#     def call(self, inputs):
#         x = inputs[0]
#         z_decoded = inputs[1]
#         z_mu = inputs[2]
#         z_log_sigma = inputs[3]
#         loss = self.vae_loss(x, z_decoded, z_mu, z_log_sigma)
#         self.add_loss(loss, inputs=inputs)
#         return x

#     # makes model saving more reliable
#     def get_config(self):
#         config = super(CustomVariationalLayer, self).get_config()
#         # config.update({"units": self.units})
#         return config

#     # There's actually no need to define `from_config` here, since returning
#     # `cls(**config)` is the default behavior.
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

"""## Full autoencoder statement"""

def build_autoencoder(input_shape, encoder_type, encoder_output, latent_dim, decoder_type, decoder_output, use_custom_loss, custom_loss, learning_rate, decay, is_variational, beta):

    input_length = np.prod(input_shape)

    if is_variational:

        encoder = build_encoder(encoder_type, encoder_output, input_shape, 2*latent_dim)
        decoder = build_decoder(decoder_type, decoder_output, input_shape, latent_dim)

        x = layers.Input(shape=(input_length,), name='encoder_input')
        
        z_parameters = encoder(x)
        z_mu = z_parameters[:, :latent_dim]
        z_log_sigma = z_parameters[:, latent_dim:]

        # sample vector from the latent distribution
        def sampling(args):
            z_mu, z_log_sigma = args
            epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                                      mean=0., stddev=1.)
            return z_mu + K.exp(z_log_sigma) * epsilon
        
        z = layers.Lambda(sampling, name='sampling_layer')([z_mu, z_log_sigma])

        x_decoded = decoder(z)

        # apply the custom loss to the input images and the decoded latent distribution sample
        y = utils.CustomVariationalLayer(beta=beta)([x, x_decoded, z_mu, z_log_sigma])

        # Full autoencoder statement
        autoencoder = Model(x, y, name="full_auto_encoder")

        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate, decay=decay)
        autoencoder.compile(optimizer=optimizer, loss=None)

    else:

        encoder = build_encoder(encoder_type, encoder_output, input_shape, latent_dim)
        decoder = build_decoder(decoder_type, decoder_output, input_shape, latent_dim)

        x = layers.Input(shape=(input_length,), name='encoder_input')
        x_encoded = encoder(x)
        x_decoded = decoder(x_encoded)

        # Full autoencoder statement
        autoencoder = Model(x, x_decoded, name="full_auto_encoder")

        # Loss function
        if use_custom_loss:
            loss = custom_loss
        else:
            loss = ["mse"]

        optimizer   = keras.optimizers.RMSprop(learning_rate=learning_rate, decay=decay)
        autoencoder.compile(optimizer=optimizer, loss=loss)
    
    return autoencoder, encoder, decoder

# Diagrams of models
# if __name__ == "__main__":
  
    # autoencoder_scheme_path = root_path + 'models/autoencoder_scheme.png'
    # plot_model(autoencoder, to_file=autoencoder_scheme_path, show_shapes=True, show_layer_names=True)
    
    # encoder_scheme_path = root_path + 'models/encoder_{}_scheme.png'.format(ENCODER_TYPE)
    # plot_model(encoder, to_file=encoder_scheme_path, show_shapes=True, show_layer_names=True)
    
    # decoder_scheme_path = root_path + 'models/decoder_scheme.png'
    # plot_model(decoder, to_file=decoder_scheme_path, show_shapes=True, show_layer_names=True)

"""# Training

## Calbacks
"""

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = T.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(T.time() - self.epoch_time_start)

"""## Training"""

def train_autoencoder(autoencoder, n_epochs, batch_size, X_train, X_test):

    time_callback = TimeHistory()
   
    autoencoder_train = autoencoder.fit(
            x=X_train,  
            y=X_train, #IMPORTANT: For an autoencoder, x=y
            shuffle=True,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=(X_test, X_test),
            callbacks=[
                      # time_callback,
                      #early_stopping_callback, #Gives error: 'EarlyStopping' object has no attribute 'on_test_begin'
                      #tensorboard_callback
            ]
    )

    return autoencoder_train



