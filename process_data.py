import json
import os
import shutil

import numpy as np

import utils.utils as utils
import utils.data_preprocessing as data_preprocessing

if __name__ == '__main__':

    import sys

    if len(sys.argv) != 2:
        raise utils.UserError(f"Illegal command. Please type \'>> python <python_script.py> <params_file.json>\'.")

    # Load parameters from json file
    with open(sys.argv[1]) as f:
        PREPROCESSING_PARAMS = json.load(f)

    # Retrieve paths from parameters
    source_dir = PREPROCESSING_PARAMS['source_dir']
    target_dir = PREPROCESSING_PARAMS['target_dir']

    # Check source directory.
    if not os.path.exists(source_dir):
        raise utils.UserError(
            f"Parameter source_dir={source_dir} not allowed. The path does not exists.")

    # Check target directory and confirm overwriting if exists.
    if os.path.exists(target_dir):
        confirm = input(f"Folder \'{target_dir}\' already exists. Do you want to overwrite it? (y/n): ")
        if confirm == 'y':
            shutil.rmtree(target_dir)
        else:
            raise utils.UserError('User interrupted execution. Answer \'y\' to run script.')
    os.makedirs(target_dir)

    # Load metadata
    with open(os.path.join(source_dir, 'metadata.json')) as f:
        metadata = json.load(f)

    sample_rate = metadata['loading_params']['sample_rate']
    stop_time = metadata['loading_params']['stop_time']

    feature = PREPROCESSING_PARAMS['feature']

    if feature != 'wav':
        librosa_params = PREPROCESSING_PARAMS[''.join([feature, '_params'])]

    # Load waves
    X_wav = np.load(os.path.join(source_dir, 'X_wav.npy'))

    if feature == 'wav':

        X_processed = X_wav.copy()

    elif feature == 'stft':

        # Process waves to STFT
        X_processed = data_preprocessing.wav2stft(
            X_wav,
            sample_rate,
            stop_time,
            librosa_params,
        )

    elif feature == 'mel':

        # Process waves to mel-spectrograms
        X_processed = data_preprocessing.wav2mel(
            X_wav,
            sample_rate,
            stop_time,
            librosa_params,
        )

    elif feature == 'mfcc':

        # Process waves to MFCC
        X_processed = data_preprocessing.wav2mfcc(
            X_wav,
            sample_rate,
            stop_time,
            librosa_params,
        )

    else:
        raise utils.UserError(f"Invalid parameter feature={feature}. Expected \'wav\', \'stft\', \'mel\' or \'mfcc\'")

    # Retrieve input_shape
    PREPROCESSING_PARAMS['input_shape'] = X_processed.shape[1:]

    # Save datasets into target directory                                                   ")
    np.save(os.path.join(target_dir, 'X_processed.npy'), X_processed)

    # Save metadata
    metadata['preprocessing_params'] = PREPROCESSING_PARAMS
    with open(os.path.join(target_dir, 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp, sort_keys=True, indent=4)

    # Verbosity
    print(f"Data stored in {PREPROCESSING_PARAMS['target_dir']}.                                                  ")
