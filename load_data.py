import json
import os
import shutil
from scipy.io import wavfile
import numpy as np
import pandas as pd

import utils.utils as utils

import sys
if len(sys.argv) != 2:
    raise utils.UserError(f"Illegal command. Please type \'>> python <python_script.py> <params_file.json>\'.")

# Load parameters from json file
with open(sys.argv[1]) as f:
    LOADING_PARAMS = json.load(f)

# Retrieve paths from parameters
path_to_recordings = LOADING_PARAMS['path_to_recordings']
target_dir = LOADING_PARAMS['target_dir']

# Check source directory.
if not os.path.exists(path_to_recordings):
    raise utils.UserError(f"Parameter path_to_recordings={path_to_recordings} not allowed. The path does not exists.")

# Check target directory and confirm overwriting if exists.
if os.path.exists(target_dir):
    confirm = input(f"Folder \'{target_dir}\' already exists. Do you want to overwrite it? (y/n): ")
    if confirm == 'y':
        shutil.rmtree(target_dir)
    else:
        raise utils.UserError('User interrupted execution. Answer \'y\' to run script.')
os.makedirs(target_dir)

listfiles = os.listdir(path_to_recordings)
n_files = len(listfiles)

wav_shape = (1, LOADING_PARAMS['stop_time']*LOADING_PARAMS['sample_rate']) # remove recordings with more than 8k samples (0.2% of the dataset)
X_wav = np.zeros(wav_shape)
digits = []
speakers = []

for i, filename in enumerate(listfiles):

    digit = int(filename[0])
    speaker = filename.split('_')[1]

    path_to_file = os.path.join(path_to_recordings, filename)
    sample_rate, samples = wavfile.read(path_to_file)
    samples = samples.astype(np.float32)
    if sample_rate != LOADING_PARAMS['sample_rate']:
        raise utils.UserError(f"Parameter not allowed. sample_rate={LOADING_PARAMS['sample_rate']} does not fit. Expected value: {sample_rate}.")


    # if the file is stereo, let's keep only the left component
    if len(samples.shape) > 1:
        samples = samples[:,0]
        print('WARNING: Stereo recording: {} (index:{})'.format(filename, i))

    # Normalize wave
    if LOADING_PARAMS['normalize_waves']:
        samples = samples - np.mean(samples)
        samples = samples / np.max(np.abs(samples))

    n_samples = len(samples)
    time = np.arange(0, len(samples)) / sample_rate

    if n_samples <= 8000:

        new_row = np.zeros((1, 8000))
        new_row[0, 1:len(samples)+1] = samples

        X_wav = np.concatenate((X_wav, new_row), axis=0)

        digits.append(digit)
        speakers.append(speaker)

    else:

        print('WARNING: The recording is too long (more than 1 sec) -> index:{}'.format(i))

# Remove first row from "data" (contains only zeros)
X_wav = np.delete(X_wav, (0), axis=0)

# Make the speaker column ordinal
speakers_mapping = pd.Series(speakers, dtype='category').cat.categories
speakers = pd.Series(speakers, dtype='category').cat.codes.to_numpy()

# Make digits column a numpy array
digits = np.array(digits)

# Save datasets into target directory
print(f"Target directory: {target_dir}")
np.save(os.path.join(target_dir, 'X_wav.npy'), X_wav)
np.save(os.path.join(target_dir, 'y_digits.npy'), digits)
np.save(os.path.join(target_dir, 'y_speakers.npy'), speakers)
speakers_mapping.to_series().to_csv(os.path.join(target_dir, 'speakers_mapping.csv'))

# Save metadata
metadata = {}
metadata['loading_params'] = LOADING_PARAMS
with open(os.path.join(target_dir, 'metadata.json'), 'w') as fp:
    json.dump(metadata, fp, sort_keys=True, indent=4)

# Verbosity
print(f"Data stored in {LOADING_PARAMS['target_dir']}.")