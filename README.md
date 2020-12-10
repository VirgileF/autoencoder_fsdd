# autoencoder_fsdd
An autoencoder for Free Spoken Digits Dataset (FSDD).

## End-to-end commands

### Setting up the environment
Set up a conda environment using the file `environment.yml`.
```buildoutcfg
conda env create -f environment.yml
conda activate autoencoder_fsdd_env
```

### Data loading
Load the data from the recordings in `resources/recordings/` to `datasets/source_data/<data_folder>/`. Parameters in the file `loading_params.json`.
```buildoutcfg
python load_data.py loading_params.json
```

### Data preprocessing
Process audio data from wave to more advanced feature like mel spectrograms of MFCC. Parameters in the file `preprocessing_params.json`.
```buildoutcfg
python process_data.py preprocessing_params.json
```

### Autoencoder training
Train the autoencoder. Parameters in the file `autoencoder_training_params.json`. Model weights and metadat saved to `<save_model_to>` path.
```buildoutcfg
python train_autoencoder.py autoencoder_training_params.json
```

### Model selection
Select the model using jupyter notebook `model_selection.ipynb`. You will need to create an jupyter kernel running in the right environment.
Run the following command while being in your environment.
```buildoutcfg
python -m ipykernel install --user --name=autoencoder_fsdd_env
```
Then you can open your jupyter notebook and select `change kernel`, then `autoencoder_fsdd_env`.

## Speech representations

## Evaluation pipeline

## Results

