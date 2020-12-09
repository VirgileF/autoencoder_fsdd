import librosa
import os
import time as T
import numpy as np

import utils

def wav2stft(X_wav, sr, stop_time, librosa_params, google_colab=True):

    """
    Function that transforms 2d-array containing the mono waves (X_wav) into a 
    4d-array containing the modules and phases of each STFT matrix, computed 
    from the waves (X_stft).
    
    Parameters
    ----------
    X_wav : np.ndarray [shape=(n_recordings, sr*stop_time)]
        Dataset of waves (can be loaded from X_wav.npy).

    sr : int > 0 [scalar]
        Sample rate (depends on the initial .wav files)

    stop_time : real > 0 [scalar]
        Recodings' temporal length (in seconds).

    librosa_params : dictionary containing the following parameters

        - n_fft : int > 0 [scalar]
            Number of FFT components in the resulting STFT.

        - hop_length : int > 0 [scalar]
            Number audio of frames between STFT columns.
            If unspecified, defaults `win_length / 4`.

        - win_length  : int <= n_fft [scalar]
            Each frame of audio is windowed by `window()`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`.
        
        - window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            - a window specification (string, tuple, or number);
              see `scipy.signal.get_window`
            - a window function, such as `scipy.signal.hanning`
            - a vector or array of length `n_fft`.
    
        - google_colab : boolean
            Specifies if the environment is running on Google Colab
            or not.

    Returns
    -------
    X_stft : np.ndarray [shape=(n_recordings, 2, 1 + n_fft/2, sr*stop_time//hop_length+1), dtype=dtype]
             Dataset containing the STFT matrix (amplitude and phase) of each 
             recording.

    """

    n_fft = librosa_params['n_fft']
    hop_length = librosa_params['hop_length']
    win_length = librosa_params['win_length']
    window = librosa_params['window']
    
    import time as T
    t0 = T.time()

    stft_shape = (2, n_fft//2+1, stop_time*sr//hop_length+1)
    X_stft = np.zeros((X_wav.shape[0], *stft_shape))

    for i in range(X_wav.shape[0]):
      
        eta = X_wav.shape[0] * (T.time() - t0) / (i+1) - T.time() + t0 + 10**(-2)
        
        print('Wave to STFT: step {}/{}... -- ETA: {} seconds.'.format(i+1, X_wav.shape[0], round(eta, 1)), end='\r')

        samples = X_wav[i]

        # stft
        stft_matrix = librosa.core.stft(
            samples,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window)

        # amplitude and phase
        amp = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)

        # insert into spec_dataset
        X_stft[i, 0, :, :] = amp
        X_stft[i, 1, :, :] = phase

    return X_stft

def stft2wav(X_stft, librosa_params, google_colab=True):

    """
    Function that transforms 4d-array containing the modules and phases of each 
    STFT matrix into a 2d-array containing the mono waves (X_wav).
    
    Parameters
    ----------
    X_stft : np.ndarray [shape=(n_recordings, 2, 1 + n_fft/2, sr*stop_time//hop_length+1)]
        Dataset of stft matrices (loaded from X_stft.npy).

    librosa_params : dictionary containing the following parameters

        - n_fft : int > 0 [scalar]
            Number of FFT components in the resulting STFT.

        - hop_length : int > 0 [scalar]
            Number audio of frames between STFT columns.
            If unspecified, defaults `win_length / 4`.

        - win_length  : int <= n_fft [scalar]
            Each frame of audio is windowed by `window()`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`.
        
        - window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            - a window specification (string, tuple, or number);
              see `scipy.signal.get_window`
            - a window function, such as `scipy.signal.hanning`
            - a vector or array of length `n_fft`.
    
        - google_colab : boolean
            Specifies if the environment is running on Google Colab
            or not.

    Returns
    -------
    X_wav : np.ndarray [shape=(n_recordings, sr*stop_time), dtype=dtype]
             Dataset containing the wave of each recording.

    """

    assert len(X_stft.shape) == 4

    hop_length = librosa_params['hop_length']
    win_length = librosa_params['win_length']
    window = librosa_params['window']
    
    import time as T
    t0 = T.time()

    wav_shape = hop_length*(X_stft.shape[3]-1)
    X_wav = np.zeros((X_stft.shape[0], wav_shape))

    for i in range(X_stft.shape[0]):
      
        eta = X_stft.shape[0] * (T.time() - t0) / (i+1) - T.time() + t0 + 10**(-2)
        
        print('Wave to STFT: step {}/{}... -- ETA: {} seconds.'.format(i+1, X_stft.shape[0], round(eta, 1)), end='\r')

        stft_matrices = X_stft[i]

        stft_matrix = stft_matrices[0] * np.exp(1j*stft_matrices[1])

        wav_back = librosa.core.istft(
            stft_matrix,
            hop_length=hop_length,
            win_length=win_length,
            window='hann')

        # insert into spec_dataset
        X_wav[i, :] = wav_back

    return X_wav

"""# Waves to mel spectrograms"""

def wav2mel(X_wav, sr, stop_time, librosa_params, google_colab=True):

    """
    Function that transforms 2d-array containing the mono waves (X_wav) into a 
    3d-array containing the power mel spectrograms of each recording, computed 
    from the waves.
    
    Parameters
    ----------
    X_wav : np.ndarray [shape=(n_recordings, sr*stop_time)]
        Dataset of waves (loaded from X_wav.npy).

    sr : int > 0 [scalar]
        Sample rate (depends on the initial .wav files)

    stop_time : real > 0 [scalar]
        Recodings' temporal length (in seconds).

    librosa_params : dictionary containing the following parameters

        - sigma_noise : real > 0 [scalar]
            Standard-dev of the white noise replacing silence.
        
        - n_fft : int > 0 [scalar]
            Number of FFT components in the resulting STFT.

        - hop_length : int > 0 [scalar]
            Number audio of frames between STFT columns.
            If unspecified, defaults `win_length / 4`.

        - win_length  : int <= n_fft [scalar]
            Each frame of audio is windowed by `window()`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`.
        
        - window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            - a window specification (string, tuple, or number);
              see `scipy.signal.get_window`
            - a window function, such as `scipy.signal.hanning`
            - a vector or array of length `n_fft`.

        - n_mels : int > 0 [scalar]
            Number of mel-filterbanks.

        - mel_in_db : boolean 
            If the mel-spectrogram is returned in dB or not. If not, it is returned
            as a power.

        - top_db : float >= 0 [scalar]
            threshold the output at `top_db` below the peak:
            ``max(10 * log10(S)) - top_db``

        - amin : float > 0 [scalar]
            minimum threshold for `abs(S)` and `ref`
    
    google_colab : boolean
        Specifies if the environment is running on Google Colab or not.

    Returns
    -------
    X_mel : np.ndarray [shape=(n_recordings, n_mels, sr*stop_time//hop_length+1), dtype=dtype]
             Dataset containing the mel-spectrograms of each recording.

    """

    sigma_noise = librosa_params['sigma_noise']
    n_fft = librosa_params['n_fft']
    hop_length = librosa_params['hop_length']
    win_length = librosa_params['win_length']
    window = librosa_params['window']
    n_mels = librosa_params['n_mels']
    mel_in_db = librosa_params['mel_in_db']
    top_db = librosa_params['top_db']
    amin = librosa_params['amin']
    
    import time as T
    t0 = T.time()

    mel_shape = (n_mels, sr*stop_time//hop_length+1)
    X_mel = np.zeros((X_wav.shape[0], *mel_shape))

    for i in range(X_wav.shape[0]):
      
        eta = X_wav.shape[0] * (T.time() - t0) / (i+1) - T.time() + t0 + 10**(-2)
        
        print('Wave to mel-spectrogram: recording {}/{}... -- ETA: {} seconds.'.format(i+1, X_wav.shape[0], round(eta, 1)), end='\r')

        samples = X_wav[i]

        samples[samples==0] = np.random.normal(0, sigma_noise, size=np.sum(samples==0))

        mel_spec = librosa.feature.melspectrogram(
            samples,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels
        )

        if mel_in_db:
            mel_spec = librosa.core.power_to_db(mel_spec, top_db=top_db, amin=amin)

        # insert into melspec_dataset
        X_mel[i, :, :] = mel_spec

    return X_mel

def mel2wav(X_mel, sr, librosa_params, google_colab=True):

    """
    Function that transforms 3d-array containing the mel-spectrogram 
    (power or mel) into a 2d-array containing the mono waves (X_wav).
    
    Parameters
    ----------
    X_mel : np.ndarray [shape=(n_recordings, n_mels, sr*stop_time//hop_length+1)]
        Dataset of mel spectrograms (can be loaded from X_mel.npy).

    sr : int > 0 [scalar]
        Sample rate (depends on the initial .wav files)

    librosa_params : dictionary containing the following parameters

        - n_fft : int > 0 [scalar]
            Number of FFT components in the resulting STFT.

        - hop_length : int > 0 [scalar]
            Number audio of frames between STFT columns.
            If unspecified, defaults `win_length / 4`.

        - win_length  : int <= n_fft [scalar]
            Each frame of audio is windowed by `window()`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`.
        
        - window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            - a window specification (string, tuple, or number);
              see `scipy.signal.get_window`
            - a window function, such as `scipy.signal.hanning`
            - a vector or array of length `n_fft`.
        
        - n_mels : int > 0 [scalar]
            Number of mel-filterbanks.

        - mel_in_db : boolean 
            If the mel-spectrogram is returned in dB or not. If not, it is returned
            as a power.
    
    google_colab : boolean
        Specifies if the environment is running on Google Colab
        or not.


    Returns
    -------
    X_wav : np.ndarray [shape=(n_recordings, sr*stop_time), dtype=dtype]
             Dataset containing the wave of each recording.

    """

    assert len(X_mel.shape) == 3

    n_fft = librosa_params['n_fft']
    hop_length = librosa_params['hop_length']
    win_length = librosa_params['win_length']
    window = librosa_params['window']
    n_mels = librosa_params['n_mels']
    mel_in_db = librosa_params['mel_in_db']
    
    import time as T
    t0 = T.time()

    wav_shape = hop_length*(X_mel.shape[2]-1)
    X_wav = np.zeros((X_mel.shape[0], wav_shape))

    for i in range(X_mel.shape[0]):
      
        eta = X_mel.shape[0] * (T.time() - t0) / (i+1) - T.time() + t0 + 10**(-2)
        
        print('Mel to wave: step {}/{}... -- ETA: {} seconds.'.format(i+1, X_mel.shape[0], round(eta, 1)), end='\r')

        mel_spec = X_mel[i]
        
        if mel_in_db:
            mel_spec = librosa.core.db_to_power(mel_spec)

        wav_back = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window
        )

        # insert into spec_dataset
        X_wav[i, :] = wav_back

    return X_wav

"""# Waves to MFCC"""

def wav2mfcc(X_wav, sr, stop_time, librosa_params, google_colab=True):

    """
    Function that transforms 2d-array containing the mono waves (X_wav) into a 
    3d-array containing the MFCC matrix of each recording, computed 
    from the waves.
    
    Parameters
    ----------
    X_wav : np.ndarray [shape=(n_recordings, sr*stop_time)]
        Dataset of waves (loaded from X_wav.npy).

    sr : int > 0 [scalar]
        Sample rate (depends on the initial .wav files)

    stop_time : real > 0 [scalar]
        Recodings' temporal length (in seconds).

    librosa_params : dictionary containing the following parameters
        
        - sigma_noise : real > 0 [scalar]
            Standard-dev of the white noise replacing silence.
            
        - n_fft : int > 0 [scalar]
            Number of FFT components in the resulting STFT.

        - hop_length : int > 0 [scalar]
            Number audio of frames between STFT columns.
            If unspecified, defaults `win_length / 4`.

        - win_length  : int <= n_fft [scalar]
            Each frame of audio is windowed by `window()`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`.
        
        - window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            - a window specification (string, tuple, or number);
              see `scipy.signal.get_window`
            - a window function, such as `scipy.signal.hanning`
            - a vector or array of length `n_fft`.

        - n_mels : int > 0 [scalar]
            Number of mel-filterbanks.

        - n_mfcc : int > 0 [scalar]
            Number of cepstral coefficient computed in the MFCC matrix.
    
    google_colab : boolean
        Specifies if the environment is running on Google Colab or not.

    Returns
    -------
    X_mfcc : np.ndarray [shape=(n_recordings, n_mfcc, sr*stop_time//hop_length+1), dtype=dtype]
             Dataset containing the MFCC matrix of each recording.

    """

    sigma_noise = librosa_params['sigma_noise']
    n_fft = librosa_params['n_fft']
    hop_length = librosa_params['hop_length']
    win_length = librosa_params['win_length']
    window = librosa_params['window']
    n_mels = librosa_params['n_mels']
    n_mfcc = librosa_params['n_mfcc']
    
    import time as T
    t0 = T.time()

    mfcc_shape = (n_mfcc, sr*stop_time//hop_length+1)
    X_mfcc = np.zeros((X_wav.shape[0], *mfcc_shape))

    for i in range(X_wav.shape[0]):
      
        eta = X_wav.shape[0] * (T.time() - t0) / (i+1) - T.time() + t0 + 10**(-2)
        
        print('Wave to MFCC: recording {}/{}... -- ETA: {} seconds.'.format(i+1, X_wav.shape[0], round(eta, 1)), end='\r')

        samples = X_wav[i]

        samples[samples==0] = np.random.normal(0, sigma_noise, size=np.sum(samples==0))

        mfcc = librosa.feature.mfcc(
            y=samples, 
            sr=8000, 
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels
        )

        # insert into melspec_dataset
        X_mfcc[i, :, :] = mfcc

    return X_mfcc

def mfcc2wav(X_mfcc, sr, librosa_params, google_colab=True):

    """
    Function that transforms 3d-array containing the MFCC
    into a 2d-array containing the mono waves (X_wav).
    
    Parameters
    ----------
    X_mfcc : np.ndarray [shape=(n_recordings, n_mfcc, sr*stop_time//hop_length+1)]
        Dataset of mel spectrograms (can be loaded from X_mel.npy).

    sr : int > 0 [scalar]
        Sample rate (depends on the initial .wav files)

    librosa_params : dictionary containing the following parameters

        - n_fft : int > 0 [scalar]
            Number of FFT components in the resulting STFT.

        - hop_length : int > 0 [scalar]
            Number audio of frames between STFT columns.
            If unspecified, defaults `win_length / 4`.

        - win_length  : int <= n_fft [scalar]
            Each frame of audio is windowed by `window()`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`.
        
        - window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
            - a window specification (string, tuple, or number);
              see `scipy.signal.get_window`
            - a window function, such as `scipy.signal.hanning`
            - a vector or array of length `n_fft`.

        - n_mels : int > 0 [scalar]
            Number of mel-filterbanks.
    
    google_colab : boolean
        Specifies if the environment is running on Google Colab
        or not.

  
    Returns
    -------
    X_wav : np.ndarray [shape=(n_recordings, sr*stop_time), dtype=dtype]
             Dataset containing the wave of each recording.

    """

    assert len(X_mfcc.shape) == 3

    n_fft = librosa_params['n_fft']
    hop_length = librosa_params['hop_length']
    win_length = librosa_params['win_length']
    window = librosa_params['window']
    n_mels = librosa_params['n_mels']
    
    import time as T
    t0 = T.time()

    wav_shape = hop_length*(X_mfcc.shape[2]-1)
    X_wav = np.zeros((X_mfcc.shape[0], wav_shape))

    if google_colab:
        display_output = display(IPython.display.Pretty('Starting'), display_id=True)

    for i in range(X_mfcc.shape[0]):
      
        eta = X_mfcc.shape[0] * (T.time() - t0) / (i+1) - T.time() + t0 + 10**(-2)
        
        print('MFCC to wave: step {}/{}... -- ETA: {} seconds.'.format(i+1, X_mfcc.shape[0], round(eta, 1)), end='\r')

        mfcc = X_mfcc[i]

        wav_back = librosa.feature.inverse.mfcc_to_audio(
            mfcc, 
            n_mels=n_mels, 
            dct_type=2, 
            norm="ortho", 
            ref=1.0, 
            lifter=0,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window
        )

        # insert into spec_dataset
        X_wav[i, :] = wav_back

    return X_wav

