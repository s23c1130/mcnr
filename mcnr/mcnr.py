import librosa
import numpy as np


def proc_stft(x, fft_size=512, hop_size=128):
    """Compute the short-time Fourier transform of an audio signal.

    Args:
        x (np.ndarray): Input audio signal. (n_channels, n_samples)

    Returns:
        np.ndarray: Short-time Fourier transform of the input audio signal. (n_channels, n_fft//2 + 1, n_frames)
    """
    # Compute the short-time Fourier transform of an audio signal
    d = librosa.stft(x, n_fft=fft_size, hop_length=hop_size)

    return d


def proc_filtering(d, inplace=True, noise_intensity = 1.0):
    # calculate power spectrogram
    p = np.abs(d)**2

    # take the maimux channel index for each frame, each freq. bin
    p_max = np.argmax(p, axis=0)

    # make a copy of the input data
    d_out = d if inplace else d.copy()

    # leave only the maximum channel for each frame, each freq. bin
    for i in range(d.shape[0]):
        d_out[i, p_max != i] *= (1.0 - noise_intensity)

    return d_out


def proc_istft(d, hop_size=128):
    # Compute the inverse short-time Fourier transform of a complex-valued spectrogram
    x = librosa.istft(d, hop_length=hop_size)

    return x


def do_multi_channel_noise_reduction_(x, fft_size=512, hop_size=128, noise_intensity = 1.0):
    """Apply multi-channel noise reduction to an audio signal.

    Example:
        x = do_multi_channel_noise_reduction(x, fft_size=512, hop_size=128)

    Note:
        The input audio signal must have shape (n_channels, n_samples).
        Data type of the input audio signal must be float.

    Args:
        x (np.ndarray): Input audio signal. (n_channels, n_samples)
        fft_size (int): FFT size (default: 512)
        hop_size (int): Hop size (default: 128)

        noise_intensity:
        0: 除去しない
        1: 除去

    Returns:
        np.ndarray: Output audio signal. (n_channels, n_samples)
    """

    # Compute the short-time Fourier transform of the audio signal
    d = proc_stft(x, fft_size=fft_size, hop_size=hop_size)

    # Filtering
    d = proc_filtering(d, noise_intensity=noise_intensity)

    # Compute the inverse short-time Fourier transform of the complex-valued spectrogram
    x = proc_istft(d, hop_size=hop_size)

    return x


def do_multi_channel_noise_reduction(x, fft_size=512, hop_size=128, chunk_size=None, noise_intensity=1.0):
    """Apply multi-channel noise reduction to an audio signal.

    Example:
        x = do_multi_channel_noise_reduction(x, fft_size=512, hop_size=128, chunk_size=16000)

    Note:
        The input audio signal must have shape (n_channels, n_samples).
        Data type of the input audio signal must be float.

    Args:
        x (np.ndarray): Input audio signal. (n_channels, n_samples)
        fft_size (int): FFT size (default: 512)
        hop_size (int): Hop size (default: 128)
        chunk_size (int): Chunk size (default: None)


    Returns:
        np.ndarray: Output audio signal. (n_channels, n_samples)
    """
    if chunk_size is None:
        return do_multi_channel_noise_reduction_(x, fft_size=fft_size, hop_size=hop_size, noise_intensity=noise_intensity)

    y = np.zeros_like(x)

    for i in range(0, x.shape[1], chunk_size):
        x_chunk = x[:, i:i+chunk_size]
        y_chunk = do_multi_channel_noise_reduction_(x_chunk, fft_size=fft_size, hop_size=hop_size, noise_intensity=noise_intensity)
        if y_chunk.shape[1] < chunk_size:
            y[:, i:i+y_chunk.shape[1]] = y_chunk
        else:
            y[:, i:i+chunk_size] = y_chunk

    return y
