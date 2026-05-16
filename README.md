# MCNR

signal processing based multi channel noise reduction

## Installation

from GitHub repository:

```bash
pip install git+https://github.com/s23c1130/mcnr.git
```

from the local repository:

```bash
pip install -e .
```

note: -e option is suggested to install the package in the editable mode.

## Usage

### Command Line Interface

The following command will apply the noise reduction to the input.wav and save the result to the output.wav.

The input.wav shoudl be 16kHz, 16bit, multi-channel wav file.

```bash
mcnr -i input.wav output.wav
```

Options:
- `-i`, `--input`: input wav file (required)
- `--fft_size`: FFT size (default: 512)
- `--hop_size`: hop size (default: 128)

- `--NoiseIntensity`: Noise reduction power (0.0: weak ~ 1.0: strongest) (default:1.0)

### As a Python Module

```python
import mcnr
import soundfile

# load the input wav file
x, fs = soundfile.read('input.wav')

# transpose the input signal (n_samples, n_channels) -> (n_channels, n_samples)
x = x.T

# apply the noise reduction
y = mcnr.do_multi_channel_noise_reduction(x)

# save the result
soundfile.write('output.wav', y.T, fs)
```

The parameters of the noise reduction can be adjusted by the following arguments:

```python
y = mcnr.do_multi_channel_noise_reduction(x, fft_size=512, hop_size=128)
```

## Reference

The algorithm is based on the following paper:

1. Aoki M, Aoki S, Okamoto M. Sound source segregation using inter-channel phase and intensity differences. Proc Acoust Soc Japan Vol. 1-7-13, p 489–490, 1996.
