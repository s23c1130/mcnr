import soundfile
import numpy as np
from .mcnr import do_multi_channel_noise_reduction


def load_audio_file(filename):
    # Load an audio file and return it as a numpy array
    y, sr = soundfile.read(filename)
    y = y.T

    # check if sr is 16000
    # assert sr == 16000, f"Sample rate of {sr} is not supported. Please resample to 16000."

    # check if y is multi-channel
    assert y.shape[0] > 1, f"Input audio file is not multi-channel."

    return y, sr


def save_audio_file(filename, x, sr=16000):
    # Save an audio file
    soundfile.write(filename, x.T, sr, subtype="PCM_16")

    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Filtering the multi-channel audio signal")
    parser.add_argument("--input", "-i", type=str, help="Input audio file", required=True)
    parser.add_argument("--fft_size", type=int, default=512, help="FFT size (default: 512)")
    parser.add_argument("--hop_size", type=int, default=128, help="Hop size (default: 128)")
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk size (default: None)")
    
    # Add
    parser.add_argument("--NoiseIntensity", type=float, default=1.0, help="Noise Reduction Intensity (0(weak) ~ 1(Strong))")

    parser.add_argument("output", type=str, help="Output audio file")

    args = parser.parse_args()

    # Load an audio file
    x, sr = load_audio_file(args.input)

    # Apply multi-channel noise reduction
    x = do_multi_channel_noise_reduction(x, fft_size=args.fft_size, hop_size=args.hop_size, chunk_size=args.chunk_size, noise_intensity=args.NoiseIntensity)

    # Save the output audio file
    save_audio_file(args.output, x, sr=sr)

if __name__ == "__main__":
    main()
