import os
import numpy as np
import soundfile as sf
from pesq import pesq

from pns.noise_suppressor import NoiseSuppressor  # فرض بر این است که این ماژول در دسترس است.

# Ensure export directory exists
EXPORT_DIR = "export"
os.makedirs(EXPORT_DIR, exist_ok=True)

def apply_window(frame, window_type="hamming"):
    """Apply a windowing function to the frame."""
    if window_type == "hamming":
        window = np.hamming(len(frame))
    elif window_type == "hann":
        window = np.hanning(len(frame))
    else:
        window = np.ones(len(frame))  # No window
    return frame * window

def denoise_and_enhance(input_file, output_file, window_type="hamming", normalize=True):
    """Denoise and enhance audio for better clarity."""
    # Ensure export directory exists
    os.makedirs(EXPORT_DIR, exist_ok=True)
    output_file = os.path.join(EXPORT_DIR, os.path.basename(output_file))

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    noisy_wav, fs = sf.read(input_file)
    channels = noisy_wav.shape[1] if noisy_wav.ndim > 1 else 1
    print(f"Input file: {input_file}")
    print(f"Sample rate: {fs} Hz")
    print(f"Number of channels: {channels}")
    print(f"Output file: {output_file}")

    if channels > 1:
        xfinal = np.zeros(noisy_wav.shape)
        for ch in range(channels):
            noise_suppressor = NoiseSuppressor(fs)
            x = noisy_wav[:, ch]
            frame_size = noise_suppressor.get_frame_size()
            k = 0
            while k + frame_size < len(x):
                frame = x[k: k + frame_size]
                frame = apply_window(frame, window_type)  # Apply window
                xfinal[k: k + frame_size, ch] = noise_suppressor.process_frame(frame)
                k += frame_size
            max_val = max(np.abs(xfinal[:, ch]))
            if normalize and max_val > 0:
                xfinal[:, ch] /= max_val
    else:
        noise_suppressor = NoiseSuppressor(fs)
        x = noisy_wav
        frame_size = noise_suppressor.get_frame_size()
        xfinal = np.zeros(len(x))
        k = 0
        while k + frame_size < len(x):
            frame = x[k: k + frame_size]
            frame = apply_window(frame, window_type)  # Apply window
            xfinal[k: k + frame_size] = noise_suppressor.process_frame(frame)
            k += frame_size
        max_val = max(np.abs(xfinal))
        if normalize and max_val > 0:
            xfinal /= max_val

    # Save the processed file
    sf.write(output_file, xfinal, fs)
    print("Denoising and enhancement complete.")

if __name__ == "__main__":
    # Example usage
    try:
        denoise_and_enhance("data/sp02_train_sn5.wav", "sp02_train_sn5_enhanced.wav", window_type="hamming", normalize=True)
    except Exception as e:
        print(f"An error occurred: {e}")
