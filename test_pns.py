import numpy as np
import soundfile as sf
from pesq import pesq
import os

from pns.noise_suppressor import NoiseSuppressor  # فرض بر این است که این ماژول در دسترس است.

def test():
    # Prepare Data 
    clean_files = ["data/sp02.wav", "data/sp04.wav", "data/sp06.wav", "data/sp09.wav"]
    input_files = ["data/sp02_train_sn5.wav", "data/sp04_babble_sn10.wav", "data/sp06_babble_sn5.wav", "data/sp09_babble_sn10.wav"]
    output_files = ["data/sp02_train_sn5_processed.wav", "data/sp04_babble_sn10_processed.wav", "data/sp06_babble_sn5_processed.wav", "data/sp09_babble_sn10_processed.wav"]
    
    # Ensure file lists have the same length
    assert len(clean_files) == len(input_files) == len(output_files), "File lists must have the same length."

    for i in range(len(input_files)):
        clean_file = clean_files[i]
        input_file = input_files[i]
        output_file = output_files[i]

        # Check if files exist
        if not os.path.exists(clean_file) or not os.path.exists(input_file):
            print(f"Skipping {input_file} due to missing file.")
            continue

        # Read clean and noisy files
        clean_wav, _ = sf.read(clean_file)
        noisy_wav, fs = sf.read(input_file)

        # Ensure signals are the same length
        min_len = min(len(clean_wav), len(noisy_wav))
        clean_wav = clean_wav[:min_len]
        noisy_wav = noisy_wav[:min_len]

        # Initialize noise suppressor
        noise_suppressor = NoiseSuppressor(fs)

        # Process frames
        frame_size = noise_suppressor.get_frame_size()
        xfinal = np.zeros(len(noisy_wav))
        k = 0
        while k + frame_size < len(noisy_wav):
            frame = noisy_wav[k: k + frame_size]
            xfinal[k: k + frame_size] = noise_suppressor.process_frame(frame)
            k += frame_size

        # Normalize output
        max_val = max(np.abs(xfinal))
        if max_val > 0:
            xfinal = xfinal / max_val

        # Save processed output
        sf.write(output_file, xfinal, fs)

        # Calculate PESQ metrics
        print(f"\nProcessing file: {input_file}")
        try:
            pesq_nb = pesq(ref=clean_wav, deg=noisy_wav, fs=fs, mode='nb')
            print(f"Input PESQ (NB): {pesq_nb:.4f}")
            pesq_nb = pesq(ref=clean_wav, deg=xfinal, fs=fs, mode='nb')
            print(f"Output PESQ (NB): {pesq_nb:.4f}")
            if fs >= 16000:
                pesq_wb = pesq(ref=clean_wav, deg=noisy_wav, fs=fs, mode='wb')
                print(f"Input PESQ (WB): {pesq_wb:.4f}")
                pesq_wb = pesq(ref=clean_wav, deg=xfinal, fs=fs, mode='wb')
                print(f"Output PESQ (WB): {pesq_wb:.4f}")
        except Exception as e:
            print(f"Error calculating PESQ: {e}")

def denoise_file(input_file, output_file):
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
                xfinal[k: k + frame_size, ch] = noise_suppressor.process_frame(frame)
                k += frame_size
            max_val = max(np.abs(xfinal[:, ch]))
            if max_val > 0:
                xfinal[:, ch] /= max_val
    else:
        noise_suppressor = NoiseSuppressor(fs)
        x = noisy_wav
        frame_size = noise_suppressor.get_frame_size()
        xfinal = np.zeros(len(x))
        k = 0
        while k + frame_size < len(x):
            frame = x[k: k + frame_size]
            xfinal[k: k + frame_size] = noise_suppressor.process_frame(frame)
            k += frame_size
        max_val = max(np.abs(xfinal))
        if max_val > 0:
            xfinal /= max_val

    # Save the processed file
    sf.write(output_file, xfinal, fs)

if __name__ == "__main__":
    # Example usage
    try:
        denoise_file("data/sp02_train_sn5.wav", "data/sp02_train_sn5_processed.wav")
        test()
    except Exception as e:
        print(f"An error occurred: {e}")
