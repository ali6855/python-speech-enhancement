import os
import numpy as np
import soundfile as sf
from pesq import pesq
from pns.noise_suppressor import NoiseSuppressor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure export directory exists
EXPORT_DIR = "export"
os.makedirs(EXPORT_DIR, exist_ok=True)

def test():
    # Prepare Data 
    clean_files = ["data/sp02.wav", "data/sp04.wav", "data/sp06.wav", "data/sp09.wav"]
    input_files = ["data/sp02_train_sn5.wav", 
                   "data/sp04_babble_sn10.wav", 
                   "data/sp06_babble_sn5.wav", 
                   "data/sp09_babble_sn10.wav"]
    output_files = [os.path.join(EXPORT_DIR, "sp02_train_sn5_processed.wav"), 
                    os.path.join(EXPORT_DIR, "sp04_babble_sn10_processed.wav"),
                    os.path.join(EXPORT_DIR, "sp06_babble_sn5_processed.wav"), 
                    os.path.join(EXPORT_DIR, "sp09_babble_sn10_processed.wav")]

    for i in range(len(input_files)):
        clean_file = clean_files[i]
        input_file = input_files[i]
        output_file = output_files[i]

        try:
            clean_wav, _  = sf.read(clean_file)
            noisy_wav, fs = sf.read(input_file)
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            continue
        except Exception as e:
            logging.error(f"Error reading files: {e}")
            continue

        # Initialize
        noise_suppressor = NoiseSuppressor(fs)
        frame_size = noise_suppressor.get_frame_size()
        xfinal = np.zeros(len(noisy_wav))

        # Start Processing
        k = 0
        while k + frame_size < len(noisy_wav):
            frame = noisy_wav[k : k + frame_size]
            xfinal[k : k + frame_size] = noise_suppressor.process_frame(frame)
            k += frame_size

        # Save Results
        xfinal = xfinal / (np.max(np.abs(xfinal)) + 1e-10)
        sf.write(output_file, xfinal, fs)

        # Performance Metrics
        logging.info(f"Processing completed for {input_file}")
        try:
            pesq_nb = pesq(ref=clean_wav, deg=noisy_wav, fs=fs, mode='nb')
            logging.info(f"Input PESQ (NB): {pesq_nb:.4f}")
            pesq_nb = pesq(ref=clean_wav, deg=xfinal, fs=fs, mode='nb')
            logging.info(f"Output PESQ (NB): {pesq_nb:.4f}")

            if fs > 8000:
                pesq_wb = pesq(ref=clean_wav, deg=noisy_wav, fs=fs, mode='wb')
                logging.info(f"Input PESQ (WB): {pesq_wb:.4f}")
                pesq_wb = pesq(ref=clean_wav, deg=xfinal, fs=fs, mode='wb')
                logging.info(f"Output PESQ (WB): {pesq_wb:.4f}")
        except Exception as e:
            logging.error(f"Error calculating PESQ: {e}")

def denoise_all_files(input_files, output_files):
    for input_file, output_file in zip(input_files, output_files):
        try:
            noisy_wav, fs = sf.read(input_file)
        except FileNotFoundError:
            logging.error(f"File not found: {input_file}")
            continue
        except Exception as e:
            logging.error(f"Error reading file {input_file}: {e}")
            continue

        channels = noisy_wav.shape[1] if noisy_wav.ndim > 1 else 1
        logging.info(f"Input file: {input_file}")
        logging.info(f"Sample rate: {fs} Hz")
        logging.info(f"Number of channels: {channels}")

        if channels > 1:
            xfinal = np.zeros_like(noisy_wav)

            for ch in range(channels):
                noise_suppressor = NoiseSuppressor(fs)
                frame_size = noise_suppressor.get_frame_size()

                k = 0
                while k + frame_size < len(noisy_wav):
                    frame = noisy_wav[k : k + frame_size, ch]
                    xfinal[k : k + frame_size, ch] = noise_suppressor.process_frame(frame)
                    k += frame_size

                xfinal[:, ch] = xfinal[:, ch] / (np.max(np.abs(xfinal[:, ch])) + 1e-10)
        else:
            noise_suppressor = NoiseSuppressor(fs)
            frame_size = noise_suppressor.get_frame_size()
            xfinal = np.zeros(len(noisy_wav))

            k = 0
            while k + frame_size < len(noisy_wav):
                frame = noisy_wav[k : k + frame_size]
                xfinal[k : k + frame_size] = noise_suppressor.process_frame(frame)
                k += frame_size

            xfinal = xfinal / (np.max(np.abs(xfinal)) + 1e-10)

        sf.write(output_file, xfinal, fs)
        logging.info(f"Output file saved: {output_file}")

if __name__ == "__main__":
    input_files = ["data/sp02_train_sn5.wav", 
                   "data/sp04_babble_sn10.wav", 
                   "data/sp06_babble_sn5.wav", 
                   "data/sp09_babble_sn10.wav"]
    output_files = [os.path.join(EXPORT_DIR, "sp02_train_sn5_processed.wav"), 
                    os.path.join(EXPORT_DIR, "sp04_babble_sn10_processed.wav"),
                    os.path.join(EXPORT_DIR, "sp06_babble_sn5_processed.wav"), 
                    os.path.join(EXPORT_DIR, "sp09_babble_sn10_processed.wav")]

    denoise_all_files(input_files, output_files)
    # Uncomment the following line to test batch processing with metrics
    # test()
