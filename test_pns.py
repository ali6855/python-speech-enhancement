import os
import numpy as np
import soundfile as sf
from tkinter import Tk, Label, Button, Scale, HORIZONTAL, filedialog, StringVar, OptionMenu
from pesq import pesq

from pns.noise_suppressor import NoiseSuppressor

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

def process_audio(input_file, output_file, frame_size, window_type="hamming", normalize=True):
    """Process audio with selected parameters."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    noisy_wav, fs = sf.read(input_file)
    noise_suppressor = NoiseSuppressor(fs)
    x = noisy_wav
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
    sf.write(output_file, xfinal, fs)
    return output_file

def browse_file():
    """Browse and select an audio file."""
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    input_file_var.set(file_path)

def run_processing():
    """Run the audio processing with selected parameters."""
    input_file = input_file_var.get()
    if not input_file:
        print("Please select an input file.")
        return

    output_file = os.path.join(EXPORT_DIR, os.path.basename(input_file).replace(".wav", "_processed.wav"))
    frame_size = frame_size_scale.get()
    window_type = window_type_var.get()

    try:
        processed_file = process_audio(input_file, output_file, frame_size, window_type=window_type)
        print(f"Processing complete. File saved at: {processed_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Create GUI
root = Tk()
root.title("Audio Enhancer")

# Input file selection
Label(root, text="Input File:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
input_file_var = StringVar()
Button(root, text="Browse", command=browse_file).grid(row=0, column=1, padx=10, pady=5)
Label(root, textvariable=input_file_var).grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")

# Frame size selection
Label(root, text="Frame Size (samples):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
frame_size_scale = Scale(root, from_=128, to=4096, resolution=128, orient=HORIZONTAL)
frame_size_scale.set(512)  # Default value
frame_size_scale.grid(row=2, column=1, padx=10, pady=5)

# Window type selection
Label(root, text="Window Type:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
window_type_var = StringVar(value="hamming")
window_type_menu = OptionMenu(root, window_type_var, "hamming", "hann", "none")
window_type_menu.grid(row=3, column=1, padx=10, pady=5)

# Process button
Button(root, text="Process", command=run_processing).grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
