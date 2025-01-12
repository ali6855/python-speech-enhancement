import os
import numpy as np
import soundfile as sf
from pesq import pesq
from ipywidgets import interact, widgets
from google.colab import files
from pns.noise_suppressor import NoiseSuppressor  # فرض بر این است که این ماژول موجود است.

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

def upload_file_and_process():
    """Upload file, process it, and download the result."""
    uploaded = files.upload()
    for filename in uploaded.keys():
        input_file = filename
        output_file = os.path.join(EXPORT_DIR, os.path.basename(input_file).replace(".wav", "_processed.wav"))

        # Process audio
        processed_file = process_audio(
            input_file=input_file,
            output_file=output_file,
            frame_size=frame_size_slider.value,
            window_type=window_type_dropdown.value
        )

        print(f"Processing complete. File saved at: {processed_file}")
        files.download(processed_file)

# Interactive widgets
frame_size_slider = widgets.IntSlider(value=512, min=128, max=4096, step=128, description="Frame Size:")
window_type_dropdown = widgets.Dropdown(options=["hamming", "hann", "none"], value="hamming", description="Window Type:")

# Display widgets and button
print("1. آپلود فایل صوتی:")
print("2. تنظیم پارامترها:")
display(frame_size_slider, window_type_dropdown)

process_button = widgets.Button(description="Process and Download")
process_button.on_click(lambda x: upload_file_and_process())
display(process_button)
