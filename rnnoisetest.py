import argparse
import sounddevice as sd
import numpy as np
from pyrnnoise import RNNoise  # Import pyrnnoise for noise suppression

# Helper function for argument parsing
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

# Argument parser setup
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true', help='Show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(
    description="Real-time audio processing with noise suppression",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser]
)
parser.add_argument('-i', '--input-device', type=int_or_str, help='Input device (numeric ID or substring)')
parser.add_argument('-o', '--output-device', type=int_or_str, help='Output device (numeric ID or substring)')
parser.add_argument('-c', '--channels', type=int, default=1, help='Number of channels (1 for mono, 2 for stereo)')
parser.add_argument('--dtype', help='Audio data type', default='float32')
parser.add_argument('--samplerate', type=float, help='Sampling rate', default=48000)  # Default to 48kHz
parser.add_argument('--blocksize', type=int, help='Block size')
parser.add_argument('--latency', type=float, help='Latency in seconds')
args = parser.parse_args(remaining)

# Initialize RNNoise denoiser
denoiser = RNNoise(sample_rate=int(args.samplerate))

# Define the audio callback function
def callback(indata, outdata, frames, time, status):
    """Processes live audio input, applies noise suppression, and plays back the denoised sound."""
    if status:
        print("⚠️ Stream Status:", status)

    print(f"🎙️ Capturing {frames} frames of audio... Shape: {indata.shape}")

    # Ensure `indata` is always 2D (stereo format)
    is_mono = False
    if indata.ndim == 1:  # If it's mono (1D), reshape it to 2D
        indata = indata.reshape(-1, 1)  # Convert mono (1D) to stereo-like (2D)
        is_mono = True

    # Normalize input audio to the range expected by RNNoise
    indata = indata.astype(np.float32)

    # Apply denoising
    denoised_audio = np.zeros_like(indata)
    for ch in range(indata.shape[1]):  # Iterate over channels
        denoised_audio[:, ch] = denoiser.process_frame(indata[:, ch], last=False)

    # If the original input was mono, reshape it back to 1D
    if is_mono:
        denoised_audio = denoised_audio.flatten()

    # Ensure the output shape matches input shape
    outdata[:] = denoised_audio.reshape(outdata.shape)

# Set up and start the real-time audio stream
try:
    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=args.samplerate, blocksize=args.blocksize,
                   dtype=args.dtype, latency=args.latency,
                   channels=args.channels, callback=callback):
        print('#' * 80)
        print('🔊 Real-time noise suppression active. Speak into your mic.')
        print('🎤 Press Return to stop.')
        print('#' * 80)
        input()  # Wait for user input to stop the script
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
