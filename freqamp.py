import argparse
import sounddevice as sd
import numpy as np
import scipy.signal as signal  # For filtering

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


# Initial argument parsing
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-l', '--list-devices', action='store_true', help='Show list of audio devices and exit')
args, remaining = parser.parse_known_args()

if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

# Main argument parser
parser = argparse.ArgumentParser(
    description="Audio streaming with frequency amplification for presbycusis",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser]
)
parser.add_argument('-i', '--input-device', type=int_or_str, help='Input device (numeric ID or substring)')
parser.add_argument('-o', '--output-device', type=int_or_str, help='Output device (numeric ID or substring)')
parser.add_argument('--dtype', help='Audio data type')
parser.add_argument('--samplerate', type=float, default=44100, help='Sampling rate (default: 44100 Hz)')
parser.add_argument('--blocksize', type=int, help='Block size')
parser.add_argument('--latency', type=float, help='Latency in seconds')

args = parser.parse_args(remaining)

# Set defaults based on your available devices
if args.input_device is None:
    args.input_device = 1  # Default Microphone

if args.output_device is None:
    args.output_device = 2  # Default Speakers

# Get device info for selected input device
input_device_info = sd.query_devices(args.input_device)
max_input_channels = input_device_info['max_input_channels']

# Ensure the number of channels doesn't exceed the device limit
parser.add_argument('-c', '--channels', type=int, default=max_input_channels, help='Number of channels')
args = parser.parse_args(remaining)  # Re-parse to update args.channels

print(f"Using input device {args.input_device}: {input_device_info['name']} (Supports {max_input_channels} channels)")
print(f"Using output device {args.output_device}")

# ===========================
# Adjustable Parameters
# ===========================

GAIN_HIGH_FREQUENCY = 2.0  # Amplification factor for high frequencies (default: 2.0x boost)
CUTOFF_FREQUENCY = 2000  # Frequency (Hz) above which we boost (default: 2000 Hz for high frequencies)

# ===========================
# High-Frequency Boosting Filter
# ===========================
def high_frequency_boost_filter(fs, gain=GAIN_HIGH_FREQUENCY, cutoff=CUTOFF_FREQUENCY):
    """Designs a high-pass filter to boost high frequencies."""
    nyquist = 0.5 * fs  # Nyquist frequency
    normalized_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = signal.butter(2, normalized_cutoff, btype='high', analog=False)
    
    # Apply gain to high frequencies
    b *= gain  
    return b, a

# Precompute filter coefficients
b, a = high_frequency_boost_filter(args.samplerate)

# ===========================
# Audio Processing Callback
# ===========================
def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    
    # Convert input to float for processing
    processed_audio = indata.copy()
    
    # Apply high-frequency amplification
    processed_audio = signal.lfilter(b, a, processed_audio, axis=0)

    # Clip values to avoid distortion
    processed_audio = np.clip(processed_audio, -1.0, 1.0)
    
    outdata[:] = processed_audio

# ===========================
# Audio Streaming
# ===========================
try:
    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=args.samplerate, blocksize=args.blocksize,
                   dtype=args.dtype, latency=args.latency,
                   channels=args.channels, callback=callback):
        print('#' * 80)
        print('Press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
