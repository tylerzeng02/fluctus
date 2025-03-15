import argparse
import tempfile
import queue
import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from pyrnnoise import RNNoise  # Import RNNoise for noise reduction

# Helper function for argument parsing
def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

# Argument parser setup
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='Show list of audio devices and exit'
)
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(
    description="Record audio and apply noise reduction using pyrnnoise",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser]
)
parser.add_argument('filename', nargs='?', metavar='FILENAME', help='Audio file to store recording')
parser.add_argument('-d', '--device', type=int_or_str, help='Input device (numeric ID or substring)')
parser.add_argument('-r', '--samplerate', type=int, help='Sampling rate')
parser.add_argument('-c', '--channels', type=int, default=1, help='Number of input channels')
parser.add_argument('-t', '--subtype', type=str, help='Sound file subtype (e.g. "PCM_24")')
args = parser.parse_args(remaining)

q = queue.Queue()

# Callback function for real-time audio streaming
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = int(device_info['default_samplerate'])  # Convert to int

    if args.filename is None:
        args.filename = tempfile.mktemp(prefix='delme_rec_unlimited_', suffix='.wav', dir='')

    print(f"🎤 Recording audio at {args.samplerate} Hz, saving to {args.filename}")

    # Open file before recording
    with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
                      channels=args.channels, subtype=args.subtype) as file:
        with sd.InputStream(samplerate=args.samplerate, device=args.device,
                            channels=args.channels, callback=callback):
            print('#' * 80)
            print('🎙️ Press Ctrl+C to stop recording')
            print('#' * 80)
            while True:
                file.write(q.get())

except KeyboardInterrupt:
    print(f'\n✅ Recording finished: {args.filename}')

    # Apply noise reduction after recording
    denoised_filename = args.filename.replace(".wav", "_denoised.wav")
    print(f"🔄 Denoising file: {args.filename} -> {denoised_filename}")

    # Load the recorded file
    with sf.SoundFile(args.filename, 'r') as file:
        audio_data = file.read(dtype=np.float32)  # Read as float32
        samplerate = file.samplerate
        channels = file.channels

    # Ensure the audio is in the correct format
    if channels == 1:
        audio_data = np.expand_dims(audio_data, axis=1)  # Make it 2D (N, 1)

    # Initialize RNNoise
    denoiser = RNNoise(sample_rate=samplerate)

    # Process frame by frame
    denoised_audio = np.array([denoiser.process_frame(audio_data[:, ch], last=False) for ch in range(audio_data.shape[1])]).T

    # Save the denoised file
    with sf.SoundFile(denoised_filename, 'w', samplerate=samplerate,
                      channels=channels, subtype='PCM_16') as file:
        file.write(denoised_audio)

    print(f"✅ Denoised audio saved as: {denoised_filename}")

    parser.exit(0)

except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
