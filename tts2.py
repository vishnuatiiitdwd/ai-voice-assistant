import torch
import soundfile as sf

# Set the device
device = torch.device('cpu')

# Load the TTS model and utils
model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='en',
    speaker='v3_en',
    device=device
)

# Define your input text
text = "Hello! This is a test using Silero Text to Speech."

# Generate audio
audio = model.apply_tts(
    text=text,
    speaker='en_53',  # You can try other speaker IDs
    sample_rate=48000
)

# Save audio to file
sf.write('silero_tts_output.wav', audio, 48000)
print("Speech saved as 'silero_tts_output.wav'")
