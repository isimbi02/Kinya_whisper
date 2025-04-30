import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="urankunda.wav", duration=3, fs=44100):
    print("🎤 Recording for", duration, "seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("✅ Recording saved as:", filename)

record_audio()