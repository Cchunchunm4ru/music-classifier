import sounddevice as sd
import numpy as np
import wavio
import librosa

# Record audio function
def record_audio(duration, fs):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    print("Recording finished.")
    return recording

# Save audio function
def save_audio_to_file(filename, audio_data, fs):
    audio_data_normalized = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
    wavio.write(filename, audio_data_normalized, fs, sampwidth=2)

# Extract audio features (MFCC, Spectral Contrast, Chromagram, Tonnetz)
# Modify the extract_audio_features function
# Modify the extract_audio_features function
def extract_audio_features(y, sr):
    # Extract 40 MFCCs (adjust based on your training setup)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=140)  # Adjust n_mfcc based on your training data
    mfccs_mean = mfccs.mean(axis=1)

    # Extract spectral contrast with enough bands to contribute to feature count
    fmin = 50  # Set the minimum frequency (consistent with training)
    n_bands = 7  # Adjust number of bands based on the training process
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=fmin, n_bands=n_bands)
    spectral_contrast_mean = spectral_contrast.mean(axis=1)

    # Extract chromagram
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    chromagram_mean = chromagram.mean(axis=1)

    # Extract tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = tonnetz.mean(axis=1)

    # Combine all features into a single list
    features_list = np.concatenate([mfccs_mean, spectral_contrast_mean, chromagram_mean, tonnetz_mean])

    # Ensure that the final feature count is 166
    return features_list[:166].tolist()


# Main function
if __name__ == "__main__":
    duration = 5  # Duration of recording
    fs = 10000  # Sampling frequency
    output_filename = "recorded_audio.wav"

    # Record and save audio
    recorded_audio = record_audio(duration, fs)
    save_audio_to_file(output_filename, recorded_audio, fs)

    # Load the audio file with librosa
    audio_file = output_filename
    y, sr = librosa.load(audio_file, sr=None)  # sr=None keeps the native sampling rate

    # Extract the audio features as a list
    audio_features_list = extract_audio_features(y, sr)
    
    # Output the extracted features as a list
    
