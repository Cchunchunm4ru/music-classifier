import os
import librosa
import numpy as np
import pandas as pd

folder = r"Data\genres_original"
GENRE_DICT = {
    0: "blues",
    1: "hiphop",
    2: "disco",
    3: "rock",
    4: "jazz",
    5: "reggae",
    6: "metal",
    7: "classical",
    8: "pop",
    9: "country"
}

def extract_features(file_name):
    try:
        y, sr = librosa.load(file_name, sr=None)
        
        features = np.array([])
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        features = np.hstack((features, mfccs))
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        features = np.hstack((features, chroma))
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        features = np.hstack((features, mel))
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
        features = np.hstack((features, contrast))
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        features = np.hstack((features, tonnetz))
        return features,mfccs,chroma,mel,tonnetz,contrast
    except Exception as e:
        print(f"Error extracting features from {file_name}: {str(e)}")
        return None
# feat_list = []
# labels = []
# for root, dirs, files in os.walk(folder):
#     for file in files:
#         if file.endswith(".wav") or file.endswith(".mp3"):
#             file_path = os.path.join(root, file)  
#             if os.path.exists(file_path):
#                 try:
#                     print(f"Processing file: {file_path}")  
#                     feature_row = extract_features(file_path)
#                     genre = os.path.basename(root)  
#                     labels.append(genre)
#                     if feature_row is not None:
#                         feat_list.append(feature_row)
#                 except Exception as e:
#                     print(f"Error processing file {file_path}: {str(e)}")
#             else:
#                 print(f"File not found: {file_path}")
# print(f"Extracted features for {len(feat_list)} files.")
# if feat_list:  
#     df = pd.DataFrame(feat_list)
#     df.to_csv('extracted_features.csv', index=False)
#     df = pd.DataFrame(feat_list)
#     df['genre'] = labels
#     print("Features saved to extracted_features.csv")
# else:
#     print("No features extracted. CSV not created.")
