from preprocessing import GENRE_DICT, folder
from input import record_audio, extract_audio_features, save_audio_to_file
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import librosa
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

df = pd.read_csv('extracted_features.csv')
df_trimmed = df.iloc[:, :154]
df_trimmed.to_csv('trimmed_features.csv', index=False)

duration = 5  
fs = 10000  
output_filename = "recorded_audio.wav"

recorded_audio = record_audio(duration, fs)
save_audio_to_file(output_filename, recorded_audio, fs)

y, sr = librosa.load(output_filename, sr=None)
audio_features_list = extract_audio_features(y, sr)

df = pd.read_csv('trimmed_features.csv')

genre = []
for i in os.listdir(folder):
    genre_path = os.path.join(folder, i)
    if os.path.isdir(genre_path):
        for j in os.listdir(genre_path):
            if j.endswith('.wav') or j.endswith('.mp3'):
                file_path = os.path.join(genre_path, j)
                genre.append((file_path, i))

df2 = pd.DataFrame(genre, columns=['file_path', 'genre'])
result = pd.concat([df, df2], axis=1)
df_cleaned = result.dropna()

def add_noise(data, noise_factor=0.05):
    noise = np.random.normal(0, 1, data.shape)
    return data + noise_factor * noise

x = df_cleaned.iloc[:, :-2]
y = df_cleaned.iloc[:, -1]
audio_features_array = np.array(audio_features_list).reshape(1, -1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

label_encoder = LabelEncoder()
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
audio_features_scaled = scaler.transform(audio_features_array)
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

x_train_scaled_noisy = add_noise(x_train_scaled)

input_shape = (14, 11, 1)

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.3),
    
    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.4),
    
    Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.5),
    
    Flatten(),
    Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(np.unique(y_train_encoded)), activation='softmax')
])

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

x_train_reshaped = x_train_scaled_noisy.reshape(-1, 14, 11, 1)
x_test_reshaped = x_test_scaled.reshape(-1, 14, 11, 1)
audio_features_reshaped = audio_features_scaled.reshape(1, 14, 11, 1)

history = model.fit(
    x_train_reshaped, 
    y_train_encoded,
    validation_data=(x_test_reshaped, y_test_encoded),
    epochs=80,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

test_loss, test_accuracy = model.evaluate(x_test_reshaped, y_test_encoded)
print(f"Test accuracy: {test_accuracy:.2f}")

predicted_label_encoded = model.predict(audio_features_reshaped)
predicted_label_index = np.argmax(predicted_label_encoded)
predicted_genre = label_encoder.inverse_transform([predicted_label_index])
print(f"Predicted genre: {predicted_genre[0]}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()