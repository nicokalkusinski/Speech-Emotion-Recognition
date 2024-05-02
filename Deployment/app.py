import json
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import os
import pyaudio
import wave
import threading
import pickle
import opensmile
import audiofile
import tensorflow as tf
import os
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
import numpy as np
import joblib

app = Flask(__name__)
socketio = SocketIO(app)

# Global variables
RECORDING = False
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 1
ITERATION = 1
RECORD_FOLDER = 'audio'
WAVE_OUTPUT_FILENAME = f"output_{ITERATION}.wav"
PREDICTION = 'NONE'
PREVIOUS_PREDS = []

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def process_audio(file_path):
    if os.path.exists(file_path):
        audio_file = open(file_path, 'rb')
        output_text = prediction(audio_file)
        audio_file.close()
        return output_text
    else:
        return "No audio file captured."

def get_features(file_path):
  signal, sampling_rate = audiofile.read(file_path, duration=4, always_2d=True)
  features = smile.process_signal(signal, sampling_rate)
  return features

def load_cnn():
    #load the model
    model = tf.keras.models.load_model("model/model_92.keras")

    with open('model/encoder_final_92.pickle', 'rb') as f:
        encoder = pickle.load(f)

    with open('model/scaler_final_92.pickle', 'rb') as f:
        scaler = pickle.load(f)
    
    print("Model, encoder and scaler loaded successfully.")
    return [model, encoder, scaler]


def load_disgust():
    #load the model
    with open('model/svm_disgust_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('model/svm_disgust_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    print("Disgust model and scaler loaded successfully.")
    return [model, scaler]

def load_fear():
    #load the model
    with open('model/svm_fear_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('model/svm_fear_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    print("Fear model and scaler loaded successfully.")
    return [model, scaler]

def load_surprise():
    #load the model
    with open('model/svm_surprise_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('model/svm_surprise_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    print("Surprise model and scaler loaded successfully.")
    return [model, scaler]

def pred_disgust(data):
    _data = scaler_disgust.transform(data)
    prediction = model_disgust.predict(_data)
    return prediction[0] == 1

def pred_fear(data):
    _data = scaler_fear.transform(data)
    prediction = model_fear.predict(_data)
    return prediction[0] == 1

def pred_surprise(data):
    _data = scaler_surprise.transform(data)
    prediction = model_surprise.predict(_data)
    return prediction[0] == 1

def pred_cnn(data):
    #scale the data using CNN scaler
    _data = scaler_cnn.transform(data)

    # Predict emotion using the CNN model
    predictions = model_cnn.predict(_data).reshape(1, -1)
    y_pred = encoder_cnn.inverse_transform(predictions)

    return y_pred[0][0]

def prediction(audio_file):
    global PREDICTION
    global PREVIOUS_PREDS

    # Get features from the saved audio file
    res = get_features(audio_file)

    #Predict using FEAR, DISGUST and SURPRISE SVM models first
    if pred_disgust(res):
        PREDICTION = "Disgust"
        PREVIOUS_PREDS.append(PREDICTION)
        return PREDICTION, PREVIOUS_PREDS
    
    if pred_fear(res):
        PREDICTION = "Fear"
        PREVIOUS_PREDS.append(PREDICTION)
        return PREDICTION, PREVIOUS_PREDS

    if pred_surprise(res):
        PREDICTION = "Surprise"
        PREVIOUS_PREDS.append(PREDICTION)
        return PREDICTION, PREVIOUS_PREDS

    # IF 3 SVM MODELS PREDICTED DIFFERENT EMOTION
    PREDICTION = pred_cnn(res)
    PREVIOUS_PREDS.append(PREDICTION)

    # Emit prediction event to update HTML
    socketio.emit('update_prediction', {'prediction': PREDICTION, 'previous_preds': PREVIOUS_PREDS})

    return PREDICTION, PREVIOUS_PREDS

def remove_file(file_path):
    os.remove(file_path)

def record_audio():
    global RECORDING
    global WAVE_OUTPUT_FILENAME
    global ITERATION

    audio, stream, frames = setup_recording()

    def cut():
        stream.stop_stream()
        stream.close()
        audio.terminate()

        wf = wave.open(f"{RECORD_FOLDER}/{WAVE_OUTPUT_FILENAME}", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    print(f"Iteration {ITERATION}: Recording started...")

    # while the recording is on, cut the audio in 3s clips
    while RECORDING:
        data = stream.read(CHUNK)
        frames.append(data)

        if len(frames) > int(RATE / CHUNK * RECORD_SECONDS):
            print(f"Iteration {ITERATION}: Recording stopped...")
            cut()
            handle_prediction(f"{RECORD_FOLDER}/{WAVE_OUTPUT_FILENAME}")

            ITERATION += 1
            if RECORDING:
                WAVE_OUTPUT_FILENAME = f"output_{ITERATION}.wav"
                audio, stream, frames = setup_recording()
                print(f"Iteration {ITERATION}: Recording restarted...")

    # when user manually stops recording, cut the audio
    print(f"Iteration {ITERATION}: Recording stopped by the user...")
    cut()
    handle_prediction(f"{RECORD_FOLDER}/{WAVE_OUTPUT_FILENAME}")
    reset()

def setup_recording():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []

    return audio, stream, frames

def handle_prediction(file_path):
    global PREDICTION
    global PREVIOUS_PREDS

    PREDICTION, PREVIOUS_PREDS = prediction(file_path)
    remove_file(file_path)

    # return render_template('index.html', prediction=PREDICTION, recording=RECORDING, PREVIOUS_PREDS=PREVIOUS_PREDS)
    print("Prediction: ", PREDICTION)

def reset():
    global ITERATION
    global PREDICTION
    ITERATION = 1
    PREDICTION = 'NONE'

model_cnn, encoder_cnn, scaler_cnn = load_cnn()
model_disgust, scaler_disgust = load_disgust()
model_fear, scaler_fear = load_fear()
model_surprise, scaler_surprise = load_surprise()

@app.route('/')
def index():
    global PREDICTION
    global RECORDING
    global PREVIOUS_PREDS
    
    return render_template('index.html', prediction=PREDICTION, recording=RECORDING, previous_preds=PREVIOUS_PREDS)

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global RECORDING
    global WAVE_OUTPUT_FILENAME
    global ITERATION

    if not RECORDING:
        RECORDING = True
        threading.Thread(target=record_audio).start()

    return render_template('index.html', prediction=PREDICTION, recording=RECORDING, previous_preds=PREVIOUS_PREDS)

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global RECORDING

    RECORDING = False

    print("*Start recording")

    return render_template('index.html', prediction=PREDICTION, recording=RECORDING, previous_preds=PREVIOUS_PREDS)

@socketio.on('connect')
def handle_connect():
    emit('update_prediction', {'prediction': PREDICTION, 'previous_preds': PREVIOUS_PREDS})

if __name__ == '__main__':
    socketio.run(app, debug=True)
