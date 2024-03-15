from flask import Flask, render_template, request
import pickle
import opensmile
import audiofile
import tensorflow as tf
import os
import time
import pyaudio
import wave
# from collections import defaultdict

app = Flask(__name__)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

previous_predictions = {}

capturing = False
start_time = None
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # Adjust this as needed
WAVE_OUTPUT_FILENAME = "temp_audio.wav"

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    print("* Recording audio...")

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Audio recording finished")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def process_audio():
    if os.path.exists(WAVE_OUTPUT_FILENAME):
        audio_file = open(WAVE_OUTPUT_FILENAME, 'rb')
        output_text = prediction(audio_file)
        audio_file.close()
        return output_text
    else:
        return "No audio file captured."


def get_features(file_path):
  signal, sampling_rate = audiofile.read(file_path, duration=4, always_2d=True)
  features = smile.process_signal(signal, sampling_rate)
  return features

def load_body():
    #load the model
    # root_folder = "/Users/nicokalkusinski/Documents/Studies/FYP/Deployment/"
    model = tf.keras.models.load_model("model/model_92.keras")

    with open('model/encoder_final_92.pickle', 'rb') as f:
        encoder = pickle.load(f)

    with open('model/scaler_final_92.pickle', 'rb') as f:
        scaler = pickle.load(f)
    
    print("Model, encoder and scaler loaded successfully.")
    return [model, encoder, scaler]

def prediction(audio_file):
    # Save the uploaded audio file temporarily
    temp_audio_path = "temp_audio.wav"
    audio_file.save(temp_audio_path)

    # Get features from the saved audio file
    res = get_features(temp_audio_path)
    res = scaler.transform(res)

    # Predict emotion using the model
    predictions = model.predict(res).reshape(1, -1)
    y_pred = encoder.inverse_transform(predictions)
    print("Predicted emotion:", y_pred[0][0])

    # Store the prediction with the input file name as key
    filename = audio_file.filename
    previous_predictions[filename] = y_pred[0][0]

    # Remove the temporary audio file
    os.remove(temp_audio_path)

    return "Predicted emotion: " + y_pred[0][0]

model, encoder, scaler = load_body()

@app.route('/', methods=['GET', 'POST'])
def index():
    global capturing, start_time
    output_text = None
    timer = None
    if request.method == 'POST':
        if 'start_capture' in request.form:
            if not capturing:
                capturing = True
                start_time = time.time()  # Start the timer
                record_audio()
                return render_template('index.html', capturing=capturing)
        elif 'stop_capture' in request.form:
            if capturing:
                print("*Processing audio")
                capturing = False
                output_text = process_audio()  # Process the audio file
                timer = round(time.time() - start_time, 2)  # Calculate elapsed time
                return render_template('index.html', output_text=output_text, timer=timer, capturing=capturing)
    return render_template('index.html', output_text=output_text, timer=timer, capturing=capturing)

if __name__ == '__main__':
    app.run(debug=True)