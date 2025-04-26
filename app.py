import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_socketio import SocketIO, emit
import librosa
import soundfile as sf
import os
import time
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import uuid
import shutil

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')  # added async_mode=eventlet

app.secret_key = 'your_secret_key'

# Folder setup
if not os.path.exists('static/input'):
    os.makedirs('static/input')
if not os.path.exists('static/output'):
    os.makedirs('static/output')

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def add_reverb(audio_segment):
    reverb_audio = audio_segment + 5  # Adjust reverb here
    return reverb_audio

@app.route('/', methods=['GET', 'POST'])
def index():
    output_file = None
    input_file = None

    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        pitch_shift = request.form.get('pitch_shift', 4)
        time_stretch_rate = request.form.get('time_stretch', 0.15)
        lowcut = request.form.get('lowcut', 300)
        highcut = request.form.get('highcut', 3000)

        try:
            pitch_shift = int(pitch_shift)
            time_stretch_rate = float(time_stretch_rate)
            lowcut = int(lowcut)
            highcut = int(highcut)
        except ValueError:
            flash('Invalid input parameters. Please use numeric values.', 'error')
            return render_template('index.html', output_file=output_file, input_file=input_file)

        if uploaded_file and uploaded_file.filename.endswith('.mp3'):
            unique_id = str(uuid.uuid4())
            filename = f"{unique_id}_{uploaded_file.filename}"
            input_file = os.path.join('input', filename).replace('\\', '/')
            mp3_path = os.path.join('static', input_file).replace('\\', '/')

            # Save original file
            uploaded_file.save(mp3_path)
            flash('File uploaded successfully!', 'success')
            socketio.emit('progress', {'progress': 5})

            try:
                y, sr = librosa.load(mp3_path, sr=None)
                socketio.emit('progress', {'progress': 15})

                # Pitch shifting
                y_pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
                socketio.emit('progress', {'progress': 30})

                # Harmonic and percussive separation
                harmonic, percussive = librosa.effects.hpss(y)
                harmonic_path = os.path.join('static', 'output', f'harmonic_{unique_id}.wav').replace('\\', '/')
                percussive_path = os.path.join('static', 'output', f'percussive_{unique_id}.wav').replace('\\', '/')
                sf.write(harmonic_path, harmonic, sr)
                sf.write(percussive_path, percussive, sr)
                socketio.emit('progress', {'progress': 50})

                # Time stretching
                y_time_stretched = librosa.effects.time_stretch(y, rate=time_stretch_rate)
                socketio.emit('progress', {'progress': 65})

                # Combine pitch-shifted and time-stretched audio
                min_length = min(len(y_pitch_shifted), len(y_time_stretched))
                combined_audio = y_pitch_shifted[:min_length] + y_time_stretched[:min_length]
                socketio.emit('progress', {'progress': 75})

                # Apply bandpass filter
                y_filtered = bandpass_filter(combined_audio, lowcut=lowcut, highcut=highcut, fs=sr)
                filtered_audio_path = os.path.join('static', 'output', f'filtered_audio_{unique_id}.wav').replace('\\', '/')
                sf.write(filtered_audio_path, y_filtered, sr)
                socketio.emit('progress', {'progress': 85})

                # Add reverb
                audio_segment = AudioSegment.from_wav(filtered_audio_path)
                y_reverb = add_reverb(audio_segment)
                reverb_audio_path = os.path.join('static', 'output', f'reverb_audio_{unique_id}.wav').replace('\\', '/')
                y_reverb.export(reverb_audio_path, format="wav")
                socketio.emit('progress', {'progress': 90})

                # Adjust volume
                volume_change_db = 5
                y_adjusted_volume = y_reverb + volume_change_db
                adjusted_volume_audio_path = os.path.join('static', 'output', f'adjusted_volume_audio_{unique_id}.wav').replace('\\', '/')
                y_adjusted_volume.export(adjusted_volume_audio_path, format="wav")
                socketio.emit('progress', {'progress': 95})

                # Convert to MP3
                final_audio_path = os.path.join('static', 'output', f'transformed_audio_{unique_id}.mp3').replace('\\', '/')
                y_final_audio = AudioSegment.from_wav(adjusted_volume_audio_path)
                y_final_audio.export(final_audio_path, format="mp3")
                socketio.emit('progress', {'progress': 100})

                output_file = f'output/transformed_audio_{unique_id}.mp3'

            except Exception as e:
                flash(f'Error processing audio: {str(e)}', 'error')
                return render_template('index.html', output_file=None, input_file=None)

        else:
            flash('Please upload a valid MP3 file.', 'error')

    return render_template('index.html', output_file=output_file, input_file=input_file)

if __name__ == '__main__':
    socketio.run(app, debug=True)
