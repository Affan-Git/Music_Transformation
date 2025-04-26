from flask import Flask, request, render_template
from pydub import AudioSegment
import os

app = Flask(__name__)

# Create the output directory if it doesn't exist
os.makedirs('static/output', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    output_file = None  # Variable to hold the path of the output MP3 file

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            # Save the uploaded file in the static/input folder
            mp3_path = os.path.join('static', 'input', uploaded_file.filename)
            uploaded_file.save(mp3_path)

            # Process the file (convert it to WAV first)
            output_wav_path = os.path.join('static', 'output', 'output.wav')
            AudioSegment.from_mp3(mp3_path).export(output_wav_path, format='wav')

            # Convert the WAV to MP3 and save it
            output_mp3_path = os.path.join('static', 'output', 'output.mp3')
            AudioSegment.from_wav(output_wav_path).export(output_mp3_path, format='mp3')

            # Set the path for the MP3 file to be used in the frontend
            output_file = 'output/output.mp3'

    # Render the template and pass the output file path to it
    return render_template('index.html', output_file=output_file)

if __name__ == '__main__':
    app.run(debug=True)
