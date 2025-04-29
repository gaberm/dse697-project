from flask import Flask, request, render_template, redirect, url_for, send_file
import os
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = '/gpfs/wolf2/olcf/trn040/scratch/mgaber/dse697-project/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Upload and Stream Video</title>
    <h1>Upload a Video File</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="video">
        <input type="submit" value="Upload">
    </form>
    <h2>Stream a Saved Video</h2>
    <video id="videoPlayer" width="640" height="360" controls>
        <source src="/stream" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <script>
        // Start the video at 30 seconds
        const video = document.getElementById('videoPlayer');
        video.addEventListener('loadedmetadata', () => {
            video.currentTime = 30;
        });
    </script>
    '''

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No file part", 400

    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Save the uploaded file as the default video to stream
    default_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'GMT20250121-210713_Recording_1920x1080.mp4')
    os.rename(file_path, default_video_path)

    return f"Video {file.filename} uploaded and set for streaming!"

@app.route('/stream')
def stream_video():
    # Path to the default video to stream
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'GMT20250121-210713_Recording_1920x1080.mp4')
    if not os.path.exists(video_path):
        return "No video available for streaming", 404
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)