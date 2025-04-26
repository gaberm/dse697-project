import whisper_timestamped as whisper
import os
import json
from tqdm import tqdm
from moviepy import VideoFileClip

def get_mp4_files(directory):
    mp4_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.mp4'):
            mp4_files.append(os.path.join(directory, file))
    return mp4_files


def get_mp3_files(directory):
    mp3_files = []
    for file in os.listdir(directory):
        if file.lower().endswith('.mp3'):
            mp3_files.append(os.path.join(directory, file))
    return mp3_files


def main():
    video_dir = "data"
    audio_dir = "data"
    output_dir = "transcripts"
    model_size = "tiny"
    os.makedirs(output_dir, exist_ok=True)

    mp4_files = get_mp4_files(video_dir)
    if not mp4_files:
        print("No MP4 files found in the directory.")
    
    # Extract audio from each .mp4 files
    for mp4_file in tqdm(mp4_files, desc="Extracting audio", unit="file"):
        video_clip = VideoFileClip(mp4_file)
        audio_clip = video_clip.audio
        file_name = os.path.join(audio_dir, f"{mp4_file}.mp3".replace('.mp4', ''))
        audio_clip.write_audiofile(mp4_file.replace('.mp4', '.mp3'))
        audio_clip.close()
        video_clip.close()
    print("Audio extraction successful!")

    # Load audio file path
    mp3_files = get_mp3_files(audio_dir)
    if not mp3_files:
        print("No MP3 files found in the directory.")
        pass

    model = whisper.load_model(model_size, device="cpu")
    for mp3_file in tqdm(mp3_files, desc="Transcribing", unit="file"):
        audio = whisper.load_audio(mp3_file)
        transcript = whisper.transcribe(model, audio, language="en")
        file_name = f"{output_dir}/{os.path.basename(mp3_file)}.json"
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

    


