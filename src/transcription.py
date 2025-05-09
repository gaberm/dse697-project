import whisper_timestamped as whisper
import torch
import os
import json
from moviepy import VideoFileClip


def transcripe(mp4_file, audio_dir, transcript_dir, model_size, chunk_size, overlap, lecture, date):
    # Create audio directory in the same directory as the video file
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(f"{transcript_dir}/raw", exist_ok=True)
    os.makedirs(f"{transcript_dir}/processed", exist_ok=True)
    
    # Extract audio from the video file
    video_clip = VideoFileClip(mp4_file)
    audio_clip = video_clip.audio
    file_name = os.path.join(audio_dir, f"lecture-{lecture}-{date}.mp3")
    print(f"🎙️ Extracting audion from {mp4_file}")
    audio_clip.write_audiofile(file_name)
    audio_clip.close()
    video_clip.close()

    # Load the model and transcribe the audio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emoji = "🐇" if device == "cuda" else "🐢"
    print(f"{emoji} Cuda available: {torch.cuda.is_available()}")
    
    model = whisper.load_model(model_size, device=device)
    audio = whisper.load_audio(file_name)

    print(f"📝 Transcribing {file_name}...")
    transcript = whisper.transcribe(model, audio, language="en")

    # Save the raw transcript
    raw_file_name = f"{transcript_dir}/raw/{os.path.basename(file_name).replace(".mp3", "")}.json"
    with open(raw_file_name, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    # Since the automatic segments are very short, we will combine them into larger chunks, 
    text_chunks = []
    i = 0
    while i < len(transcript["segments"]):
        start_time = transcript["segments"][i]["start"]
        end_time = start_time + chunk_size

        chunk_text = []
        ids = []
        j = i
        while j < len(transcript["segments"]) and transcript["segments"][j]["end"] <= end_time:
            ids.append(transcript["segments"][j]["id"])
            chunk_text.append(transcript["segments"][j]["text"])
            j += 1

        text_chunks.append({
            "lecture": lecture,
            "date": date,
            "start": start_time,
            "end": end_time,
            "ids": ids,
            "text": " ".join(chunk_text),
        })

        next_start_time = start_time + (chunk_size - overlap)
        
        while i < len(transcript["segments"]) and transcript["segments"][i]["start"] < next_start_time:
            i += 1

    # Save the processed transcript
    transcript_file_name = f"{transcript_dir}/processed/{os.path.basename(file_name).replace(".mp3", "")}.txt"
    with open(transcript_file_name, "w") as f:
        for segment in text_chunks:
            json.dump(segment, f)
            f.write("\n")
