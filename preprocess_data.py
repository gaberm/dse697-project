from src.transcription import transcripe
from src.embeddings import generate_file_embedding
from tqdm import tqdm
import os

def main():
    video_files = ["data/GMT20250121-210713_Recording_1920x1080.mp4"]
    transcript_dir = "transcripts"
    model_size = "tiny"
    chunk_size = 30 # seconds
    overlap = 10 # seconds
    lectures = [1]
    dates = ["2025-01-21"]

    for video_file, lecture, date in tqdm(zip(video_files, lectures, dates), total=len(video_files), desc="Transcribing video files"):
        transcripe(video_file, transcript_dir, transcript_dir, model_size, chunk_size, overlap, lecture, date)

    for file in tqdm(os.listdir(f"{transcript_dir}/preprocess"), desc="Generating embeddings"):
        if file.endswith(".txt"):
            file_path = os.path.join(f"{transcript_dir}/preprocess", file)
            generate_file_embedding(file_path)

if __name__ == "__main__":
    main()
