from embedding import transcripe
from src.embedding import generate_file_embedding, transcripe
from tqdm import tqdm
import os

def main():
    video_dir = "/gpfs/wolf2/olcf/trn040/proj-shared/mgaber_6i0/videos/"
    video_files = [f"{video_dir}lecture-{lecture}.mp4" for lecture in range(1, 24)]
    audio_dir = "/gpfs/wolf2/olcf/trn040/proj-shared/mgaber_6i0/audio/"
    transcript_dir = "/gpfs/wolf2/olcf/trn040/proj-shared/mgaber_6i0/transcripts/"
    model_size = "tiny"
    chunk_size = 30 # seconds
    overlap = 10 # seconds
    lectures = list(range(1, 24))
    dates = ["2025-01-21", "2025-01-23", "2025-01-28", "2025-01-30", "2025-02-04", "2025-02-06", "2025-02-11", "2025-02-13", "2025-02-18", "2025-02-25", "2025-02-27", "2025-03-04", "2025-03-06", "2025-03-11", "2025-03-13", "2025-03-25", "2025-03-27", "2025-04-01", "2025-04-03", "2025-04-08", "2025-04-15", "2025-04-22", "2025-04-24"]

    for video_file, lecture, date in tqdm(zip(video_files, lectures, dates), total=len(video_files), desc="Transcribing video files"):
        transcripe(video_file, audio_dir, transcript_dir, model_size, chunk_size, overlap, lecture, date)

    for file in tqdm(os.listdir(f"{transcript_dir}processed"), desc="🧠 Generating embeddings"):
        if file.endswith(".txt"):
            file_path = os.path.join(f"{transcript_dir}/processed", file)
            generate_file_embedding(file_path)

if __name__ == "__main__":
    main()
