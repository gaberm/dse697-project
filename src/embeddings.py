from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def generate_file_embedding(file):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    segments = []
    with open(file) as f:
        for line in f:
            segment = json.loads(line)
            segments.append(segment)

    for idx, segment in enumerate(segments):
        segments[idx]["embedding"] = model.encode(segment["text"], convert_to_tensor=True).tolist()

    with open(file, "w") as f:
        for segment in segments:
            json.dump(segment, f)
            f.write("\n")
