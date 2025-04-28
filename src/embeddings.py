from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
from sklearn.metrics.pairwise import cosine_similarity


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def encode_text(text, tokenizer, model):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].tolist()


def retrieve_similar_segments(segments, query_embedding, top_k=-1, threshold=0.9):
    similarities = []
    for segment in segments:
        segment_embedding = torch.tensor(segment["embedding"])
        similarity = cosine_similarity([query_embedding], [segment_embedding])[0][0]
        if similarity > threshold:
            similarities.append((segment, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def generate_file_embedding(file):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    segments = []
    with open(file) as f:
        for line in f:
            segment = json.loads(line)
            segments.append(segment)

    for idx, segment in enumerate(segments):
        segments[idx]["embedding"] = encode_text(segment["text"], tokenizer, model)

    with open(file, "w") as f:
        for segment in segments:
            json.dump(segment, f)
            f.write("\n")
    print(f"Generated embeddings for {file}")
