from transformers import AutoTokenizer, AutoModel
from src.embedding import mean_pooling, encode_text, retrieve_similar_segments

def chat(query):
    model = AutoModel.from_pretrained('models/all-MiniLM-L6-v2')
    