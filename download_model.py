from transformers import AutoTokenizer, AutoModel
import os

def main():
    os.makedirs("models/all-MiniLM-L6-v2", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    tokenizer.save_pretrained("models/all-MiniLM-L6-v2")
    model.save_pretrained("models/all-MiniLM-L6-v2")

if __name__ == "__main__":
    main()