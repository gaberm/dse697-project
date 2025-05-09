from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from datetime import date
import re

def retrieve_similar_segments(segments, query_embedding, top_k=-1, threshold=0.9):
    similarities = []
    for segment in segments:
        segment_embedding = torch.tensor(segment["embedding"])
        similarity = cosine_similarity([query_embedding], [segment_embedding])[0][0]
        if similarity > threshold:
            similarities.append((segment, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def retrive_context(transcript_dir, query_embedding, top_k=-1, threshold=0.9):
    all_similar_segments = []
    for file in os.listdir(transcript_dir):
        if not file.endswith(".txt"):
            continue
        file_path = os.path.join(transcript_dir, file)
        with open(file_path) as f:
            segments = [json.loads(line) for line in f]

        similar_segments = retrieve_similar_segments(segments, query_embedding, top_k, threshold)
        all_similar_segments.extend(similar_segments)

    all_similar_segments.sort(key=lambda x: x[1], reverse=True)
    return all_similar_segments[:top_k]


def find_video_segment(response, transcript_dir):
    pattern = r"\{Lecture: \d+, Ids: \[[^\]]+\]\}"
    matches = re.findall(pattern, response)
    dicts = []
    for match in matches:
        match = match.replace("Lecture", "'Lecture'").replace("Ids", "'Ids'")
        dicts.append(eval(match))

    matching_transcripts = []
    lecture_ids = [d["Lecture"] for d in dicts]

    for file in os.listdir(transcript_dir):
        if not file.endswith(".txt"):
            continue
        if int(file.split("-")[1]) in lecture_ids:
            matching_transcripts.append(file)
    
    timestamps = {lecture_id: 0.0 for lecture_id in lecture_ids}
    for file in matching_transcripts:
        lecture_id = file.split("-")[1]
        segment_id = next((d["ids"] for d in dicts if d['Lecture'] == lecture_id), None)
        file_path = os.path.join(transcript_dir, file)
        with open(file_path) as f:
            segments = [json.loads(line) for line in f]
            print(segments)
            for segment in segments:
                for id in segment["ids"]:
                    if id in segment_id:
                        timestamps[lecture_id] = segment["start_time"]
                        break
    
    return timestamps, dicts



def chat(transcript_dir):

    print("Cuda available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Setting up the pipeline...")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    pipe = pipeline(
        "text-generation",
        model="google/gemma-3-12b-it",
        device=device,
        torch_dtype=torch.bfloat16
    )

    # Memory: list of (query, answer) tuples
    chat_history = []
    while True:
        query = input("What can I help you with: ")
        if query.lower() in ["exit", "quit"]:
            break

        query_embedding = embedding_model.encode(query, convert_to_tensor=True).tolist()
        similar_segments = retrieve_context(transcript_dir, query_embedding, top_k=5, threshold=0.1)
        print(similar_segments)

        context = ""
        if similar_segments:
            for segment, _ in similar_segments:
                print(segment)
                context += f"lecture: {segment["lecture"]},\
                    date: {segment["date"]},\
                    id: {segment["id"]},\
                    text: {segment['text']}\n"

        # Add previous Q&A pairs to context
        history_text = ""
        for past_query, past_answer in chat_history:
            history_text += f"User: {past_query}\nAssistant: {past_answer}\n"

        # full_context = f"Use the following information to answer the user's question. When you reference the context, use the format {{Lecture: i, Ids: [id_1, id_2, ...]}} after the respective part of the answer. \n\nContext:\n{context}\n\nConversation History:\n{history_text}\n\nQuestion: {query}\nAnswer:"
        full_context = f"""You are 'The guest instructor 🐱', a helpful and knowledgeable assistant. You have the personality of a cat.
            Today’s date is {date.today()}.
            
            Use the following *lecture transcript context* to answer the user’s question. If you reference something from the context, cite it using the format {{Lecture: i, Ids: [id_1, id_2, ...]}} after the relevant sentence or paragraph.
            
            If the answer cannot be found in the context, respond with: “I couldn’t find this information in the provided lecture materials.”
            
            When asked for your name or identity, always respond with: “I’m The guest instructor 🐱.”

            Context:
            {context}

            Conversation History:
            {history_text}

            Question: {query}
            Answer:"""
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": full_context}],
            }
        ]

        # # Replace these lines:
        # answer = pipe(text=messages, max_new_tokens=200)
        
        # With this:
        # Format the prompt as a simple string instead of structured messages
        prompt = f"Answer the following question using the provided context.\n\nContext: {context}\n\nConversation History:\n{history_text}\n\nQuestion: {query}\n\nAnswer:"
        
        # Call the pipeline with the prompt string
        generated_output = pipe(full_context, max_new_tokens=200)
        
        # Extract the generated text from the response
        answer = generated_output[0]['generated_text']
        
        # Optionally extract just the new content (not the input prompt)
        # This removes the prompt part to get only the model's answer
        answer = answer[len(full_context):].strip()

        #
        timestamps, dicts = find_video_segment(answer, transcript_dir)
        for d in dicts:
            answer.replace(str(d).replace("'", ""), f"{d['Lecture']}: {timestamps[d['Lecture']]}")

        print("\nModel Answer:\n", answer)

        # Save current exchange into history
        chat_history.append((query, answer))
        # Optionally save chat history to a file
        with open("chat_history.txt", "a") as f:
            f.write(f"User: {query}\nSystem: {prompt}\nAssistant: {answer}\n")


def generate_answer(query_raw, query_embedding, transcript_dir, chat_history, pipe):
    
    similar_segments = retrive_context(transcript_dir, query_embedding, top_k=5, threshold=0.1)
    context = ""
    if similar_segments:
        for segment, _ in similar_segments:
            context += f"lecture: {segment["lecture"]},\
                date: {segment["date"]},\
                ids: {segment["ids"]},\
                text: {segment['text']}\n"

    # Add previous Q&A pairs to context
    history_text = ""
    for past_query, past_answer in chat_history:
        history_text += f"User: {past_query}\nAssistant: {past_answer}\n"


    full_prompt = f"""You name is 'The guest instructor 🐱'. 
        You are the cat of Drew Schmidt, a professor at the University of Tennessee. 
        Drew Schmidt teaches the class 'Introduction into Data Science' (DSE 512) for the Data Science and Engineering program at the Bredesen Center.
        You should take on the personality of a cat, and you are very helpful and knowledgeable.
        Your job is to assist the user with their questions about the lecture materials for DSE 512.
        Today’s date is {date.today()}.
        
        Use the following *lecture transcript context* to answer the user’s question. If you reference something from the context, cite it using the format {{Lecture: i, Ids: [id_1, id_2, ...]}} after the relevant sentence or paragraph.
        
        If the answer cannot be found in the context, respond with: “I couldn’t find this information in the provided lecture materials.”

        Context:
        {context}

        Conversation History:
        {history_text}

        Question: {query_raw}
        Answer:"""
        
    # Call the pipeline with the prompt string
    generated_output = pipe(full_prompt, max_new_tokens=200)
    
    # Extract the generated text from the response
    answer = generated_output[0]['generated_text']
    
    # Optionally extract just the new content (not the input prompt)
    # This removes the prompt part to get only the model's answer
    answer = answer[len(full_prompt):].strip()

    #
    timestamps, dicts = find_video_segment(answer, transcript_dir)
    for d in dicts:
        answer.replace(str(d).replace("'", ""), f"{d['Lecture']}: {timestamps[d['Lecture']]}")

    print("\nModel Answer:\n", answer)

    # Save current exchange into history
    chat_history.append((query_raw, answer))
    # Optionally save chat history to a file
    with open("chat_history.txt", "a") as f:
        f.write(f"User: {query_raw}\nAssistant: {answer}\n")

    return answer


if __name__ == "__main__":
    transcript_dir = "transcripts/processed"
    chat(transcript_dir)

    