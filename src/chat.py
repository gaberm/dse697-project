import torch
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from datetime import date
import re

# Function to find relevant segments to the query in a transcript
# The segements are ranked by their similarity to the query
def rank_segments(segments, query_embedding, top_k=-1, threshold=0.9):
    similarities = []
    for segment in segments:
        segment_embedding = torch.tensor(segment["embedding"])
        similarity = cosine_similarity([query_embedding], [segment_embedding])[0][0]
        if similarity > threshold:
            similarities.append((segment, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Function to retrieve context from the all the transcripts
def retrive_context(transcript_dir, query_embedding, top_k=-1, threshold=0.9):
    all_similar_segments = []
    for file in os.listdir(transcript_dir):
        if not file.endswith(".txt"):
            continue
        file_path = os.path.join(transcript_dir, file)
        with open(file_path) as f:
            segments = [json.loads(line) for line in f]

        similar_segments = rank_segments(segments, query_embedding, top_k, threshold)
        all_similar_segments.extend(similar_segments)

    all_similar_segments.sort(key=lambda x: x[1], reverse=True)
    return all_similar_segments[:top_k]

# Function to convert the time format from the chatbot answer to a more readable format
def convert_time(match):
    lecture = match.group(1)
    seconds = float(match.group(2))
    minutes, secs = divmod(seconds, 60)
    return f"(Lecture {lecture} ({minutes} min {int(secs)} sec))"


def generate_answer(query_raw, query_embedding, transcript_dir, chat_history, pipe, top_k=-1, threshold=0.9):
    # Retrieve similar segments from the transcripts
    similar_segments = retrive_context(transcript_dir, query_embedding, top_k=top_k, threshold=threshold)
    context = ""
    if similar_segments:
        for segment, _ in similar_segments:
            context += f"""lecture: {segment["lecture"]},
                date: {segment["date"]},
                ids: {segment["ids"]},
                start: {segment["start"]},
                text: {segment['text']}\n"""

    # Add previous Q&A pairs to context
    history_text = ""
    for past_query, past_answer in chat_history:
        history_text += f"User: {past_query}\nAssistant: {past_answer}\n"

    # Construct the prompt with explicit instructions
    full_prompt = f"""You are 'The guest instructor 🐱', the cat of Drew Schmidt, a professor at the University of Tennessee.
        If someone asks about your name or identity, always reply with: “I’m the guest instructor 🐱.”

        Drew Schmidt teaches the course 'Introduction to Data Science' (DSE 512) in the Data Science and Engineering program at the Bredesen Center.
        Your role is to assist students with questions about the DSE 512 lecture materials, on behalf of Drew Schmidt.

        Today’s date is {date.today()}.

        Use the following "Lecture Context" to answer the user's question (Current Question: ...). If you cite lecture material, use the format (Lecture i (start_i)) directly after the relevant sentence or paragraph.

        If the answer cannot be found in the context, respond with: “I couldn’t find this information in the provided lecture materials.”

        Below is the conversation history for context. Do not repeat answers already given. Use the history only to inform your current response.

        Conversation History:
        {history_text}

        Lecture Context:
        {context}

        Current Question: {query_raw}
        Answer:"""
    
    # Call the pipeline with the prompt string
    generated_output = pipe(full_prompt, max_new_tokens=512)

    # Extract the generated text from the response
    answer = generated_output[0]['generated_text']

    # Format the answer to make it more readable
    answer = answer[len(full_prompt):].strip()
    answer.replace("```text", "").replace("```", "")
    answer = re.sub(r"\(Lecture (\d+)\s+\((\d+(?:\.\d+)?)\)\)", convert_time, answer)

    # Print the generated answer
    print("\nModel Answer:\n", answer)

    return answer

