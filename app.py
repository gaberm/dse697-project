from flask import Flask, request, render_template, jsonify
from src.chat import generate_answer
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

app = Flask(__name__)

chat_history = []

# Settings for the retrieval process
top_k = 5
threshold = 0.1
transcript_dir = "/gpfs/wolf2/olcf/trn040/proj-shared/mgaber_6i0/transcripts/"

# Loading the embedding and chatbot models
print("🚀 Initializing models... Please wait.")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
device = "cuda" if torch.cuda.is_available() else "cpu"
text_generation_pipeline = pipeline(
    "text-generation",
    model="google/gemma-3-27b-it",
    device=device,
    torch_dtype=torch.bfloat16 if device == "cuda" else None
)
print("🎉 Models initialized successfully!")

@app.route('/')
def index():
    return r'''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chat with The Guest Instructor 🐱</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            body {
                background-color: #f5f5f5;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
            }
            .chat-container {
                width: 100%;
                max-width: 800px;
                height: 90vh;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .chat-header {
                background-color: #f08034;
                color: white;
                padding: 15px;
                font-size: 24px;
                font-weight: bold;
                text-align: center;
            }
            .chat-content {
                flex: 1;
                overflow-y: auto;
                padding: 10px;
            }
            .chat-history {
                display: flex;
                flex-direction: column;
                gap: 8px; /* Reduced spacing between messages */
            }
            .chat-message {
                max-width: 80%;
                padding: 10px 15px;
                border-radius: 15px;
                margin-bottom: 4px; /* Further reduced space between messages */
                word-wrap: break-word;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }
            .chat-message p {
                margin: 0;
            }
            .chat-message.user {
                align-self: flex-end;
                background-color: #e3f2fd;
                margin-left: auto;
                border-bottom-right-radius: 5px;
            }
            .chat-message.assistant {
                align-self: flex-start;
                background-color: #f1f1f1;
                border-bottom-left-radius: 5px;
                max-width: 70%; /* Narrower assistant messages */
            }
            .chat-input {
                display: flex;
                align-items: center;
                padding: 15px;
                background-color: #f9f9f9;
                border-top: 1px solid #ddd;
                position: relative;
            }
            #query {
                flex: 1;
                padding: 12px 15px;
                border: 1px solid #ddd;
                border-radius: 25px;
                outline: none;
                resize: none;
                font-size: 16px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            #sendButton {
                background-color: #f08034;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 10px 20px;
                margin-left: 10px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            #sendButton:hover {
                background-color: #e06814;
            }
            .loading-spinner {
                position: absolute;
                right: 110px; /* Adjusted to align with send button */
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #f08034;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">Chat with The Guest Instructor 🐱</div>
            <div class="chat-content">
                <div id="chatHistory" class="chat-history"></div>
            </div>
            <form id="chatForm" autocomplete="off" onsubmit="event.preventDefault(); sendButton.click();">
                <div class="chat-input">
                    <textarea id="query" rows="1" placeholder="Type your message here..." autocomplete="off"></textarea>
                    <div id="loadingSpinner" class="loading-spinner" style="display: none;"></div>
                    <button id="sendButton" type="submit">Send</button>
                </div>
            </form>
        </div>
        <script>
        const chatHistory = document.getElementById('chatHistory');
        const queryInput = document.getElementById('query');
        const sendButton = document.getElementById('sendButton');
        const loadingSpinner = document.getElementById('loadingSpinner');

        sendButton.onclick = async () => {
            const query = queryInput.value.trim();
            if (!query) return;

            const userMessage = document.createElement('div');
            userMessage.className = 'chat-message user';
            userMessage.innerHTML = `<p>${query}</p>`;
            chatHistory.appendChild(userMessage);

            queryInput.value = '';
            loadingSpinner.style.display = 'block';
            chatHistory.scrollTop = chatHistory.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query })
                });
                const data = await response.json();
                const assistantMessage = document.createElement('div');
                assistantMessage.className = 'chat-message assistant';
                assistantMessage.innerHTML = `<p>${data.answer}</p>`;
                chatHistory.appendChild(assistantMessage);
            } catch (error) {
                console.error('Error fetching response:', error);
                const errorMessage = document.createElement('div');
                errorMessage.className = 'chat-message assistant';
                errorMessage.innerHTML = `<p>Sorry, something went wrong. Please try again.</p>`;
                chatHistory.appendChild(errorMessage);
            } finally {
                loadingSpinner.style.display = 'none';
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }
        };

        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendButton.click();
            }
        });

        // Auto-resize textarea
        queryInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
            // Limit max height
            if (this.scrollHeight > 100) {
                this.style.height = '100px';
                this.style.overflowY = 'auto';
            } else {
                this.style.overflowY = 'hidden';
            }
        });
        </script>
    </body>
    </html>
    '''

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    global chat_history
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    query_embedding = embedding_model.encode(query, convert_to_tensor=True).tolist()
    answer = generate_answer(
        query,
        query_embedding,
        transcript_dir,
        chat_history,
        text_generation_pipeline,
        top_k=top_k,
        threshold=threshold
    )

    chat_history.append({'query': query, 'answer': answer})

    return jsonify({'answer': str(answer)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)