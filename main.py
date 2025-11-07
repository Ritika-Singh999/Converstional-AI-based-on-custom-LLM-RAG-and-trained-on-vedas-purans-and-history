import uvicorn
import os
import asyncio
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from google.api_core.exceptions import TooManyRequests
from scripture_retriever import ScriptureRetriever
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load environment variables
load_dotenv()

# Configure Gemini
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=API_KEY)

# Initialize Scripture Retriever
try:
    retriever = ScriptureRetriever()
except Exception as e:
    print(f"Failed to load ScriptureRetriever: {e}")
    retriever = None

# Initialize open-source LLM pipeline (using a small model for demonstration)
try:
    llm_pipeline = pipeline("text-generation", model="microsoft/DialoGPT-small", device=-1)  # Use CPU
except Exception as e:
    print(f"Failed to load open-source LLM: {e}")
    llm_pipeline = None

def search_web(query):
    url = f"https://www.bing.com/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for result in soup.find_all('li', class_='b_algo')[:5]:
                title = result.find('h2').text if result.find('h2') else ''
                link = result.find('a')['href'] if result.find('a') else ''
                snippet = result.find('p').text if result.find('p') else ''
                results.append({'title': title, 'link': link, 'snippet': snippet})
            return results
    except Exception as e:
        print(f"Web search error: {e}")
        return []



# FastAPI app
app = FastAPI(title="VedaAI - Sacred Texts Assistant", description="AI-powered queries on ancient Indian scriptures")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VedaAI - Ancient Wisdom Meets Modern Intelligence</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&family=Space+Grotesk:wght@300..700&display=swap');
            @keyframes shine {
                0%, 100% { background-position: 200% 0; }
                50% { background-position: -200% 0; }
            }
            .hero-heading { font-family: 'Space Grotesk', sans-serif; }
            .sanskrit-text { font-family: 'Playfair Display', serif; font-style: italic; }
            .feature-card:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 10px 40px rgba(168, 85, 247, 0.2); }
            .input-glow:focus { box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.5); outline: none; }
            .typing-cursor::after { content: '|'; animation: blink-caret 1s step-end infinite; }
            @keyframes blink-caret { from, to { color: transparent; } 50% { color: white; } }
            body { font-family: 'Playfair Display', serif; background: #0d0a1b; color: #E2E8F0; overflow-x: hidden; position: relative; min-height: 100vh; }
            .bg-animation { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; background: radial-gradient(circle at center, #1a0f2e 0%, #0d0a1b 100%); overflow: hidden; }
            .star { position: absolute; background-color: #8B5CF6; border-radius: 50%; animation: move-star 20s linear infinite; }
            @keyframes move-star { 0% { transform: translate(0, 0); opacity: 0.5; } 50% { opacity: 1; } 100% { transform: translate(100vw, 100vh); opacity: 0; } }
            .vedic-gradient { background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%); }
            .animate-slideIn { animation: slideIn 0.5s ease-out; }
            @keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
            .chat-message { display: flex; margin: 10px 0; align-items: flex-end; }
            .user-message { flex-direction: row-reverse; }
            .message-bubble { max-width: 70%; padding: 12px 16px; border-radius: 18px; word-wrap: break-word; }
            .user-bubble { background: linear-gradient(135deg, #8B5CF6, #EC4899); color: white; margin-left: auto; }
            .ai-bubble { background: rgba(255, 255, 255, 0.1); color: #E2E8F0; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2); }
            .avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 8px; }
            .user-avatar { background: linear-gradient(135deg, #8B5CF6, #EC4899); }
            .ai-avatar { background: rgba(255, 255, 255, 0.1); color: #8B5CF6; }
            #chat-container { height: 60vh; overflow-y: auto; padding: 20px; }
            #input-container { display: flex; padding: 20px; background: rgba(0, 0, 0, 0.5); border-top: 1px solid rgba(255, 255, 255, 0.1); }
            #message-input { flex: 1; padding: 12px 16px; border: none; border-radius: 25px; background: rgba(255, 255, 255, 0.1); color: white; outline: none; }
            #send-button { margin-left: 10px; padding: 12px 20px; background: linear-gradient(135deg, #8B5CF6, #EC4899); border: none; border-radius: 25px; color: white; cursor: pointer; }
            .welcome-message { text-align: center; padding: 20px; color: #A78BFA; font-style: italic; }
        </style>
    </head>
    <body class="bg-animation">
        <!-- Stars background -->
        <div id="stars"></div>

        <div class="container mx-auto px-4 py-8 max-w-4xl">
            <!-- Header -->
            <header class="text-center mb-12">
                <h1 class="hero-heading text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-500 mb-4">
                    VedaAI
                </h1>
                <p class="text-xl text-gray-300 max-w-2xl mx-auto">
                    Ancient Wisdom Meets Modern Intelligence
                </p>
                <p class="sanskrit-text text-lg mt-4 opacity-80">
                    "ॐ तत् सत्" - Om Tat Sat
                </p>
            </header>

            <!-- Chat Container -->
            <div id="chat-container" class="bg-black bg-opacity-20 rounded-2xl backdrop-blur-md border border-white border-opacity-10 mb-6">
                <div class="welcome-message">
                    <p>🕉 Welcome to VedaAI. Ask me about ancient Indian scriptures, Vedas, Puranas, or any question!</p>
                </div>
            </div>

            <!-- Input Container -->
            <div id="input-container">
                <input type="text" id="message-input" class="input-glow" placeholder="Ask about Vedas, Puranas, or anything..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button id="send-button" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            // Create stars
            function createStars() {
                const starsContainer = document.getElementById('stars');
                for (let i = 0; i < 100; i++) {
                    const star = document.createElement('div');
                    star.className = 'star';
                    star.style.left = Math.random() * 100 + '%';
                    star.style.top = Math.random() * 100 + '%';
                    star.style.width = Math.random() * 3 + 1 + 'px';
                    star.style.height = star.style.width;
                    star.style.animationDelay = Math.random() * 20 + 's';
                    starsContainer.appendChild(star);
                }
            }
            createStars();

            const chatContainer = document.getElementById('chat-container');
            const messageInput = document.getElementById('message-input');
            const sendButton = document.getElementById('send-button');

            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;

                // Add user message
                addMessage(message, 'user');
                messageInput.value = '';

                // Add thinking indicator
                const thinkingDiv = addMessage('Thinking...', 'ai');
                sendButton.disabled = true;
                sendButton.textContent = 'Sending...';

                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message})
                    });
                    const data = await response.json();
                    // Remove thinking
                    chatContainer.removeChild(thinkingDiv);
                    addMessage(data.response, 'ai');
                } catch (error) {
                    chatContainer.removeChild(thinkingDiv);
                    addMessage('Error: ' + error.message, 'ai');
                } finally {
                    sendButton.disabled = false;
                    sendButton.textContent = 'Send';
                }
            }

            function addMessage(text, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `chat-message ${sender === 'user' ? 'user-message' : ''} animate-slideIn`;
                
                const bubble = document.createElement('div');
                bubble.className = `message-bubble ${sender === 'user' ? 'user-bubble' : 'ai-bubble'}`;
                bubble.innerHTML = text.replace(/\n/g, '<br>');

                const avatar = document.createElement('div');
                avatar.className = `avatar ${sender === 'user' ? 'user-avatar' : 'ai-avatar'}`;
                avatar.innerHTML = sender === 'user' ? '👤' : '🕉';

                if (sender === 'user') {
                    messageDiv.appendChild(bubble);
                    messageDiv.appendChild(avatar);
                } else {
                    messageDiv.appendChild(avatar);
                    messageDiv.appendChild(bubble);
                }

                // Remove welcome if first message
                const welcome = chatContainer.querySelector('.welcome-message');
                if (welcome) welcome.remove();

                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;

                return messageDiv;
            }

            // Event listeners
            sendButton.addEventListener('click', sendMessage);
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    """

# Request model
class ChatRequest(BaseModel):
    message: str

# Chat endpoint
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message.strip()

    try:
        # Step 1: Retrieve relevant passages from the indexed corpus
        relevant_passages = retriever.retrieve(user_message)
        # Step 2: Search web for additional information
        web_results = search_web(user_message)
        # Prepare context from relevant passages and web results
        context = ""
        if relevant_passages:
            context = "\n\n".join([
                f"From {p['source']}:\n{p['text']}"
                for p in relevant_passages
            ])
        web_context = "\n\n".join([
            f"Web: {r['title']} - {r['snippet']}"
            for r in web_results
        ])
        context += web_context

        # Prepare prompt for Gemini
        if relevant_passages:
            # RAG mode: Scripture and web
            prompt = f"""You are VedaAI, an expert AI trained on ancient Indian sacred texts and web knowledge. Your purpose is to answer questions based on the provided passages and web results, merging offline scripture data with online information for accurate, comprehensive responses.

First, analyze the user's question and the provided "Relevant Passages and Web Results." Merge the information from scriptures and web to provide the best answer.

- Generate a logical, accurate response citing sources from scriptures and web when possible.
- If the question is about ancient Indian scriptures, prioritize scripture passages but enhance with web info.
- For general questions, use web results and your knowledge.
- Keep responses concise but informative (2-3 lines or more if needed).
- Respond in English, incorporating Sanskrit or Hindi terms for religious concepts where appropriate. Provide natural, informative responses.

Relevant Passages and Web Results:
{context}

Question: {user_message}

Answer:"""
        else:
            # General mode: Use web and Gemini's knowledge
            web_context = "\n\n".join([
                f"Web: {r['title']} - {r['snippet']}"
                for r in web_results
            ])
            prompt = f"""You are VedaAI, a helpful AI assistant. Answer the user's question using your knowledge and the provided web results. If the question is about ancient Indian scriptures, provide accurate information with citations if possible. Keep responses short and accurate.

Respond in English, incorporating Sanskrit or Hindi terms for religious concepts where appropriate. Provide natural, informative responses.

Web Results:
{web_context}

Question: {user_message}

Answer:"""

        # Try Gemini first, fallback to open-source LLM if available
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    response = model.generate_content(prompt)
                    return {"response": response.text}
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota exceeded" in error_msg.lower():
                        if attempt < max_retries - 1:
                            match = re.search(r'retry_delay {\s*seconds: (\d+(?:\.\d+)?)\s*}', error_msg)
                            if match:
                                delay = float(match.group(1))
                                print(f"Rate limit hit, retrying in {delay} seconds...")
                                await asyncio.sleep(delay)
                            else:
                                print("Rate limit hit, retrying in 10 seconds...")
                                await asyncio.sleep(10)
                            continue
                        else:
                            raise
                    else:
                        raise
        except Exception as e:
            print(f"Gemini failed: {e}, trying open-source LLM...")
            if llm_pipeline:
                try:
                    # Use open-source LLM as fallback
                    full_prompt = f"Context: {context}\n\nQuestion: {user_message}\n\nAnswer:"
                    generated = llm_pipeline(full_prompt, max_length=200, num_return_sequences=1, temperature=0.7)
                    response_text = generated[0]['generated_text'].replace(full_prompt, "").strip()
                    return {"response": response_text}
                except Exception as e2:
                    print(f"Open-source LLM also failed: {e2}")
                    raise HTTPException(status_code=500, detail="Both Gemini and open-source LLM failed")
            else:
                raise HTTPException(status_code=500, detail="Gemini failed and no open-source LLM available")

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print("Exception traceback:\n", traceback_str)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Run the app
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
