import os
import pickle
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- IMPORTANT: API KEY SETUP ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY environment variable not set!")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
print("Gemini API key configured successfully!")

# --- LOAD PRE-COMPUTED ARTIFACTS ---
print("Loading pre-computed artifacts...")
index = faiss.read_index("faiss_index.bin")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./cache')
print("Artifacts loaded successfully.")

# --- LLM and RAG CHAIN SETUP ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, max_tokens=1024)

def search_similar_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    _, indices = index.search(query_embedding.astype('float32'), top_k)
    return [chunks[idx] for idx in indices[0]]

def format_docs(docs):
    return "\n\n".join([f"CONTEXT SECTION {i+1}:\n{doc}" for i, doc in enumerate(docs)])

# --- NEW CONVERSATIONAL PROMPT ---
rag_prompt_template = """
You are an intelligent HR assistant. Your goal is to provide accurate answers based on the CONTEXT provided.
You must follow a two-step process:

Step 1: Analyze the user's question and the conversation history.
- The user's latest question is: "{question}"
- The conversation history is:
{chat_history}

Step 2: Analyze the retrieved CONTEXT below.
--- CONTEXT START ---
{context}
--- CONTEXT END ---

Step 3: Decide your action.
- **If the CONTEXT provides different information based on a user's role, position, or level (e.g., different travel budgets for Associates vs. CEOs), AND the user has NOT yet provided this information in the conversation history, your ONLY response must be to ask for the missing information.** For example, respond with: "To provide the most accurate information, could you please tell me your position or level?"
- **Otherwise, answer the user's latest question using the conversation history and the retrieved CONTEXT.** If the user is providing the information you just asked for, use it to answer their original question. If the context doesn't contain the answer, say so.
"""
rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

def format_chat_history(history):
    if not history:
        return "No history."
    return "\n".join([f"{msg['role']}: {msg['text']}" for msg in history])

rag_chain = (
    {
        "context": lambda x: format_docs(search_similar_chunks(x["question"])),
        "question": lambda x: x["question"],
        "chat_history": lambda x: format_chat_history(x["chat_history"])
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)
print("Gemini Conversational RAG chain is ready!")


# --- FLASK API ---
app = Flask(__name__)
CORS(app)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    chat_history = data.get('chat_history', []) # Expect chat history from frontend
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    try:
        # Pass the question and history to the chain
        answer = rag_chain.invoke({"question": question, "chat_history": chat_history}).strip()
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error in RAG chain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_text():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400
    try:
        response = llm.invoke(prompt)
        return jsonify({"text": response.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
