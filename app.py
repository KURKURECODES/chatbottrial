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
# Load the FAISS index from disk
index = faiss.read_index("faiss_index.bin")
# Load the text chunks from disk
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./cache')
print("Artifacts loaded successfully.")

# --- RAG CHAIN SETUP ---
def search_similar_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    _, indices = index.search(query_embedding.astype('float32'), top_k)
    return [chunks[idx] for idx in indices[0]]

def format_docs(docs):
    return "\n\n".join([f"CONTEXT SECTION {i+1}:\n{doc}" for i, doc in enumerate(docs)])

rag_prompt_template = """
You are a helpful and intelligent assistant. Your job is to answer questions using the CONTEXT provided below.
Follow these guidelines strictly:
1. Read the context carefully to find the most relevant information.
2. Provide clear, direct, and accurate answers based ONLY on the provided context.
3. If the answer cannot be found in the context, you must respond with: "The provided document does not contain enough information to answer this question."
4. Do not invent information or use any external knowledge.
--- CONTEXT START ---
{context}
--- CONTEXT END ---
QUESTION: {question}
Answer (based only on the context):
"""
rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, max_tokens=1024)

rag_chain = (
    {"context": lambda x: format_docs(search_similar_chunks(x, top_k=5)), "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
print("Gemini RAG chain is ready!")


# --- FLASK API ---
# The 'app' variable is what Gunicorn will look for
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    try:
        answer = rag_chain.invoke(question).strip()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

# Note: The if __name__ == '__main__': block is no longer needed
# as Gunicorn is now our entry point.
