# STEP 1: INSTALL REQUIRED LIBRARIES
# pip install Flask Flask-Cors pypdf PyMuPDF sentence-transformers faiss-cpu langchain langchain-google-genai langchain-core waitress

# STEP 2: IMPORT LIBRARIES
import os
import fitz  # PyMuPDF
import re
import pickle
import numpy as np
from pathlib import Path
import warnings
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sentence_transformers import SentenceTransformer
import faiss

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from waitress import serve # Import waitress

warnings.filterwarnings('ignore')

# --- IMPORTANT: API KEY SETUP ---
# Load the API key from an environment variable for security
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY environment variable not set! Please set it before running the app.")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
print("Gemini API key configured successfully!")


# --- ONE-TIME SETUP: This code runs only once when the server starts ---

print("Starting server... This may take a moment.")

# STEP 3: DOCUMENT PROCESSING FUNCTIONS
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        text += doc.load_page(page_num).get_text()
    doc.close()
    return text

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def add_context_to_chunks(chunks, document_title):
    return [f"Document: {document_title} | Section {i+1}: {chunk}" for i, chunk in enumerate(chunks)]

# STEP 4: LOAD AND PROCESS DOCUMENTS
pdf_files = ["Placement Chronicles 2023-24.pdf", "SI Chronicles 23-24 Sem I.pdf"]
if not all(os.path.exists(f) for f in pdf_files):
    print(f"Error: Missing required PDFs. Make sure {pdf_files} are in the same directory.")
    exit() # Exit if files are missing

documents = {file: extract_text_from_pdf(file) for file in pdf_files if file.endswith('.pdf')}
print(f"Extracted text from {len(documents)} documents.")

all_chunks = []
chunk_metadata = []
for filename, text in documents.items():
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)
    contextualized = add_context_to_chunks(chunks, filename)
    for i, (chunk, context_chunk) in enumerate(zip(chunks, contextualized)):
        all_chunks.append(context_chunk)
        chunk_metadata.append({
            'source': filename,
            'chunk_id': i,
            'text': chunk,
            'contextualized_text': context_chunk
        })
print(f"Created {len(all_chunks)} total text chunks.")

# STEP 5: CREATE EMBEDDINGS AND FAISS INDEX
print("Creating embeddings and vector store... (This is a one-time process)")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(all_chunks, normalize_embeddings=True, show_progress_bar=True)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))
print(f"FAISS index created with {index.ntotal} vectors.")

# STEP 6: RETRIEVAL AND RAG CHAIN SETUP
def search_similar_chunks(query, top_k=5):
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    _, indices = index.search(query_embedding.astype('float32'), top_k)
    return [chunk_metadata[idx]['text'] for idx in indices[0] if idx < len(chunk_metadata)]

def format_docs(docs):
    return "\n\n".join([f"CONTEXT SECTION {i+1}:\n{doc}" for i, doc in enumerate(docs)])

rag_prompt_template = """
You are a highly intelligent assistant with deep expertise in BITS Pilani's placement and internship chronicles.
Your job is to answer questions using the CONTEXT provided below. Always follow these guidelines:
INSTRUCTIONS:
1. Read ALL sections of the context thoroughly.
2. Extract and synthesize specific facts, figures, dates, names, and examples.
3. Give clear and direct answers. Be as detailed and specific as possible.
4. If the answer is not in the context, respond with: "The document does not contain enough information to answer this."
5. Do NOT make up information. Stick strictly to the facts from the documents.
6. Use bullet points or numbered lists wherever it improves clarity.
7. Mention company names, job roles, stipend/salary figures, interview rounds, and stats when relevant.
--- CONTEXT START ---
{context}
--- CONTEXT END ---
QUESTION: {question}
Answer (strictly using context):
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
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app) 

# API endpoint to handle questions
@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint to ask a question to the RAG pipeline."""
    if not request.json or 'question' not in request.json:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    question = request.json['question']
    print(f"Received Question: {question}")

    try:
        answer = rag_chain.invoke(question).strip()
        print(f"Sending Answer: {answer[:100]}...")
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return jsonify({"error": "Failed to process the question."}), 500

# Route to serve the frontend (index.html)
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    # Use Waitress as the production server
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting production server on http://0.0.0.0:{port}")
    serve(app, host='0.0.0.0', port=port)
