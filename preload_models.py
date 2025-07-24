import os
import fitz  # PyMuPDF
import re
import pickle
from sentence_transformers import SentenceTransformer
import faiss

print("--- Starting Preloading and Indexing ---")

# --- Document Processing Functions ---
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

# --- Load and Process Document ---
pdf_file = "LogicLadder.pdf"
if not os.path.exists(pdf_file):
    raise FileNotFoundError(f"Error: Could not find {pdf_file}")

print(f"Processing document: {pdf_file}")
text = extract_text_from_pdf(pdf_file)
cleaned = clean_text(text)
chunks = chunk_text(cleaned)
print(f"Created {len(chunks)} text chunks.")

# --- Create Embeddings and FAISS Index ---
print("Loading sentence-transformer model...")
# Use the local cache folder we created in the Dockerfile
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./cache')
print("Creating embeddings...")
embeddings = embedding_model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))
print(f"FAISS index created with {index.ntotal} vectors.")

# --- Save Artifacts to Disk ---
faiss.write_index(index, "faiss_index.bin")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("--- Preloading and Indexing Complete ---")
