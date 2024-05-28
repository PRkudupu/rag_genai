from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from doc_store import doc_store, save_doc_store

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)  # Dimension of sentence-transformer model

def encode_and_store(text, doc_id, index):
    embeddings = model.encode([text])
    index.add(np.array(embeddings, dtype='float32'))
    doc_store[doc_id] = text  # Store the document text
    return embeddings

def save_index(index, index_path='faiss_index'):
    faiss.write_index(index, index_path)
    save_doc_store()

def load_index(index_path='faiss_index'):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        return faiss.IndexFlatL2(384)
