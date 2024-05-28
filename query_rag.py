from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from doc_store import doc_store

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('faiss_index')

qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def rag_query(query, index, model, qa_pipeline):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype='float32'), k=5)  # Retrieve top 5 documents
    print("Document IDs:", I)  # Debugging statement
    
    # Retrieve the actual document texts using the document IDs
    documents = [doc_store.get(str(doc_id), "") for doc_id in I[0]]
    context = ' '.join(documents)
    print("Context:", context)  # Debugging statement
    answer = qa_pipeline(question=query, context=context)
    return answer
