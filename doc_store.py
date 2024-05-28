import json
import os

doc_store = {}

def load_doc_store(doc_store_path='doc_store.json'):
    global doc_store
    if os.path.exists(doc_store_path):
        with open(doc_store_path, 'r') as f:
            doc_store = json.load(f)
    return doc_store

def save_doc_store(doc_store_path='doc_store.json'):
    with open(doc_store_path, 'w') as f:
        json.dump(doc_store, f)

# Load doc_store initially
load_doc_store()
