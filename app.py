from flask import Flask, request, jsonify
import os
from extract_text import extract_text_from_pdf
from encode_and_store import encode_and_store, save_index, load_index
from query_rag import rag_query
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
from doc_store import load_doc_store

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models and index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = load_index()
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

# Ensure doc_store is loaded
load_doc_store()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        text = extract_text_from_pdf(file_path)
        print("Extracted text:", text[:500])  # Debugging statement
        encode_and_store(text, file.filename, index)
        save_index(index)
        return jsonify({'message': 'File uploaded and processed successfully', 'file_path': file_path})

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query')
    print("Query:", query)  # Debugging statement
    answer = rag_query(query, index, model, qa_pipeline)
    print("Answer:", answer)  # Debugging statement
    return jsonify(answer)

if __name__ == '__main__':
    app.run(debug=True,port=5002)
