import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import pickle

class ScriptureRetriever:
    def __init__(self):
        self.CORPUS_DIR = os.path.join(os.path.dirname(__file__), 'corpus')
        self.FAISS_INDEX_FILE = os.path.join(os.path.dirname(__file__), 'faiss_index.idx')
        self.DATA_FILE = os.path.join(os.path.dirname(__file__), 'data.pkl')
        self.MODEL_NAME = 'all-MiniLM-L6-v2'
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(self.MODEL_NAME)
        
        # Initialize FAISS index
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Data storage
        self.documents = []
        self.metadatas = []
        
        # Load existing index if available
        self.load_index()

    def preprocess_text(self, text):
        # Remove page markers and unnecessary whitespace
        text = re.sub(r'--- Page \d+ ---', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text, min_length=100):
        # Split into meaningful chunks (verses/paragraphs)
        chunks = []
        current_chunk = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                if current_chunk and len(' '.join(current_chunk)) >= min_length:
                    chunks.append(' '.join(current_chunk))
                current_chunk = []
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def load_index(self):
        if os.path.exists(self.FAISS_INDEX_FILE) and os.path.exists(self.DATA_FILE):
            self.index = faiss.read_index(self.FAISS_INDEX_FILE)
            with open(self.DATA_FILE, 'rb') as f:
                self.documents, self.metadatas = pickle.load(f)
            print("Loaded existing FAISS index and data.")

    def save_index(self):
        faiss.write_index(self.index, self.FAISS_INDEX_FILE)
        with open(self.DATA_FILE, 'wb') as f:
            pickle.dump((self.documents, self.metadatas), f)
        print("FAISS index and data saved to disk.")

    def index_corpus(self):
        total_chunks = 0
        for filename in os.listdir(self.CORPUS_DIR):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.CORPUS_DIR, filename)
                source_name = filename.replace('_text.txt', '').replace('.txt', '')
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Preprocess the text
                text = self.preprocess_text(text)
                
                # Split into chunks
                chunks = self.chunk_text(text)
                
                if not chunks:
                    continue

                # Create embeddings for chunks
                embeddings = self.embedder.encode(chunks)
                
                # Normalize embeddings for cosine similarity
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                # Add to FAISS index
                self.index.add(embeddings.astype('float32'))
                
                # Store documents and metadatas
                self.documents.extend(chunks)
                self.metadatas.extend([{"source": source_name} for _ in chunks])
                
                total_chunks += len(chunks)
                print(f"Indexed {len(chunks)} chunks from {source_name}")
        
        print(f"\nTotal chunks indexed: {total_chunks}")

        # Persist the index and data to disk
        self.save_index()
        print("Index and data persisted to disk successfully!")

    def retrieve(self, query, top_k=3):
        if self.index.ntotal == 0:
            return []
        
        # Encode and normalize query
        query_embedding = self.embedder.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding.astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        passages = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid index
                relevance = float(dist)  # Inner product is similarity
                if relevance > 0.3:  # Threshold for relevance
                    passages.append({
                        'text': self.documents[idx],
                        'source': self.metadatas[idx]['source'],
                        'relevance': relevance
                    })
        
        return passages

if __name__ == "__main__":
    retriever = ScriptureRetriever()
    print("Starting indexing of sacred texts...")
    retriever.index_corpus()
    print("Indexing complete! The system is ready for queries.")