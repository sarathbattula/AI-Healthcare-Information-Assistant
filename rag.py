import os
import glob
import numpy as np

# RAG Libraries
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")

class RAGPipeline:
    def __init__(self, knowledge_dir=KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        self.embedder = None
        self.reranker = None
        self.index = None
        self.chunks = []  # Stores dict: {'text': ..., 'source': ..., 'page': ..., 'id': ...}
        self.is_indexed = False

    def initialize_models(self):
        """Lazy loader for embedding and re-ranking models."""
        if self.embedder is None:
            print(f"[RAG] Loading embedding model: {self.embedding_model_name}...")
            self.embedder = SentenceTransformer(self.embedding_model_name)
        return self.embedder

    def load_documents(self):
        """Loads and extracts text chunks from PDF and TXT files in knowledge directory."""
        documents_chunks = []
        file_paths = glob.glob(os.path.join(self.knowledge_dir, "*.*"))
        
        chunk_counter = 0
        for path in file_paths:
            filename = os.path.basename(path)
            ext = os.path.splitext(filename)[1].lower()

            if ext == '.pdf':
                try:
                    reader = PdfReader(path)
                    for page_idx, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        page_chunks = self._chunk_text(text, filename, page_number=page_idx + 1)
                        for c in page_chunks:
                            c['id'] = chunk_counter
                            documents_chunks.append(c)
                            chunk_counter += 1
                except Exception as e:
                    print(f"[RAG] Error reading PDF {filename}: {e}")

            elif ext == '.txt':
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    page_chunks = self._chunk_text(text, filename, page_number=1)
                    for c in page_chunks:
                        c['id'] = chunk_counter
                        documents_chunks.append(c)
                        chunk_counter += 1
                except Exception as e:
                    print(f"[RAG] Error reading TXT {filename}: {e}")

        self.chunks = documents_chunks
        return documents_chunks

    def _chunk_text(self, text, source_name, page_number=1, chunk_size=400, overlap=60):
        """Splits text into overlapping paragraph-level or token-level chunks."""
        cleaned_text = " ".join(text.split())
        if not cleaned_text:
            return []

        chunks = []
        start = 0
        text_len = len(cleaned_text)

        while start < text_len:
            end = start + chunk_size
            chunk_str = cleaned_text[start:end]

            # Try to break cleanly at sentence boundary if possible
            if end < text_len:
                last_period = chunk_str.rfind('. ')
                if last_period > 100:
                    end = start + last_period + 1
                    chunk_str = cleaned_text[start:end]

            chunks.append({
                'text': chunk_str.strip(),
                'source': source_name,
                'page': page_number
            })

            start = end - overlap if (end - overlap) > start else end

        return chunks

    def build_index(self, force_rebuild=False):
        """Embeds all document chunks and builds a FAISS index for vector retrieval."""
        if self.is_indexed and not force_rebuild:
            return

        self.initialize_models()
        self.load_documents()

        if not self.chunks:
            print("[RAG] Warning: No knowledge documents found to index.")
            return

        print(f"[RAG] Indexing {len(self.chunks)} text chunks into FAISS vector index...")
        texts = [c['text'] for c in self.chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Normalize vectors for Cosine Similarity using Inner Product Index
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.is_indexed = True
        print(f"[RAG] FAISS Index successfully built with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k=4, score_threshold=0.30, use_reranker=False):
        """
        Retrieves top relevant knowledge chunks for a user query.
        Returns:
            (retrieved_chunks_list, max_similarity_score)
        """
        if not self.is_indexed:
            self.build_index()

        if not self.index or self.index.ntotal == 0:
            return [], 0.0

        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        # Retrieve top 10 candidates first
        fetch_k = min(10, self.index.ntotal)
        scores, indices = self.index.search(query_vec, fetch_k)

        raw_scores = scores[0]
        raw_indices = indices[0]

        candidates = []
        for score, idx in zip(raw_scores, raw_indices):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk_copy = dict(self.chunks[idx])
            # Clip score to range [0.0, 1.0]
            sim_score = max(0.0, min(1.0, float(score)))
            chunk_copy['similarity_score'] = round(sim_score, 3)
            candidates.append(chunk_copy)

        if not candidates:
            return [], 0.0

        # Optional Cross-Encoder Re-ranking
        if use_reranker and len(candidates) > 1:
            try:
                if self.reranker is None:
                    print(f"[RAG] Loading Cross-Encoder re-ranker: {self.reranker_model_name}...")
                    self.reranker = CrossEncoder(self.reranker_model_name)
                
                pairs = [[query, c['text']] for c in candidates]
                rerank_scores = self.reranker.predict(pairs)
                
                for idx, r_score in enumerate(rerank_scores):
                    # Sigmoid transform to normalize cross-encoder logit into [0, 1]
                    norm_r_score = float(1.0 / (1.0 + np.exp(-r_score)))
                    candidates[idx]['similarity_score'] = round(norm_r_score, 3)

                candidates = sorted(candidates, key=lambda x: x['similarity_score'], reverse=True)
            except Exception as e:
                print(f"[RAG] Cross-Encoder re-ranking warning: {e}. Falling back to cosine scores.")

        final_chunks = candidates[:top_k]
        max_score = max([c['similarity_score'] for c in final_chunks]) if final_chunks else 0.0

        # Filter out chunks below threshold
        valid_chunks = [c for c in final_chunks if c['similarity_score'] >= score_threshold]

        return valid_chunks, max_score

# Singleton RAG instance
rag_engine = RAGPipeline()

if __name__ == '__main__':
    rag_engine.build_index()
    results, max_s = rag_engine.retrieve("What are the symptoms of hypertension?", use_reranker=False)
    print(f"Max Score: {max_s}")
    for r in results:
        print(f"Source: {r['source']} (p.{r['page']}) | Score: {r['similarity_score']}\nText: {r['text'][:120]}...\n")
