import os
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, text

from app.config import settings
from app.models import Document, DocumentChunk, TrustedSource

# Initialize sentence transformer model separately from LLM
_embedding_model: Optional[SentenceTransformer] = None

def get_embedding_model() -> Optional[SentenceTransformer]:
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            print(f"[RAG Engine] Warning: Failed to load SentenceTransformer '{settings.EMBEDDING_MODEL_NAME}': {e}")
            return None
    return _embedding_model

# Initialize Chroma vector store client
os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
_chroma_client = chromadb.PersistentClient(
    path=settings.CHROMA_DB_DIR,
    settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False)
)
_collection = _chroma_client.get_or_create_collection(
    name="healthcare_knowledge_v2",
    metadata={"hnsw:space": "cosine"}
)

def index_document_chunks(db: Session, doc: Document, chunks: List[Dict[str, Any]]) -> List[DocumentChunk]:
    """Index chunks in both SQLite FTS5 table and ChromaDB vector store."""
    model = get_embedding_model()

    db_chunks = []
    chroma_ids = []
    chroma_embeddings = []
    chroma_metadatas = []
    chroma_documents = []

    for chunk_data in chunks:
        embed_id = f"doc_{doc.id}_chunk_{chunk_data['chunk_index']}_{uuid.uuid4().hex[:6]}"
        db_chunk = DocumentChunk(
            doc_id=doc.id,
            chunk_index=chunk_data['chunk_index'],
            page_number=chunk_data['page_number'],
            content=chunk_data['content'],
            token_count=chunk_data['token_count'],
            embedding_id=embed_id
        )
        db.add(db_chunk)
        db.flush()  # assign db_chunk.id

        # SQLite FTS5 insertion via SQLAlchemy DB session
        db.execute(
            text("INSERT INTO chunk_fts(chunk_id, doc_id, page_number, content) VALUES (:chunk_id, :doc_id, :page_number, :content)"),
            {
                "chunk_id": db_chunk.id,
                "doc_id": doc.id,
                "page_number": chunk_data['page_number'],
                "content": chunk_data['content']
            }
        )

        # Prepare ChromaDB payload
        if model:
            try:
                embedding_vec = model.encode(chunk_data['content']).tolist()
                chroma_ids.append(embed_id)
                chroma_embeddings.append(embedding_vec)
                chroma_documents.append(chunk_data['content'])
                chroma_metadatas.append({
                    "chunk_db_id": db_chunk.id,
                    "doc_id": doc.id,
                    "page_number": chunk_data['page_number'],
                    "doc_type": doc.doc_type,
                    "owner_id": doc.owner_id or 0,
                    "title": doc.title,
                    "publisher": doc.publisher,
                    "url": doc.url or "",
                    "publication_date": doc.publication_date or ""
                })
            except Exception as e:
                print(f"[RAG Engine] Chunk encoding error: {e}")
        db_chunks.append(db_chunk)

    db.commit()

    # Store in Chroma
    if chroma_ids:
        try:
            _collection.add(
                ids=chroma_ids,
                embeddings=chroma_embeddings,
                metadatas=chroma_metadatas,
                documents=chroma_documents
            )
        except Exception as e:
            print(f"[RAG Engine] Chroma store error: {e}")

    return db_chunks


def delete_document_chunks(db: Session, doc_id: int):
    """Delete document chunks from SQLite DB, FTS5 index, and Chroma vector store."""
    try:
        _collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass

    # Delete from FTS5 via SQLAlchemy session
    db.execute(text("DELETE FROM chunk_fts WHERE doc_id = :doc_id"), {"doc_id": doc_id})
    db.query(DocumentChunk).filter(DocumentChunk.doc_id == doc_id).delete()
    db.commit()


def hybrid_search(
    db: Session,
    query: str,
    user_id: Optional[int] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform hybrid vector + FTS5 keyword search with Reciprocal Rank Fusion (RRF).
    Enforces per-user access control in SQL and vector store filters on every call.
    """
    # 1. Fetch allowed doc_ids for this user
    target_user_id = user_id if user_id is not None else -1
    allowed_docs_query = db.query(Document).filter(
        Document.status == "indexed",
        or_(
            Document.doc_type == "trusted",
            and_(Document.doc_type == "private", Document.owner_id == target_user_id)
        )
    )
    allowed_docs = allowed_docs_query.all()
    allowed_doc_ids = {d.id for d in allowed_docs}
    if not allowed_doc_ids:
        return []

    doc_map = {d.id: d for d in allowed_docs}

    # 2. Vector Search via Chroma
    vector_ranks: Dict[int, int] = {} # chunk_id -> rank (1-indexed)
    vector_meta: Dict[int, Dict[str, Any]] = {}
    
    try:
        model = get_embedding_model()
        if model:
            query_vec = model.encode(query).tolist()
            vector_results = _collection.query(
                query_embeddings=[query_vec],
                n_results=top_k * 3
            )
            if vector_results and vector_results.get("metadatas"):
                metas = vector_results["metadatas"][0]
                distances = vector_results["distances"][0] if vector_results.get("distances") else []
                for rank, meta in enumerate(metas, start=1):
                    doc_id = meta.get("doc_id")
                    chunk_db_id = meta.get("chunk_db_id")
                    if doc_id in allowed_doc_ids and chunk_db_id:
                        if chunk_db_id not in vector_ranks:
                            vector_ranks[chunk_db_id] = rank
                            vector_meta[chunk_db_id] = {
                                "distance": distances[rank - 1] if rank - 1 < len(distances) else 0.5,
                                "metadata": meta
                            }
    except Exception as e:
        print(f"[RAG Engine] Vector search error: {e}")

    # 3. Keyword Search via SQLite FTS5 through SQLAlchemy
    sanitized_q = ' OR '.join([f'"{w.strip()}"' for w in query.replace('"', '').split() if len(w.strip()) > 2])
    fts_ranks: Dict[int, int] = {}
    
    if sanitized_q:
        try:
            placeholders = ','.join([f":id_{i}" for i in range(len(allowed_doc_ids))])
            params = {f"id_{i}": doc_id for i, doc_id in enumerate(allowed_doc_ids)}
            params["q"] = sanitized_q
            params["limit"] = top_k * 3
            
            stmt = text(f"""
                SELECT chunk_id, doc_id, page_number, content, rank
                FROM chunk_fts
                WHERE chunk_fts MATCH :q AND doc_id IN ({placeholders})
                ORDER BY rank LIMIT :limit
            """)
            rows = db.execute(stmt, params).fetchall()
            for rank, row in enumerate(rows, start=1):
                c_id = row[0]
                if c_id not in fts_ranks:
                    fts_ranks[c_id] = rank
        except Exception:
            pass

    # 4. Reciprocal Rank Fusion (RRF)
    all_chunk_ids = set(vector_ranks.keys()).union(set(fts_ranks.keys()))
    fusion_scores = []

    for c_id in all_chunk_ids:
        r_vec = vector_ranks.get(c_id, 999)
        r_fts = fts_ranks.get(c_id, 999)
        score = (1.0 / (60.0 + r_vec)) + (1.0 / (60.0 + r_fts))
        fusion_scores.append((score, c_id))

    fusion_scores.sort(key=lambda x: x[0], reverse=True)
    top_chunk_ids = [c_id for _, c_id in fusion_scores[:top_k]]

    if not top_chunk_ids:
        return []

    # 5. Hydrate Chunk Metadata from SQLAlchemy DB
    chunks_in_db = db.query(DocumentChunk).filter(DocumentChunk.id.in_(top_chunk_ids)).all()
    chunk_dict = {c.id: c for c in chunks_in_db}

    results = []
    for idx, (score, c_id) in enumerate(fusion_scores[:top_k], start=1):
        c_obj = chunk_dict.get(c_id)
        if not c_obj:
            continue
        doc_obj = doc_map.get(c_obj.doc_id)
        if not doc_obj:
            continue

        is_outdated = False
        is_superseded = False
        if doc_obj.trusted_source:
            if doc_obj.trusted_source.status == "outdated":
                is_outdated = True
            elif doc_obj.trusted_source.status == "superseded":
                is_superseded = True

        distance = vector_meta.get(c_id, {}).get("distance", 0.5)
        similarity_score = round(max(0.0, 1.0 - distance), 4)

        results.append({
            "citation_id": f"E{idx}",
            "chunk_id": c_obj.id,
            "doc_id": doc_obj.id,
            "doc_type": doc_obj.doc_type,
            "title": doc_obj.title,
            "publisher": doc_obj.publisher,
            "url": doc_obj.url or "",
            "page_number": c_obj.page_number,
            "publication_date": doc_obj.publication_date or "N/A",
            "content": c_obj.content,
            "rrf_score": round(score, 5),
            "similarity_score": similarity_score,
            "is_outdated": is_outdated,
            "is_superseded": is_superseded
        })

    return results
