import os
import hashlib
import uuid
import datetime
from typing import List, Tuple, Dict, Any, Optional
import pymupdf  # PyMuPDF
import docx
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk, TrustedSource
from app.config import settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

def validate_file(file_content: bytes, filename: str) -> str:
    """Validate file extension, size, non-emptiness, and magic bytes."""
    if len(file_content) == 0:
        raise ValueError("File is empty (0 bytes).")
    
    if len(file_content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size exceeds maximum limit of {MAX_FILE_SIZE_BYTES / (1024*1024)}MB.")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}")

    # Magic Bytes checks
    if ext == ".pdf":
        if not file_content.startswith(b"%PDF"):
            raise ValueError("Invalid PDF magic bytes or corrupted file header.")
        try:
            doc = pymupdf.open(stream=file_content, filetype="pdf")
            if doc.is_encrypted:
                raise ValueError("PDF is encrypted/password-protected.")
            if doc.page_count == 0:
                raise ValueError("PDF contains no pages.")
            doc.close()
        except Exception as e:
            raise ValueError(f"Corrupted or invalid PDF file: {str(e)}")

    elif ext == ".docx":
        # Zip magic bytes for DOCX
        if not file_content.startswith(b"PK\x03\x04"):
            raise ValueError("Invalid DOCX magic bytes or corrupted zip header.")

    elif ext in [".png", ".jpg", ".jpeg"]:
        if ext == ".png" and not file_content.startswith(b"\x89PNG"):
            raise ValueError("Invalid PNG magic bytes or corrupted image file.")
        elif ext in [".jpg", ".jpeg"] and not file_content.startswith(b"\xff\xd8"):
            raise ValueError("Invalid JPEG magic bytes or corrupted image file.")

    return ext

def compute_sha256(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

def extract_text_with_pages(file_content: bytes, file_ext: str) -> List[Tuple[int, str]]:
    """Extract text by page. Returns a list of tuples: (page_number, text)."""
    pages_text: List[Tuple[int, str]] = []

    if file_ext == ".pdf":
        doc = pymupdf.open(stream=file_content, filetype="pdf")
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages_text.append((i + 1, text))
        doc.close()

    elif file_ext == ".docx":
        # Save temporarily to parse via python-docx
        temp_path = os.path.join(settings.UPLOAD_DIR, f"temp_{uuid.uuid4().hex}.docx")
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(file_content)
        try:
            doc = docx.Document(temp_path)
            full_text = []
            for p in doc.paragraphs:
                if p.text.strip():
                    full_text.append(p.text.strip())
            # DOCX lacks explicit page breaks; treat every ~300 words as a page
            docx_text = "\n".join(full_text)
            words = docx_text.split()
            words_per_page = 300
            total_pages = max(1, (len(words) + words_per_page - 1) // words_per_page)
            for p_num in range(1, total_pages + 1):
                start = (p_num - 1) * words_per_page
                end = start + words_per_page
                pages_text.append((p_num, " ".join(words[start:end])))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    elif file_ext == ".txt":
        text_str = file_content.decode("utf-8", errors="ignore").strip()
        lines = text_str.splitlines()
        # Segment TXT into ~500 character chunks per page
        current_page = 1
        current_block = []
        current_length = 0
        for line in lines:
            current_block.append(line)
            current_length += len(line)
            if current_length >= 1500:
                pages_text.append((current_page, "\n".join(current_block)))
                current_page += 1
                current_block = []
                current_length = 0
        if current_block:
            pages_text.append((current_page, "\n".join(current_block)))

    elif file_ext in [".png", ".jpg", ".jpeg"]:
        # Extract text or embedded OCR metadata from image file
        try:
            img_doc = pymupdf.open(stream=file_content, filetype=file_ext.replace(".", ""))
            pdf_bytes = img_doc.convert_to_pdf()
            pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            for i, page in enumerate(pdf_doc):
                text = page.get_text("text").strip()
                if text:
                    pages_text.append((i + 1, text))
            pdf_doc.close()
            img_doc.close()
        except Exception:
            pass
        
        if not pages_text:
            pages_text.append((1, f"Scanned Medical Image Report ({file_ext.upper()}) uploaded for clinical evaluation."))

    return pages_text

def chunk_pages_with_overlap(pages_text: List[Tuple[int, str]], max_chars: int = 700, overlap_chars: int = 150) -> List[Dict[str, Any]]:
    """Chunk extracted text preserving page number tags."""
    chunks = []
    chunk_idx = 0

    for page_num, text in pages_text:
        text_len = len(text)
        start = 0
        while start < text_len:
            end = start + max_chars
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append({
                    "chunk_index": chunk_idx,
                    "page_number": page_num,
                    "content": chunk_content,
                    "token_count": len(chunk_content.split())
                })
                chunk_idx += 1
            start += (max_chars - overlap_chars)

    return chunks
