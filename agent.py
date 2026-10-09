import os
import json

import safety
import db
from rag import rag_engine

# Attempt to import anthropic, handle gracefully if not installed
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

class HealthcareAgent:
    def __init__(self):
        self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.client = None

        if HAS_ANTHROPIC and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                print(f"[Agent] Anthropic client initialized with model: {self.model_name}")
            except Exception as e:
                print(f"[Agent] Failed to initialize Anthropic client: {e}")
                self.client = None

    def process_query(self, user_query: str, language: str = "English", use_reranker: bool = False) -> dict:
        """
        Main processing pipeline for user health queries:
        1. PII Masking
        2. Emergency Detection
        3. Risky Topic Safety Check
        4. RAG Retrieval & Confidence Scoring
        5. LLM Grounded Generation (or local synthesis fallback)
        6. Audit Logging & Review Queue Escalation
        """
        # Ensure database is ready
        db.init_db()

        # Step 1: PII Masking
        sanitized_query, has_pii = safety.mask_pii(user_query)

        # Step 2: Emergency Detection
        is_emergency, emergency_kw = safety.detect_emergency(sanitized_query)
        if is_emergency:
            emergency_response = {
                "response_type": "EMERGENCY",
                "message": (
                    f"CRITICAL HEALTH WARNING: Your query contained emergency indicators ('{emergency_kw}'). "
                    "We cannot provide medical treatment advice for emergency conditions."
                ),
                "emergency_numbers": [
                    {"label": "National Emergency Services (India)", "number": "112"},
                    {"label": "Ambulance Services (India)", "number": "108"},
                    {"label": "Police Emergency", "number": "100"}
                ],
                "action_required": "Please call emergency services (112 or 108) or visit the nearest hospital emergency room immediately.",
                "citations": [],
                "confidence_score": 1.0,
                "sanitized_query": sanitized_query,
                "is_pii_masked": has_pii
            }
            db.log_audit(
                user_query=user_query,
                sanitized_query=sanitized_query,
                response_type="EMERGENCY",
                sources=[],
                confidence_score=1.0,
                language=language,
                is_pii_masked=has_pii
            )
            return emergency_response

        # Step 3: Risky Topic Safety Check
        is_risky, risky_kw = safety.detect_risky_topic(sanitized_query)
        if is_risky:
            # Escalation to Human Professional Queue
            review_id = db.add_to_review_queue(
                question=user_query,
                context=f"Detected risky keyword: '{risky_kw}'",
                reason="RISKY_TOPIC",
                pii_sanitized_question=sanitized_query
            )
            escalated_response = {
                "response_type": "ESCALATED",
                "reason": "RISKY_TOPIC",
                "review_id": review_id,
                "message": (
                    f"Your question involves high-risk medical advice (keyword: '{risky_kw}', such as dosage, prescription, or diagnosis). "
                    "For your safety, this question has been routed to our professional reviewer queue."
                ),
                "status_info": f"Queue ID #{review_id}. A qualified reviewer will evaluate this request.",
                "citations": [],
                "confidence_score": 0.0,
                "sanitized_query": sanitized_query,
                "is_pii_masked": has_pii
            }
            db.log_audit(
                user_query=user_query,
                sanitized_query=sanitized_query,
                response_type="ESCALATED",
                sources=[],
                confidence_score=0.0,
                language=language,
                is_pii_masked=has_pii
            )
            return escalated_response

        # Step 4: RAG Knowledge Retrieval
        chunks, max_confidence = rag_engine.retrieve(
            query=sanitized_query,
            top_k=4,
            score_threshold=0.28,
            use_reranker=use_reranker
        )

        if not chunks or max_confidence < 0.28:
            review_id = db.add_to_review_queue(
                question=user_query,
                context=f"Max retrieval similarity score: {max_confidence:.3f} (below 0.28 threshold)",
                reason="LOW_RETRIEVAL_CONFIDENCE",
                pii_sanitized_question=sanitized_query
            )
            escalated_response = {
                "response_type": "ESCALATED",
                "reason": "LOW_RETRIEVAL_CONFIDENCE",
                "review_id": review_id,
                "message": (
                    "Our trusted medical knowledge base does not contain sufficient grounded evidence to answer your query with high confidence."
                ),
                "status_info": f"Queue ID #{review_id}. Escalated to expert review to avoid hallucinated information.",
                "citations": [],
                "confidence_score": round(max_confidence, 3),
                "sanitized_query": sanitized_query,
                "is_pii_masked": has_pii
            }
            db.log_audit(
                user_query=user_query,
                sanitized_query=sanitized_query,
                response_type="ESCALATED",
                sources=[],
                confidence_score=round(max_confidence, 3),
                language=language,
                is_pii_masked=has_pii
            )
            return escalated_response

        # Step 5: Synthesize Grounded Answer with LLM or Fallback Engine
        context_str = "\n\n".join([
            f"[Source: {c['source']}, Page {c['page']}, Similarity Score: {c['similarity_score']}]\n{c['text']}"
            for c in chunks
        ])

        citations = [
            {
                "source": c["source"],
                "page": c["page"],
                "similarity_score": c["similarity_score"]
            }
            for c in chunks
        ]

        answer_text = self._generate_llm_answer(sanitized_query, context_str, language)

        # Step 6: Check for LLM "INSUFFICIENT" response flag
        if "INSUFFICIENT" in answer_text.strip().upper()[:20]:
            review_id = db.add_to_review_queue(
                question=user_query,
                context=context_str,
                reason="INSUFFICIENT_EVIDENCE",
                pii_sanitized_question=sanitized_query
            )
            escalated_response = {
                "response_type": "ESCALATED",
                "reason": "INSUFFICIENT_EVIDENCE",
                "review_id": review_id,
                "message": "The available knowledge documents could not provide a complete grounded answer.",
                "status_info": f"Queue ID #{review_id}. Escalated for professional evaluation.",
                "citations": citations,
                "confidence_score": round(max_confidence, 3),
                "sanitized_query": sanitized_query,
                "is_pii_masked": has_pii
            }
            db.log_audit(
                user_query=user_query,
                sanitized_query=sanitized_query,
                response_type="ESCALATED",
                sources=citations,
                confidence_score=round(max_confidence, 3),
                language=language,
                is_pii_masked=has_pii
            )
            return escalated_response

        # Step 7: Successful Grounded Answer
        response_data = {
            "response_type": "GROUNDED_ANSWER",
            "answer": answer_text,
            "citations": citations,
            "confidence_score": round(max_confidence, 3),
            "language": language,
            "sanitized_query": sanitized_query,
            "is_pii_masked": has_pii,
            "disclaimer": "For information only. Not medical advice."
        }

        db.log_audit(
            user_query=user_query,
            sanitized_query=sanitized_query,
            response_type="GROUNDED_ANSWER",
            sources=citations,
            confidence_score=round(max_confidence, 3),
            language=language,
            is_pii_masked=has_pii
        )

        return response_data

    def _generate_llm_answer(self, query: str, context: str, language: str) -> str:
        """Helper to invoke Anthropic Claude API or use deterministic RAG synthesis fallback."""
        system_prompt = (
            "You are a strict, trusted Healthcare Information Assistant. "
            "Your job is to answer user queries ONLY based on the provided healthcare context passages. "
            "RULES:\n"
            "1. Answer ONLY using the facts present in the provided context.\n"
            "2. Never use external memory or make up information.\n"
            "3. If the context does NOT contain enough information to answer the user's question, reply strictly with the single word: INSUFFICIENT.\n"
            "4. NEVER provide medical diagnosis, prescribe drugs, or give specific medication dosages.\n"
            "5. Maintain a helpful, empathetic, and clear tone.\n"
            f"6. Provide the response in {language}."
        )

        user_content = f"CONTEXT INFORMATION:\n{context}\n\nUSER QUESTION: {query}"

        # If Anthropic API Client is active and configured
        if self.client:
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=600,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}]
                )
                if response.content and len(response.content) > 0:
                    return response.content[0].text.strip()
            except Exception as e:
                print(f"[Agent] Anthropic API call error: {e}. Switching to RAG synthesis engine.")

        # Fallback Engine: Deterministic Grounded Synthesis from retrieved context
        return self._synthesize_fallback_answer(query, context, language)

    def _synthesize_fallback_answer(self, query: str, context: str, language: str) -> str:
        """Grounded synthesis fallback when external LLM API key is not configured."""
        lines = [line.strip() for line in context.split("\n") if line.strip() and not line.startswith("[Source:")]
        evidence_summary = " ".join(lines[:6])

        if language.lower() == "telugu":
            return (
                f"ఆరోగ్య సమాచారం (విశ్వసనీయ పత్రాల నుండి):\n\n{evidence_summary}\n\n"
                "గమనిక: ఈ సమాచారం అధికారిక WHO/CDC పత్రాల నుండి సేకరించబడింది. వైద్యుల సలహా తీసుకోండి."
            )
        elif language.lower() == "hindi":
            return (
                f"स्वास्थ्य जानकारी (विश्वसनीय दस्तावेजों से):\n\n{evidence_summary}\n\n"
                "नोट: यह जानकारी केवल आधिकारिक WHO/CDC दस्तावेजों पर आधारित है। डॉक्टर से परामर्श लें।"
            )
        else:
            return (
                f"Based on our trusted healthcare knowledge documents:\n\n{evidence_summary}\n\n"
                "Please consult a certified healthcare professional for individualized care."
            )

    def summarize_document(self, file_text: str, filename: str) -> dict:
        """
        Summarizes an uploaded medical document (PDF or TXT):
        - Key findings
        - Values outside normal range
        - Questions to ask a doctor
        - Safety Disclaimer (No Diagnosis)
        """
        if not file_text.strip():
            return {
                "error": "Document contains no readable text.",
                "filename": filename
            }

        prompt = (
            "Analyze the following medical text and provide a structured plain-language summary.\n"
            "Format the output into 3 distinct sections:\n"
            "1. KEY FINDINGS: Clear summary of the document\n"
            "2. VALUES OUTSIDE NORMAL RANGE: Any lab values, blood pressure, or measurements flagged as abnormal (or 'None detected')\n"
            "3. QUESTIONS FOR DOCTOR: 3 helpful questions the patient should ask their doctor\n\n"
            "Do NOT provide any medical diagnosis or treatment prescribing."
        )

        if self.client:
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=700,
                    temperature=0.1,
                    system=prompt,
                    messages=[{"role": "user", "content": f"DOCUMENT NAME: {filename}\n\nTEXT:\n{file_text[:4000]}"}]
                )
                summary_content = response.content[0].text.strip()
                return self._parse_summary_response(summary_content, filename)
            except Exception as e:
                print(f"[Agent] Document summarization LLM call failed: {e}")

        # Local Fallback Summarizer
        return self._local_summarize(file_text, filename)

    def _parse_summary_response(self, text: str, filename: str) -> dict:
        """Parses LLM markdown response into structured format."""
        return {
            "filename": filename,
            "raw_summary": text,
            "disclaimer": "This summary is for informational guidance only and does not constitute a medical diagnosis."
        }

    def _local_summarize(self, text: str, filename: str) -> dict:
        """Fallback rule-based summarizer for offline environments."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        key_findings = " ".join(paragraphs[:3]) if paragraphs else text[:500]
        
        # Simple extraction for lab numbers
        abnormal_vals = []
        if "140/90" in text or "high blood pressure" in text.lower():
            abnormal_vals.append("Elevated blood pressure indicators detected (>140/90 mmHg)")
        if "fasting blood sugar" in text.lower() or "mg/dl" in text.lower():
            abnormal_vals.append("Blood glucose measurements recorded")

        return {
            "filename": filename,
            "raw_summary": (
                f"### KEY FINDINGS\n{key_findings}\n\n"
                f"### VALUES OUTSIDE NORMAL RANGE\n" + ("\n".join([f"- {v}" for v in abnormal_vals]) if abnormal_vals else "- No immediate out-of-range values identified in excerpt.") + "\n\n"
                f"### QUESTIONS FOR YOUR DOCTOR\n"
                f"1. What do these test results mean for my overall health?\n"
                f"2. Are there lifestyle or diet modifications recommended based on this document?\n"
                f"3. When should I schedule a follow-up checkup?"
            ),
            "disclaimer": "This summary is for informational guidance only and does not constitute a medical diagnosis."
        }

# Singleton Agent Instance
healthcare_agent = HealthcareAgent()
