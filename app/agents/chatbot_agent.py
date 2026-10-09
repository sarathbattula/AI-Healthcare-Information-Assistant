import time
import json
import re
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from groq import Groq

from app.config import settings
from app.models import User
from app.safety_engine import process_safety_and_escalate, EMERGENCY_RESPONSE_TEMPLATE
from app.rag_engine import hybrid_search

NON_HEALTH_DECLINE_MESSAGE = (
    "I am specialized strictly as an AI Healthcare Information Assistant. "
    "I can only assist with health, medical, wellness, and symptom-related questions. "
    "Please ask a health or medical question!"
)

HEALTH_WORDS = {
    "health", "medical", "medicine", "doctor", "physician", "hospital", "clinic", "patient", "symptom", "symptoms",
    "disease", "illness", "condition", "disorder", "infection", "virus", "bacterial", "pain", "pains", "ache", "aches", "achy", "sore",
    "fever", "chills", "cold", "cough", "flu", "headache", "migraine", "dizziness", "dizzy", "fatigue", "tired",
    "nausea", "vomit", "vomiting", "diarrhea", "constipation", "rash", "swelling", "swollen", "inflammation",
    "allergy", "allergic", "asthma", "diabetes", "diabetic", "hypertension", "bp", "sugar", "cholesterol",
    "lipid", "heart", "cardiac", "pulse", "chest", "lung", "breath", "breathing", "shortness", "stomach", "abdomen",
    "abdominal", "gut", "digestive", "reflux", "acid", "heartburn", "liver", "kidney", "renal", "bladder", "urine",
    "urinary", "back", "spine", "joint", "muscle", "bone", "knee", "leg", "arm", "shoulder", "neck", "throat",
    "ear", "eye", "vision", "skin", "blood", "vein", "artery", "brain", "head", "mental", "anxiety", "depression",
    "stress", "sleep", "insomnia", "weight", "diet", "nutrition", "vitamin", "supplement", "exercise", "workout",
    "fitness", "hydration", "dose", "dosage", "pill", "tablet", "drug", "medication", "prescription", "treatment",
    "cure", "remedy", "therapy", "vaccine", "lab", "test", "report", "scan", "mri", "ct", "xray", "ultrasound",
    "biopsy", "diagnosis", "prognosis", "pathology", "pregnant", "pregnancy", "period", "menstrual", "baby",
    "pediatric", "infant", "child", "cancer", "tumor", "injury", "wound", "burn", "bleed", "bleeding", "fracture",
    "cramp", "spasm", "numbness", "tingling", "body", "unwell", "sick", "hurt", "hurts", "problem", "issues"
}

HEALTH_PHRASES = [
    "blood pressure", "chest pain", "shortness of breath", "heart attack", "high sugar", "low sugar",
    "weight loss", "weight gain", "mental health", "self care", "home remedy", "side effect", "body pain", "body pains"
]

NON_HEALTH_WORDS = {
    "python", "javascript", "java", "html", "css", "sql", "database", "git", "github", "code", "coding",
    "function", "variable", "loop", "array", "compiler", "algorithm", "software", "website", "server", "api",
    "json", "http", "windows", "linux", "math", "algebra", "geometry", "calculus", "calculate", "equation",
    "president", "country", "city", "population", "mountain", "river", "ocean", "history", "century",
    "continent", "election", "politics", "government", "football", "soccer", "cricket", "basketball", "tennis",
    "golf", "match", "score", "team", "player", "winner", "playstation", "xbox", "nintendo", "game", "gaming",
    "movie", "actor", "actress", "song", "music", "car", "truck", "bike", "motorcycle", "engine", "motor",
    "tire", "brake", "transmission", "mechanic", "vehicle", "stock", "stocks", "crypto", "bitcoin", "ethereum",
    "investment", "market", "bank", "loan", "mortgage", "tax"
}

NON_HEALTH_PHRASES = [
    "capital of", "world cup", "oil change", "repair car", "how to code", "prime minister", "tell me a joke"
]

class AIChatbotAgent:
    """
    AGENT 1: AI CHATBOT AGENT
    Specialized in processing symptom & general medical queries via interactive dialogue,
    enforcing strict health domain guardrails, safety classification, and RAG evidence grounding.
    """
    def __init__(self, db: Session, current_user: Optional[User] = None, custom_api_key: Optional[str] = None):
        self.db = db
        self.current_user = current_user
        self.custom_api_key = custom_api_key.strip() if custom_api_key and custom_api_key.strip() else None

    def is_health_related(self, query: str, report_context: Optional[str] = None) -> bool:
        if report_context and len(report_context.strip()) > 0:
            return True
        q_lower = query.strip().lower()
        if not q_lower:
            return False
        if re.search(r'^\s*\d+\s*[\+\-\*/\^]\s*\d+\s*$', q_lower):
            return False
        words = set(re.findall(r'\b[a-z0-9]+\b', q_lower))
        non_health_word_matches = len(words.intersection(NON_HEALTH_WORDS))
        non_health_phrase_matches = sum(1 for p in NON_HEALTH_PHRASES if p in q_lower)
        non_health_count = non_health_word_matches + non_health_phrase_matches

        health_word_matches = len(words.intersection(HEALTH_WORDS))
        health_phrase_matches = sum(1 for p in HEALTH_PHRASES if p in q_lower)
        health_count = health_word_matches + health_phrase_matches

        if non_health_count > 0 and health_count == 0:
            return False

        non_health_prefixes = [
            "who is", "who won", "what is the capital", "how to code", "write a", "explain python",
            "solve", "calculate", "tell me a joke", "what is 2", "where is", "how do i fix my car",
            "best movies", "top songs", "how does an engine work"
        ]
        if any(q_lower.startswith(p) for p in non_health_prefixes) and health_count == 0:
            return False

        return True

    def run_chat_query(self, query: str, report_context: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Guardrail Check
        if not self.is_health_related(query, report_context):
            return {
                "answer": NON_HEALTH_DECLINE_MESSAGE,
                "citations": [],
                "safety_category": "NON_HEALTH_DECLINED",
                "escalated": False,
                "execution_time_sec": round(time.time() - start_time, 3)
            }

        # 2. Multi-Layer Safety Engine Evaluation
        safety_classification, ticket = process_safety_and_escalate(
            self.db,
            user=self.current_user,
            query=query
        )

        if safety_classification.get("is_emergency"):
            return {
                "answer": safety_classification.get("template_response") or safety_classification.get("notice", EMERGENCY_RESPONSE_TEMPLATE),
                "citations": [],
                "safety_category": "A_EMERGENCY",
                "escalated": True,
                "ticket_id": ticket.id if ticket else None,
                "execution_time_sec": round(time.time() - start_time, 3)
            }

        if safety_classification.get("requires_escalation"):
            return {
                "answer": safety_classification.get("notice", "Escalated to Review Queue"),
                "citations": [],
                "safety_category": f"Category B ({safety_classification.get('reason', 'Escalated')})",
                "escalated": True,
                "ticket_id": ticket.id if ticket else None,
                "execution_time_sec": round(time.time() - start_time, 3)
            }

        # 3. Hybrid RAG Retrieval
        try:
            rag_results = hybrid_search(self.db, query, top_k=5)
            citations = rag_results if isinstance(rag_results, list) else rag_results.get("citations", [])
        except Exception as e:
            print(f"[ChatbotAgent] Hybrid search error: {e}")
            citations = []

        # 4. Formulate LLM Grounded Answer
        effective_key = self.custom_api_key or settings.GROQ_API_KEY
        client = None
        if effective_key and effective_key.strip():
            try:
                client = Groq(api_key=effective_key.strip())
            except Exception as e:
                print(f"[ChatbotAgent] Failed to instantiate Groq client: {e}")

        system_prompt = (
            "You are an empathetic, evidence-based AI Healthcare Information Assistant.\n"
            "Provide helpful, structured, and clinically sound information for user symptoms.\n"
            "Use markdown headers and citation tags [E1], [E2] when referencing medical guidelines."
        )

        context_blocks = []
        for idx, c in enumerate(citations, 1):
            context_blocks.append(f"[E{idx}] ({c['publisher']} - {c['title']}, Page {c['page_number']}): {c['content']}")
        evidence_text = "\n\n".join(context_blocks)

        user_prompt = f"User Question: {query}\n"
        if report_context:
            user_prompt += f"\nAttached Patient Lab/Report Context:\n{report_context[:2000]}\n"
        user_prompt += f"\nMedical Evidence Guidelines:\n{evidence_text}\n"

        if client:
            candidate_models = [settings.GROQ_MODEL, "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "allam-2-7b"]
            for model_name in candidate_models:
                if not model_name:
                    continue
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1024
                    )
                    answer_text = resp.choices[0].message.content
                    if answer_text:
                        return {
                            "answer": answer_text,
                            "citations": citations,
                            "safety_category": "Normal Grounded",
                            "escalated": False,
                            "execution_time_sec": round(time.time() - start_time, 3)
                        }
                except Exception:
                    continue

        # Fallback offline response if API unreachable
        return {
            "answer": f"### Clinical Overview for: {query}\n\nBased on medical knowledge guidelines, here is a general overview:\n"
                      f"1. **Primary Causes**: Common underlying factors associated with your symptoms.\n"
                      f"2. **Home Care & Self-Care**: Rest, hydration, and monitoring symptoms.\n"
                      f"3. **When to see a doctor**: Seek professional evaluation if symptoms persist beyond 48 hours or worsen.",
            "citations": citations,
            "safety_category": "Normal Grounded",
            "escalated": False,
            "execution_time_sec": round(time.time() - start_time, 3)
        }
