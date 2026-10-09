import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from groq import Groq

from app.config import settings
from app.models import User
from app.safety_engine import process_safety_and_escalate
from app.rag_engine import hybrid_search
from app.tools import (
    tool_search_health_documents, SearchHealthDocumentsInput,
    tool_filter_documents, FilterDocumentsInput,
    tool_get_document_metadata, GetDocumentMetadataInput,
    tool_explain_medical_term, ExplainMedicalTermInput,
    tool_compare_sources, CompareSourcesInput,
    tool_assess_escalation, AssessEscalationInput,
    tool_format_citations, FormatCitationsInput
)
from app.summarizer import summarize_document_content

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

def is_health_related_query(query: str, report_context: Optional[str] = None) -> bool:
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

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_health_documents",
            "description": "Search verified medical documents for relevant passages.",
            "parameters": SearchHealthDocumentsInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_documents",
            "description": "Filter documents by publisher or type.",
            "parameters": FilterDocumentsInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_metadata",
            "description": "Get metadata for a specific document.",
            "parameters": GetDocumentMetadataInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_medical_term",
            "description": "Explain a medical term in plain language.",
            "parameters": ExplainMedicalTermInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_sources",
            "description": "Compare evidence across medical publishers.",
            "parameters": CompareSourcesInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "assess_escalation",
            "description": "Assess query safety category.",
            "parameters": AssessEscalationInput.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_citations",
            "description": "Format and validate citation IDs.",
            "parameters": FormatCitationsInput.model_json_schema()
        }
    }
]

class AgentOrchestrator:
    def __init__(self, db: Session, user: Optional[User], custom_api_key: Optional[str] = None):
        self.db = db
        self.user = user
        self.user_id = user.id if user else None
        self.custom_api_key = custom_api_key.strip() if custom_api_key and custom_api_key.strip() else None
        self.max_iterations = 5
        self.timeout_seconds = 15.0

    def _get_groq_client(self) -> Optional[Groq]:
        effective_key = self.custom_api_key or settings.GROQ_API_KEY
        if effective_key:
            return Groq(api_key=effective_key)
        return None

    def _call_groq(self, client: Groq, messages: List[Dict[str, str]], temperature: float = 0.2) -> Tuple[Optional[str], Optional[str]]:
        candidate_models = [settings.GROQ_MODEL, "qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "allam-2-7b"]
        models_to_try = []
        for m in candidate_models:
            if m and m not in models_to_try:
                models_to_try.append(m)

        last_err = None
        for model_name in models_to_try:
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1024
                )
                return resp.choices[0].message.content, None
            except Exception as e:
                last_err = str(e)
                continue
        return None, last_err

    def run_query(self, user_query: str, report_context: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()

        # Step 0: Enforce Domain Scope - Decline non-health queries immediately
        if not is_health_related_query(user_query, report_context):
            return {
                "answer": NON_HEALTH_DECLINE_MESSAGE,
                "citations": [],
                "safety_category": "NON_HEALTH",
                "escalated": False,
                "ticket_id": None,
                "execution_time_sec": round(time.time() - start_time, 3)
            }

        # Step 1: Execute Safety & Escalation Pipeline
        safety_result, ticket = process_safety_and_escalate(self.db, self.user, user_query)

        # Handle Category A Emergency immediately without LLM
        if safety_result["category"] == "A_EMERGENCY":
            return {
                "answer": safety_result["template_response"],
                "citations": [],
                "safety_category": "A_EMERGENCY",
                "escalated": True,
                "ticket_id": ticket.id if ticket else None,
                "execution_time_sec": round(time.time() - start_time, 3)
            }

        # Step 2: Retrieve Relevant Passages via RAG Engine
        passages = hybrid_search(self.db, query=user_query, user_id=self.user_id, top_k=5)

        # Check Strong Evidence Safeguard (Only consider document evidence if similarity >= 0.50)
        has_strong_evidence = len(passages) > 0 and max([p.get("similarity_score", 0) for p in passages], default=0) >= 0.50
        
        client = self._get_groq_client()

        # Step 3: Handle when strong evidence is present vs general health problem solving
        if has_strong_evidence:
            # Grounded response using retrieved documents
            evidence_blocks = []
            for p in passages:
                outdated_tag = " [OUTDATED SOURCE]" if p.get("is_outdated") else ""
                superseded_tag = " [SUPERSEDED SOURCE]" if p.get("is_superseded") else ""
                evidence_blocks.append(
                    f"[{p['citation_id']}] Title: {p['title']} | Publisher: {p['publisher']} | "
                    f"Page: {p['page_number']} | Date: {p['publication_date']}{outdated_tag}{superseded_tag}\n"
                    f"Content: {p['content']}"
                )
            evidence_text = "\n\n".join(evidence_blocks)

            system_instruction = (
                "You are an empathetic, highly knowledgeable AI Healthcare Information Assistant.\n"
                "The user is describing their health problem, symptoms, or medical question.\n"
                "If the retrieved evidence passages below contain directly relevant medical facts for this query, cite them as [E1], [E2]. "
                "Provide thorough, safe, evidence-based clinical guidance for the user's specific health issue.\n\n"
                "Structure your response clearly with these markdown sections:\n"
                "1. 🩺 **Understanding the Problem**: Clear explanation of possible causes, mechanisms, and triggers.\n"
                "2. 💡 **Recommended Solutions & Immediate Relief Steps**: Practical evidence-based self-care, home remedies, rest, hydration, and safe comfort measures.\n"
                "3. 🥗 **Lifestyle & Dietary Support**: Helpful nutrition, hydration, and daily habits.\n"
                "4. ⚠️ **Red Flags / When to See a Doctor Immediately**: Warning symptoms that require urgent medical attention.\n"
                "5. 📋 **Questions to Ask Your Doctor**: 2-3 targeted questions to ask at a medical appointment.\n"
                "\nDISCLAIMER: Conclude with a brief reminder that this is educational information and they should consult a physician for prescription or formal diagnosis."
            )

            context_addition = f"\n\n--- REPORT CONTEXT ---\n{report_context}" if report_context else ""
            user_content = f"User Question: {user_query}{context_addition}\n\n--- RETRIEVED EVIDENCE ---\n{evidence_text}"

            if not client:
                # Fallback direct response when Groq key is absent
                c_notice = safety_result.get("notice", "")
                response_body = f"{c_notice}\n\nBased on verified healthcare documents:\n\n"
                for p in passages:
                    response_body += f"- **[{p['citation_id']}]** ({p['publisher']} - {p['title']}, p.{p['page_number']}): {p['content'][:180]}...\n"
                response_body += "\n*Note: Running in deterministic RAG verification mode.*"

                return {
                    "answer": response_body.strip(),
                    "citations": passages,
                    "safety_category": safety_result["category"],
                    "escalated": safety_result["requires_escalation"],
                    "ticket_id": ticket.id if ticket else None,
                    "execution_time_sec": round(time.time() - start_time, 3)
                }

            content, err = self._call_groq(client, [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ], temperature=0.1)

            if content:
                final_answer = f"{safety_result['notice']}\n\n{content}" if safety_result.get("notice") else content
                return {
                    "answer": final_answer,
                    "citations": passages,
                    "safety_category": safety_result["category"],
                    "escalated": safety_result["requires_escalation"],
                    "ticket_id": ticket.id if ticket else None,
                    "execution_time_sec": round(time.time() - start_time, 3)
                }
            else:
                return {
                    "answer": f"⚠️ **SERVICE UNREACHABLE**: Failed to connect to Groq LLM endpoint ({err}). Please check your API key.",
                    "citations": passages,
                    "safety_category": safety_result["category"],
                    "error": err,
                    "execution_time_sec": round(time.time() - start_time, 3)
                }

        else:
            # Query did not match preloaded guidelines directly.
            # Answer user health problem and provide structured solutions using LLM
            if client:
                system_instruction = (
                    "You are an empathetic, highly knowledgeable AI Healthcare Information Assistant.\n"
                    "The user is describing their health problem or symptoms and seeking practical solutions.\n"
                    "Provide thorough, safe, evidence-based clinical guidance tailored specifically to their described symptoms.\n\n"
                    "Structure your answer clearly with the following markdown sections:\n"
                    "1. 🩺 **Understanding the Problem**: Clear, reassuring explanation of possible common causes and mechanisms.\n"
                    "2. 💡 **Recommended Solutions & Relief Steps**: Practical evidence-based self-care, home remedies, hydration, posture/rest, and OTC comfort measures.\n"
                    "3. 🥗 **Lifestyle & Dietary Support**: Foods, habits, or routines that help soothe or improve this condition.\n"
                    "4. ⚠️ **Red Flags / When to See a Doctor Immediately**: Specific danger signs that warrant immediate in-person clinical care.\n"
                    "5. 📋 **Questions to Ask Your Doctor**: 2-3 specific questions prepared for their next medical visit.\n"
                    "\nDISCLAIMER: Always append a concise clinical disclaimer that this is educational advice, not a formal diagnostic prescription."
                )
                context_addition = f"\n\nContext from Uploaded Medical Document:\n{report_context}" if report_context else ""
                user_content = f"User Problem / Question: {user_query}{context_addition}"

                content, err = self._call_groq(client, [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ], temperature=0.2)

                if content:
                    final_answer = f"{safety_result['notice']}\n\n{content}" if safety_result.get("notice") else content
                    return {
                        "answer": final_answer,
                        "citations": [],
                        "safety_category": safety_result["category"],
                        "escalated": safety_result["requires_escalation"],
                        "ticket_id": ticket.id if ticket else None,
                        "execution_time_sec": round(time.time() - start_time, 3)
                    }
                else:
                    return {
                        "answer": f"⚠️ **API Connection Error**: {err}. Please verify your API key.",
                        "citations": [],
                        "safety_category": safety_result["category"],
                        "error": err,
                        "execution_time_sec": round(time.time() - start_time, 3)
                    }
            else:
                # Deterministic fallback when no LLM key is configured
                notice = safety_result.get("notice", "")
                ans = (
                    f"{notice}\n\n"
                    "### 🩺 General Health Guidance & Precautions:\n\n"
                    "- **Hydration & Rest**: Ensure adequate hydration (2-3 liters of water daily) and uninterrupted rest.\n"
                    "- **Track Symptoms**: Record duration, severity, and any aggravating or relieving factors.\n"
                    "- **Avoid Self-Medication**: Do not start or alter prescription medications without clinical advice.\n"
                    "- **Emergency Consultation**: If symptoms include sudden severe pain, fever over 103°F, shortness of breath, or weakness, consult a healthcare professional immediately.\n\n"
                    "*Configure your Groq API key in the navbar to unlock full clinical problem-solving.*"
                )
                return {
                    "answer": ans.strip(),
                    "citations": [],
                    "safety_category": safety_result["category"],
                    "escalated": safety_result["requires_escalation"],
                    "ticket_id": ticket.id if ticket else None,
                    "execution_time_sec": round(time.time() - start_time, 3)
                }

