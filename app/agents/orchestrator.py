from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from app.models import User
from app.agents.chatbot_agent import AIChatbotAgent
from app.agents.summarizer_agent import ReportSummarizerAgent
from app.agents.solution_agent import RecommendedSolutionAgent
from app.agents.contact_email_agent import ContactEmailAgent, TARGET_RECIPIENT_EMAIL

class MultiAgentSystem:
    """
    MULTI-AGENT SYSTEM ORCHESTRATOR
    Coordinates 4 purpose-built specialized AI agents:
    1. chatbot_agent: AI Chatbot Agent (Symptom QA, Guardrails, RAG Search)
    2. summarizer_agent: Report Summarization Agent (Lab values, pathology & report summaries)
    3. solution_agent: Recommended Solution Agent (5-section clinical action plans)
    4. contact_email_agent: Contact Us & Email Agent (Inquiry email dispatcher to battulasarath751@gmail.com)
    """
    def __init__(self, db: Session, current_user: Optional[User] = None, custom_api_key: Optional[str] = None):
        self.db = db
        self.current_user = current_user
        self.custom_api_key = custom_api_key

        # Initialize the 4 Purpose-Built Agents
        self.chatbot_agent = AIChatbotAgent(db, current_user=current_user, custom_api_key=custom_api_key)
        self.summarizer_agent = ReportSummarizerAgent(db, user_id=current_user.id if current_user else None, custom_api_key=custom_api_key)
        self.solution_agent = RecommendedSolutionAgent(db, custom_api_key=custom_api_key)
        self.contact_email_agent = ContactEmailAgent(db, target_email=TARGET_RECIPIENT_EMAIL)

    def run_chatbot(self, query: str, report_context: Optional[str] = None) -> Dict[str, Any]:
        """Delegates to Agent 1: AIChatbotAgent"""
        return self.chatbot_agent.run_chat_query(query, report_context=report_context)

    def run_summarize(self, text_or_doc_id: Any, filename: str = "Report.txt", mode: str = "quick") -> Dict[str, Any]:
        """Delegates to Agent 2: ReportSummarizerAgent"""
        if isinstance(text_or_doc_id, int):
            return self.summarizer_agent.summarize_stored_document(doc_id=text_or_doc_id, mode=mode)
        else:
            return self.summarizer_agent.summarize_raw_text(text=str(text_or_doc_id), filename=filename, mode=mode)

    def run_solution_plan(self, query_or_report_text: str, filename: Optional[str] = None) -> str:
        """Delegates to Agent 3: RecommendedSolutionAgent"""
        return self.solution_agent.generate_solution_plan(query_or_report_text=query_or_report_text, filename=filename)

    def run_contact_email(self, name: str, user_email: str, category: str, subject: str, message: str) -> Dict[str, Any]:
        """Delegates to Agent 4: ContactEmailAgent (Sends to battulasarath751@gmail.com)"""
        return self.contact_email_agent.process_and_send_email(
            name=name,
            user_email=user_email,
            category=category,
            subject=subject,
            message=message
        )
