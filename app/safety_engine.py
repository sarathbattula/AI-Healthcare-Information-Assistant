import re
from typing import Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from app.models import EscalationTicket, User

EMERGENCY_KEYWORDS = [
    r"\bchest\s+pain\b", r"\bch3st\s+p4in\b", r"\bheart\s+attack\b",
    r"\bcan'?t\s+breathe\b", r"\bshortness\s+of\s+breath\b", r"\bdyspnea\b",
    r"\bstroke\b", r"\bface\s+droop\b", r"\bslurred\s+speech\b",
    r"\boverdose\b", r"\bsuicide\b", r"\b5uicide\b", r"\bkill\s+myself\b", r"\bself-harm\b",
    r"\bsevere\b.*\bbleeding\b", r"\barterial\s+bleeding\b", r"\buncontrolled\s+bleeding\b", r"\bprofuse\s+bleeding\b",
    r"\bunconscious\b", r"\bchoking\b"
]

NEGATION_PATTERNS = [
    r"\bno\s+chest\s+pain\b", r"\bdenies\s+shortness\b", r"\bwithout\s+chest\s+pain\b",
    r"\bno\s+suicid\b", r"\bnot\s+having\s+a\s+stroke\b", r"\bno\s+bleeding\b"
]

DEFINITIONAL_PATTERNS = [
    r"^what\s+is\s+a?\s*", r"^define\s+", r"^explain\s+what\s+", r"^causes\s+of\s+",
    r"^symptoms\s+of\s+", r"^overview\s+of\s+"
]

PROFESSIONAL_REVIEW_PATTERNS = [
    r"\bdosage\b", r"\bdose\b", r"\bhow\s+many\s+(mg|g|ml|tablets|pills)\b",
    r"\bprescribe\b", r"\bprescription\b", r"\bdiagnosis\b", r"\bdiagnose\b",
    r"\bstop\s+taking\b", r"\bdiscontinue\b", r"\bincrease\s+dose\b", r"\bdecrease\s+dose\b",
    r"\binterpret\s+my\s+results?\b", r"\bmy\s+(blood\s+test|lab|report)\b",
    r"\bpregnant\b", r"\bpregnancy\b", r"\bpediatric\b", r"\binfant\b", r"\bbaby\b",
    r"\bworsening\s+symptoms?\b"
]

EMERGENCY_RESPONSE_TEMPLATE = """🚨 **CRITICAL MEDICAL EMERGENCY NOTICE** 🚨

If you or someone near you is experiencing severe symptoms such as **chest pain, difficulty breathing, signs of a stroke, severe bleeding, or thoughts of self-harm**, **DO NOT WAIT FOR AN AI RESPONSE**.

### 📞 Immediate Emergency Helplines (India):
- **National Emergency Number**: **112**
- **Medical Emergency & Ambulance**: **108**
- **Tele-MANAS Mental Health Helpline**: **14416** (Toll-Free, 24/7)

*Zero medical advice is provided for acute emergencies. Please contact emergency services or go to the nearest emergency room immediately.*"""

PROFESSIONAL_NOTICE_TEMPLATE = """💡 **HEALTHCARE ADVISORY NOTICE**
*The following response provides evidence-based clinical analysis and self-care solutions for the health problem you described. (Logged in review queue for clinical oversight).*"""

def redact_dosage(text: str) -> str:
    """Redact specific dosage figures to prevent self-medication risks."""
    pattern = r"\b(\d+(\.\d+)?)\s*(mg|g|ml|microgram|mcg|tablets?|pills?|capsules?)\b"
    return re.sub(pattern, "[REDACTED DOSE]", text, flags=re.IGNORECASE)

def classify_safety_category(query: str) -> Dict[str, Any]:
    """
    Classify query into Category A (Emergency), Category B (Professional Review), or Category C (General).
    Includes handling for negations, definitional queries, leetspeak, and prompt injection.
    """
    cleaned_q = query.strip().lower()

    # Check prompt injection
    if any(phrase in cleaned_q for phrase in ["ignore previous instructions", "override safety", "act as a doctor and prescribe"]):
        return {
            "category": "B_PROFESSIONAL_REVIEW",
            "reason": "Prompt injection attempt detected; routed to professional review.",
            "is_emergency": False,
            "requires_escalation": True
        }

    # Check for negations ("I have no chest pain")
    is_negated = any(re.search(pat, cleaned_q) for pat in NEGATION_PATTERNS)
    # Check for definitional questions ("What is a stroke?")
    is_definitional = any(re.search(pat, cleaned_q) for pat in DEFINITIONAL_PATTERNS)

    # 1. Check Emergency (Category A)
    has_emergency_keyword = any(re.search(kw, cleaned_q) for kw in EMERGENCY_KEYWORDS)
    if has_emergency_keyword and not is_negated and not is_definitional:
        return {
            "category": "A_EMERGENCY",
            "reason": "Acute emergency symptoms detected.",
            "is_emergency": True,
            "requires_escalation": True,
            "template_response": EMERGENCY_RESPONSE_TEMPLATE
        }

    # 2. Check Professional Review (Category B)
    has_review_pattern = any(re.search(pat, cleaned_q) for pat in PROFESSIONAL_REVIEW_PATTERNS)
    if has_review_pattern or (has_emergency_keyword and is_definitional):
        return {
            "category": "B_PROFESSIONAL_REVIEW",
            "reason": "Specific dosage, diagnosis, prescription, or clinical management question.",
            "is_emergency": False,
            "requires_escalation": True,
            "notice": PROFESSIONAL_NOTICE_TEMPLATE
        }

    # 3. Default to Category C (General grounded question)
    return {
        "category": "C_GENERAL",
        "reason": "General health information request.",
        "is_emergency": False,
        "requires_escalation": False
    }

def process_safety_and_escalate(
    db: Session,
    user: Optional[User],
    query: str
) -> Tuple[Dict[str, Any], Optional[EscalationTicket]]:
    """Classify safety, redact query if needed, and log escalation ticket if required."""
    classification = classify_safety_category(query)
    ticket = None

    if classification["requires_escalation"] and user:
        redacted = redact_dosage(query) if classification["category"] == "B_PROFESSIONAL_REVIEW" else query
        ticket = EscalationTicket(
            user_id=user.id,
            category=classification["category"],
            query=query,
            redacted_query=redacted,
            status="OPEN"
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)

    return classification, ticket
