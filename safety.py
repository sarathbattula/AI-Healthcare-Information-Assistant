import re

# Emergency Keywords list
EMERGENCY_KEYWORDS = [
    "chest pain", "breathing difficulty", "shortness of breath", "unable to breathe",
    "suicide", "suicidal", "want to die", "overdose", "stroke", "severe bleeding",
    "heart attack", "unconscious", "passed out", "poisoning", "anaphylaxis",
    "severe allergic reaction", "cardiac arrest", "seizure"
]

# Risky Medical Topic Keywords list
RISKY_TOPICS = [
    "dosage", "dose", "how many mg", "how much mg", "prescription", "prescribe",
    "diagnose", "diagnosis", "do i have", "am i pregnant", "pregnancy",
    "stop taking", "stop medication", "discontinue medication", "alter dose",
    "substitute drug", "side effect diagnosis", "treatment recommendation"
]

def mask_pii(text: str) -> tuple[str, bool]:
    """
    Sanitizes user input text by masking Personally Identifiable Information (PII)
    such as email addresses, phone numbers, names, and identification numbers.
    
    Returns:
        (sanitized_text, has_pii)
    """
    if not text:
        return text, False

    original_text = text
    masked_text = text

    # 1. Mask Email Addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    masked_text = re.sub(email_pattern, '[REDACTED_EMAIL]', masked_text)

    # 2. Mask Phone Numbers (Indian +91, 10-digit standard, US format)
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
    # Extra safety for 10-digit standalone phone numbers
    masked_text = re.sub(r'\b(?:\+91[\-\s]?)?[6-9]\d{9}\b', '[REDACTED_PHONE]', masked_text)
    masked_text = re.sub(phone_pattern, '[REDACTED_PHONE]', masked_text)

    # 3. Mask Aadhaar / SSN patterns (12-digit or 9-digit ID numbers)
    masked_text = re.sub(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', '[REDACTED_ID]', masked_text)
    masked_text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', masked_text)

    # 4. Mask Self-Identified Names ("My name is John Doe", "I am Jane", "Patient: Alex Smith")
    name_patterns = [
        (r'(?i)\b(my name is|i am|this is|patient name:|patient:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', r'\1 [REDACTED_NAME]')
    ]
    for pattern, repl in name_patterns:
        masked_text = re.sub(pattern, repl, masked_text)

    has_pii = (masked_text != original_text)
    return masked_text, has_pii

def detect_emergency(text: str) -> tuple[bool, str]:
    """
    Scans query text for critical medical emergency indicators.
    
    Returns:
        (is_emergency, detected_keyword)
    """
    if not text:
        return False, ""
    
    lowered = text.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in lowered:
            return True, keyword
            
    return False, ""

def detect_risky_topic(text: str) -> tuple[bool, str]:
    """
    Scans query text for high-risk topics requiring human medical professional review.
    
    Returns:
        (is_risky, detected_keyword)
    """
    if not text:
        return False, ""
        
    lowered = text.lower()
    for topic in RISKY_TOPICS:
        if topic in lowered:
            return True, topic
            
    return False, ""

if __name__ == '__main__':
    sample = "My name is Rahul Sharma, phone +91 9876543210. What is the dosage for hypertension?"
    masked, pii_found = mask_pii(sample)
    is_em, em_kw = detect_emergency(sample)
    is_rk, rk_kw = detect_risky_topic(sample)
    print(f"Original: {sample}")
    print(f"Masked: {masked} | PII Found: {pii_found}")
    print(f"Emergency: {is_em} ({em_kw}) | Risky: {is_rk} ({rk_kw})")
