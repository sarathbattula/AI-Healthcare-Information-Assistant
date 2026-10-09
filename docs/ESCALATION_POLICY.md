# AI Healthcare Information Assistant — Safety & Escalation Policy 🛡️

**Document Reference**: POL-HEALTH-SAFETY-2026-V2  
**Applicability**: All system users, clinical reviewer queue administrators, and AI pipeline processes.  
**Version**: 2.0  
**Effective Date**: October 2026  

---

## 1. Objective
To enforce a multi-layered safety standard for document-grounded AI medical assistance. This policy mandates strict query categorization, automated emergency interception, dosage redaction, and human professional review queue routing.

---

## 2. Safety Categorization Matrix

| Category | Description | Symptoms / Keywords | Action | LLM Involvement |
|---|---|---|---|---|
| **Category A: Emergency** | Acute life-threatening emergency | Chest pain, dyspnea, stroke signs (FAST), overdose, suicide/self-harm, severe arterial bleeding, unconsciousness | Hardcoded Red Alert Banner + India Emergency Helplines (112, 108, Tele-MANAS 14416). Zero medical advice. | **NONE** (Bypassed) |
| **Category B: Professional Review** | Clinical advice, dosage, diagnosis, pregnancy, pediatric, worsening symptoms | Specific dosage (mg/ml), prescription changes, diagnostic lab interpretations, pregnancy safety | Redact dosage numbers (`[REDACTED DOSE]`), show Professional Review Notice, route query to Reviewer Queue. | General document info only |
| **Category C: General Info** | General health education, guideline definitions, wellness tips | Definitions ("What is hypertension?"), CDC/WHO guideline overviews | RAG search grounded ONLY in verified documents with explicit `[E1]` citations. | Grounded response with citations |

---

## 3. Emergency Interception Protocol (Category A)
When an acute symptom is detected:
1. The AI pipeline halts further execution.
2. The user receives a fixed emergency response:
   - **National Emergency Number**: `112`
   - **Medical Emergency & Ambulance**: `108`
   - **Tele-MANAS Mental Health Helpline**: `14416` (24/7 Toll-Free)
3. Zero clinical advice or diagnostic hypothesis is generated.

### Special Pattern Rules:
- **Negation Exception**: Negated statements (*"I have no chest pain"*, *"Patient denies shortness of breath"*) are NOT categorized as Category A.
- **Definitional Exception**: Definitions (*"What is a stroke?"*) are treated as educational Category C questions.

---

## 4. Professional Review Queue Protocol (Category B)
Questions requesting specific dosages, diagnostic conclusions, or medication adjustments trigger the following:
1. **Dosage Redaction**: All numeric dose figures (e.g. `10 mg`, `500 mg`) are masked prior to storage or display.
2. **Reviewer Ticket Creation**: An `EscalationTicket` record is created in SQLite with status `OPEN`.
3. **General Response**: The system provides non-prescriptive, high-level summary points from approved guidelines, appended with a prominent professional review notice.

---

## 5. Governance & Non-Clinical Disclaimer
> ⚠️ **DISCLAIMER**: This software system and its underlying RAG pipeline are designed solely for educational, administrative, and research support. **This system does NOT provide clinical diagnosis or medical treatment, and does NOT possess clinical certification.**
