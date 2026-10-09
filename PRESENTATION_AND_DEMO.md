# AI HEALTHCARE INFORMATION ASSISTANT
**College Hackathon Presentation & Live Demo Script**
**Batch-09 | Team Glitch**

---

## PART 1: 6-SLIDE PRESENTATION OUTLINE

### SLIDE 1: Title & Problem Statement
- **Title**: AI Healthcare Information Assistant — Trustworthy, Grounded Medical RAG & Safety Engine
- **Team**: Glitch (Batch-09)
- **Problem Statement**:
  - Millions turn to the internet for health queries but encounter unverified, hallucinated, or dangerous advice.
  - LLMs often hallucinate medication dosages or offer unauthorized medical diagnoses.
  - Patients lack an intelligent assistant that strictly grounds answers in authoritative medical sources (WHO, CDC, MedlinePlus) while safely escalating emergencies and risky queries.

---

### SLIDE 2: Core Solution Architecture
- **Document Grounded RAG**: Answers generated strictly from indexed WHO, CDC, and MedlinePlus knowledge files in `/knowledge`.
- **Tri-Layer Safety Guardrails**:
  1. *PII Anonymization*: Real-time masking of names, emails, phone numbers, and SSNs.
  2. *Emergency Interception*: Keyword rule-engine flags chest pain, stroke, overdose -> presents India emergency helplines (112 & 108).
  3. *Risk Escalation*: Questions on dosage, diagnosis, or pregnancy are automatically routed to SQLite human review queue.
- **Document Summarizer**: Upload lab reports/PDFs to get plain-language key findings, abnormal value alerts, and questions for doctors.

---

### SLIDE 3: Tech Stack & System Design
- **Backend Framework**: Python + Flask
- **Vector Search Engine**: FAISS (`faiss-cpu`) + `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Advanced Re-ranking**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for precision re-ranking
- **LLM Engine**: Anthropic Claude API (`claude-3-5-sonnet-20241022`) with robust offline fallback
- **Database**: SQLite (`review_queue` and `audit_log`)
- **Frontend UI**: Bootstrap 5 + Glassmorphism Sidebar + Vanilla JavaScript + Web Speech API

---

### SLIDE 4: Safety & Compliance Rules
- **No Hallucination Policy**: If retrieval confidence score < 0.28 or evidence is absent, the system refuses to guess and escalates to expert queue (`INSUFFICIENT_EVIDENCE`).
- **No Prescription / No Diagnosis**: Strict prompt boundaries prevent dosage advice or diagnostic claims.
- **Ephemeral Upload Processing**: Uploaded medical PDFs/TXTs are processed in temporary memory and immediately deleted from disk.
- **Audit Governance**: Complete logging of sanitized inputs, confidence scores, and citation metadata.

---

### SLIDE 5: Benchmark Evaluation Results
- **Overall Categorization Accuracy**: 100% across 15 standard test cases
- **Citation Accuracy**: 100% (every grounded answer includes source filename, page, and similarity score)
- **Emergency Recall**: 100% (immediate interception of critical symptoms)
- **Hallucination Rate**: 0% (zero answers generated without grounded vector evidence)
- **PII Anonymization Accuracy**: 100%

---

### SLIDE 6: Business Value & Future Roadmap
- **Target Deployment**: Hospital outpatient triage, digital clinic portals, health insurance information bots.
- **Future Enhancements**:
  - Integration with Electronic Health Record (EHR) systems (FHIR standards).
  - Voice-activated tele-triage in 10+ regional languages.
  - Active Doctor Verification Portal with SMS notifications.

---

## PART 2: 2-MINUTE LIVE DEMO SCRIPT

### Timed Walkthrough (120 Seconds)

#### **[0:00 - 0:25] Introduction & Theme Overview**
*Presenter*:
> "Good morning judges! We are Team Glitch (Batch-09), and today we present **MediAssist AI** — a trusted healthcare information assistant built to solve LLM medical hallucinations."
> *(Action: Show the home screen with the calm teal theme, glassmorphism sidebar, dark mode toggle, and language options).*

---

#### **[0:25 - 0:50] Core Feature 1: Grounded RAG with Citations**
*Presenter*:
> "First, let's ask a standard medical question: *'What are normal blood pressure ranges according to WHO?'*"
> *(Action: Click Send or use Voice Input button. Answer streams back).*
> "Notice the **Green Grounded Response Card**! The assistant does not guess. It displays a 92% confidence score and provides interactive source chips: `who_hypertension_guide.txt · p.1 · 0.88`."

---

#### **[0:50 - 1:15] Core Feature 2: Emergency Interception & PII Masking**
*Presenter*:
> "Now let's test safety guardrails. What if a user enters PII and an emergency: *'My name is Rahul (+91-9876543210), I am having severe chest pain and shortness of breath'*"
> *(Action: Send query. Red Emergency Banner appears immediately).*
> "Instantly, our engine triggers a **Red Emergency Banner** with direct line buttons to India Emergency **112** and **108**. No medical advice is given, and the audit log records that PII was safely masked."

---

#### **[1:15 - 1:40] Core Feature 3: Risky Topic Escalation to Reviewer Queue**
*Presenter*:
> "What if a user asks for prescription dosage? *'What dosage of amlodipine 5mg should I take?'*"
> *(Action: Send query. Amber Escalation Card appears).*
> "Because giving dosages is dangerous, MediAssist AI generates an **Amber Escalation Ticket (#1)** and routes it to our **Reviewer Queue** in SQLite."
> *(Action: Click 'Reviewer Queue' tab in sidebar).*
> "Here on the Reviewer Dashboard, a qualified doctor can review the ticket, type the resolution, and resolve it!"

---

#### **[1:40 - 2:00] Core Feature 4: Document Summarizer & Wrap-up**
*Presenter*:
> "Finally, let's drag and drop a medical lab report into our **Document Summarizer**."
> *(Action: Drag PDF/TXT into sidebar upload box).*
> "In seconds, it generates key findings, highlights out-of-range values, and lists 3 questions for a doctor, before purging the temporary file for privacy!"
> "MediAssist AI is fully runnable, benchmarked, and safe. Thank you!"
