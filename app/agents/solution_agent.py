import re
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from groq import Groq

from app.config import settings

class RecommendedSolutionAgent:
    """
    AGENT 3: RECOMMENDED SOLUTION AGENT
    Specialized in converting symptom descriptions or verified medical report parameters
    into structured, actionable clinical solution plans across 5 key dimensions:
    1. Understanding the Problem (Medical Overview)
    2. Recommended Solutions & Home Care Relief
    3. Dietary & Lifestyle Support
    4. Red Flag Warning Signs (When to seek immediate emergency care)
    5. Questions for Your Doctor (3 targeted consultation questions)
    """
    def __init__(self, db: Session, custom_api_key: Optional[str] = None):
        self.db = db
        self.custom_api_key = custom_api_key.strip() if custom_api_key and custom_api_key.strip() else None

    def generate_solution_plan(self, query_or_report_text: str, filename: Optional[str] = None) -> str:
        effective_key = self.custom_api_key or settings.GROQ_API_KEY
        client = Groq(api_key=effective_key) if effective_key else None

        system_prompt = (
            "You are a Clinical Action Plan & Recommended Solution Agent.\n"
            "Given patient symptoms or a verified medical report, generate a clear, highly structured, evidence-based solution plan.\n"
            "MUST INCLUDE EXACTLY THESE 5 MARKDOWN HEADERS:\n\n"
            "### 🧠 Understanding the Problem\n"
            "[Detailed medical overview of potential mechanisms, causes, or report parameters]\n\n"
            "### 🩺 Recommended Solutions & Action Plan\n"
            "[Practical self-care remedies, step-by-step home care, and relief measures]\n\n"
            "### 🥗 Dietary & Lifestyle Support\n"
            "[Actionable nutrition, hydration, sleep, and physical habit modifications]\n\n"
            "### 🚨 Red Flag Warning Signs\n"
            "[Explicit red flag indicators requiring immediate hospital or doctor evaluation]\n\n"
            "### ❓ Questions for Your Doctor\n"
            "[3 targeted, intelligent questions to ask a doctor during follow-up]"
        )

        user_input = query_or_report_text[:6000]
        if filename:
            user_prompt = f"Generate a comprehensive Recommended Solution Plan for Report: '{filename}'\n\nReport Text:\n{user_input}"
        else:
            user_prompt = f"Generate a comprehensive Recommended Solution Plan for Symptoms/Health Problem:\n\n{user_input}"

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
                        max_tokens=1200
                    )
                    solution_text = resp.choices[0].message.content
                    if solution_text and "Understanding the Problem" in solution_text:
                        return solution_text
                except Exception:
                    continue

        # Offline fallback structured solution plan
        return (
            "### 🧠 Understanding the Problem\n"
            "Based on the provided health inquiry or report data, your symptoms indicate potential physiological stress or parameter variations that require structured management.\n\n"
            "### 🩺 Recommended Solutions & Action Plan\n"
            "- **Adequate Rest**: Ensure 7-8 hours of quality sleep to facilitate cellular recovery.\n"
            "- **Symptom Tracking**: Keep a daily log of onset times, severity, and triggering factors.\n"
            "- **Gentle Home Care**: Apply standard non-invasive comfort measures appropriate for your symptoms.\n\n"
            "### 🥗 Dietary & Lifestyle Support\n"
            "- **Hydration**: Drink 2.5 to 3 Liters of clean water daily.\n"
            "- **Balanced Whole Foods**: Emphasize fiber-rich vegetables, lean proteins, and low-sodium whole grains.\n"
            "- **Avoid Triggers**: Minimize refined sugars, processed foods, excessive caffeine, and alcohol.\n\n"
            "### 🚨 Red Flag Warning Signs\n"
            "- Sudden severe chest pain or pressure radiating to arm or jaw.\n"
            "- High fever (>102°F) accompanied by neck stiffness or severe shortness of breath.\n"
            "- Unexplained fainting, confusion, or sudden loss of motor function.\n\n"
            "### ❓ Questions for Your Doctor\n"
            "1. What underlying factors are most likely contributing to these symptoms or lab results?\n"
            "2. Are specific diagnostic tests (such as follow-up blood work or imaging) recommended?\n"
            "3. What monitoring routine should I maintain over the next 2 to 4 weeks?"
        )
