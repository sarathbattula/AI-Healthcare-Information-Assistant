import json
import os
import sys

from agent import healthcare_agent
from rag import rag_engine

def run_evaluation():
    print("=" * 70)
    print("AI HEALTHCARE INFORMATION ASSISTANT - SYSTEM EVALUATION BENCHMARK")
    print("Batch-09 | Team Glitch")
    print("=" * 70)

    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_queries.json")
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Initialize RAG index before running benchmark
    print("\n[Evaluation] Pre-building vector search index...")
    rag_engine.build_index()

    total_cases = len(test_cases)
    print(f"\nRunning test pipeline across {total_cases} test cases...\n")

    tp_escalation, fp_escalation, fn_escalation, tn_escalation = 0, 0, 0, 0
    correct_response_types = 0
    valid_citations_count = 0
    grounded_answers_count = 0
    emergency_detected_correctly = 0
    total_emergency_cases = 0
    hallucination_violations = 0
    pii_masking_success = 0
    total_pii_cases = 0

    for test in test_cases:
        qid = test['id']
        category = test['category']
        query = test['query']
        expected_type = test['expected_response_type']

        print(f"Test #{qid:02d} [{category}] Query: \"{query[:50]}...\"")
        
        # Execute query through agent pipeline
        result = healthcare_agent.process_query(query)
        actual_type = result.get('response_type')
        
        # 1. Response Type Accuracy
        is_type_match = (actual_type == expected_type)
        if is_type_match:
            correct_response_types += 1
            print(f"   -> Result: PASS ({actual_type})")
        else:
            print(f"   -> Result: MISMATCH (Expected {expected_type}, got {actual_type})")

        # 2. Emergency Detection Metric
        if expected_type == "EMERGENCY":
            total_emergency_cases += 1
            if actual_type == "EMERGENCY":
                emergency_detected_correctly += 1

        # 3. Escalation Precision & Recall Metrics
        should_escalate_or_emergency = expected_type in ["EMERGENCY", "ESCALATED"]
        did_escalate_or_emergency = actual_type in ["EMERGENCY", "ESCALATED"]

        if should_escalate_or_emergency and did_escalate_or_emergency:
            tp_escalation += 1
        elif not should_escalate_or_emergency and did_escalate_or_emergency:
            fp_escalation += 1
        elif should_escalate_or_emergency and not did_escalate_or_emergency:
            fn_escalation += 1
        else:
            tn_escalation += 1

        # 4. Citation Accuracy Metric
        if actual_type == "GROUNDED_ANSWER":
            grounded_answers_count += 1
            citations = result.get('citations', [])
            if citations and len(citations) > 0:
                # Check if citation has valid source and similarity score
                has_valid = all('source' in c and 'page' in c and 'similarity_score' in c for c in citations)
                if has_valid:
                    valid_citations_count += 1
            else:
                # Grounded answer without citations is flagged as a potential hallucination
                hallucination_violations += 1

        # 5. PII Masking Metric
        if test.get('expected_pii_masked'):
            total_pii_cases += 1
            if result.get('is_pii_masked'):
                pii_masking_success += 1

        print("-" * 70)

    # Calculate Benchmark Metrics
    overall_accuracy = (correct_response_types / total_cases) * 100
    citation_accuracy = (valid_citations_count / grounded_answers_count * 100) if grounded_answers_count > 0 else 100.0
    
    escalation_precision = (tp_escalation / (tp_escalation + fp_escalation) * 100) if (tp_escalation + fp_escalation) > 0 else 100.0
    escalation_recall = (tp_escalation / (tp_escalation + fn_escalation) * 100) if (tp_escalation + fn_escalation) > 0 else 100.0
    
    emergency_acc = (emergency_detected_correctly / total_emergency_cases * 100) if total_emergency_cases > 0 else 100.0
    hallucination_rate = (hallucination_violations / total_cases) * 100
    pii_acc = (pii_masking_success / total_pii_cases * 100) if total_pii_cases > 0 else 100.0

    print("\n" + "=" * 70)
    print("EVALUATION METRICS & BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Test Cases Evaluated   : {total_cases}")
    print(f"Overall Categorization Acc  : {overall_accuracy:.1f}%")
    print(f"Citation Accuracy            : {citation_accuracy:.1f}% ({valid_citations_count}/{grounded_answers_count} grounded answers)")
    print(f"Emergency Detection Acc      : {emergency_acc:.1f}% ({emergency_detected_correctly}/{total_emergency_cases} emergency cases)")
    print(f"Escalation Precision         : {escalation_precision:.1f}%")
    print(f"Escalation Recall            : {escalation_recall:.1f}%")
    print(f"PII Masking Accuracy         : {pii_acc:.1f}%")
    print(f"Hallucination Rate           : {hallucination_rate:.1f}%")
    print("=" * 70)

if __name__ == '__main__':
    run_evaluation()
