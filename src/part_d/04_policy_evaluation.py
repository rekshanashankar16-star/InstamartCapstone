
import sys
import importlib.util
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
POLICY_DIR = BASE_DIR / "data" / "policies"
EVAL_FILE = BASE_DIR / "data" / "raw" / "policy_eval.csv"

sys.path.insert(0, str(BASE_DIR))


# Load the existing RAG assistant
RAG_FILE = BASE_DIR / "src" / "part_d" / "03_policy_rag.py"

spec = importlib.util.spec_from_file_location("policy_rag", RAG_FILE)
rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_module)


POLICY_FILES = [
    "refund_policy.txt",
    "delivery_policy.txt",
    "escalation_policy.txt",
    "payment_policy.txt",
    "privacy_policy.txt"
]


def build_retriever(overlap):
    """Build chunks and a TF-IDF retriever for a chosen overlap."""

    chunk_size = 40
    step_size = chunk_size - overlap

    chunks = []

    for filename in POLICY_FILES:

        text = (POLICY_DIR / filename).read_text(encoding="utf-8")
        words = text.split()

        chunk_number = 1

        for start in range(0, len(words), step_size):

            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                break

            chunks.append({
                "source": filename,
                "text": " ".join(chunk_words)
            })

            chunk_number += 1

            if start + chunk_size >= len(words):
                break

    chunks_df = pd.DataFrame(chunks)

    documents = chunks_df["text"].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    document_vectors = vectorizer.fit_transform(documents)

    def custom_retrieve(query, k=3):

        query_vector = vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            document_vectors
        )[0]

        top_indices = scores.argsort()[::-1][:k]

        results = []

        for index in top_indices:
            results.append({
                "source": chunks_df.iloc[index]["source"],
                "score": float(scores[index]),
                "text": chunks_df.iloc[index]["text"]
            })

        return results

    return custom_retrieve, len(chunks)


def evaluate(overlap, evaluation_data):

    custom_retrieve, total_chunks = build_retriever(overlap)

    answerable = evaluation_data[
        evaluation_data["expected_doc"] != "NONE"
    ]

    unanswerable = evaluation_data[
        evaluation_data["expected_doc"] == "NONE"
    ]

    doc_hits = 0
    fact_hits = 0
    fallback_hits = 0

    failed_questions = []

    print("\n" + "=" * 65)
    print(f"EVALUATION: OVERLAP = {overlap}")
    print("Total policy chunks:", total_chunks)
    print("=" * 65)

    for _, row in evaluation_data.iterrows():

        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        result = rag_module.answer(
            question,
            retrieve_function=custom_retrieve
        )

        answer = result["answer"].strip()

        if expected_doc == "NONE":

            fallback_ok = answer.lower().startswith(
                "i could not find"
            )

            if fallback_ok:
                fallback_hits += 1

            print("\nUNANSWERABLE:", question)
            print("Answer:", answer)
            print("Fallback:", "PASS" if fallback_ok else "FAIL")

            continue

        retrieved_sources = [
            item["source"]
            for item in custom_retrieve(question, k=3)
        ]

        doc_ok = expected_doc in retrieved_sources

        fact_ok = (
            isinstance(expected_fact, str)
            and expected_fact.lower() in answer.lower()
        )

        if doc_ok:
            doc_hits += 1

        if fact_ok:
            fact_hits += 1

        if not doc_ok or not fact_ok:
            failed_questions.append(question)

        print("\nQUESTION:", question)
        print("ANSWER:", answer)
        print("Expected document:", expected_doc)
        print("Document retrieval:", "PASS" if doc_ok else "FAIL")
        print("Expected fact:", expected_fact)
        print("Fact check:", "PASS" if fact_ok else "FAIL")

    print("\n" + "-" * 65)
    print("SUMMARY")
    print("-" * 65)

    print(f"Document retrieval: {doc_hits}/{len(answerable)}")
    print(f"Expected facts: {fact_hits}/{len(answerable)}")
    print(f"Unanswerable fallbacks: {fallback_hits}/{len(unanswerable)}")

    if failed_questions:
        print("\nFAILED ANSWERABLE QUESTIONS:")
        for question in failed_questions:
            print("-", question)
    else:
        print("\nAll answerable questions passed.")

    return {
        "overlap": overlap,
        "document_hits": doc_hits,
        "fact_hits": fact_hits,
        "fallback_hits": fallback_hits,
        "failed_questions": failed_questions
    }


if __name__ == "__main__":

    evaluation_data = pd.read_csv(EVAL_FILE).fillna("")

    print("TOTAL EVALUATION QUESTIONS:", len(evaluation_data))

    result_overlap_10 = evaluate(10, evaluation_data)

    result_overlap_0 = evaluate(0, evaluation_data)

    print("\n" + "=" * 65)
    print("OVERLAP COMPARISON")
    print("=" * 65)

    print(
        "Overlap 10:",
        result_overlap_10["document_hits"], "/12 documents,",
        result_overlap_10["fact_hits"], "/12 facts,",
        result_overlap_10["fallback_hits"], "/2 fallbacks"
    )

    print(
        "Overlap 0:",
        result_overlap_0["document_hits"], "/12 documents,",
        result_overlap_0["fact_hits"], "/12 facts,",
        result_overlap_0["fallback_hits"], "/2 fallbacks"
    )

    if result_overlap_0["failed_questions"]:
        print("\nQUESTIONS THAT FAILED WITH ZERO OVERLAP:")
        for question in result_overlap_0["failed_questions"]:
            print("-", question)
    else:
        print("\nNo answerable question failed with zero overlap.")
