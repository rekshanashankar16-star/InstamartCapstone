
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "outputs" / "policy_chunks.csv"

# Load policy chunks
chunks_df = pd.read_csv(INPUT_FILE)

# Prepare document text
documents = chunks_df["text"].fillna("").astype(str).tolist()

# Convert policy text into TF-IDF vectors
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

document_vectors = vectorizer.fit_transform(documents)


# Retrieve the most relevant policy chunks
def retrieve(query, k=3):

    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    # Get indices of top matching chunks
    top_indices = similarity_scores.argsort()[::-1][:k]

    results = []

    for index in top_indices:

        results.append({
            "source": chunks_df.iloc[index]["source"],
            "score": float(similarity_scores[index]),
            "text": chunks_df.iloc[index]["text"]
        })

    return results


# Test retrieval
if __name__ == "__main__":

    query = "refund policy late delivery"

    results = retrieve(query, k=3)

    print("QUERY:", query)
    print("\nTOP 3 RETRIEVED POLICY CHUNKS")
    print("-" * 60)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print("Source:", result["source"])
        print("Similarity score:", round(result["score"], 4))
        print("Text:", result["text"])

    # Validate expected retrieval sources
    sources = [result["source"] for result in results]

    print("\nRETRIEVAL VALIDATION")
    print("Top 3 sources:", sources)

    if all(source == "refund_policy.txt" for source in sources):
        print("SUCCESS: All top 3 results came from refund_policy.txt")
    else:
        print("WARNING: Top 3 results are not all from refund_policy.txt")
