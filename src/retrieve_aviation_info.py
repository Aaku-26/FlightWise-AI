
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Find the project folder
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "aviation_chunks.json"


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def retrieve_information(question, chunks, top_k=3):
    documents = [chunk.get("text", "") for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    document_vectors = vectorizer.fit_transform(documents)
    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector, document_vectors
    ).flatten()

    top_indices = scores.argsort()[::-1]
    results = []

    for index in top_indices:
        if scores[index] < 0.05:
            break

        results.append({
            "source": chunks[index].get(
                "source", "Aviation handbook"
            ),
            "page": chunks[index].get("page", "Unknown"),
            "score": round(float(scores[index]), 3),
            "text": chunks[index].get("text", "")
        })

        if len(results) >= top_k:
            break

    return results


def generate_simple_answer(question, results):
    if not results:
        return "No relevant information found."

    text = results[0]["text"]
    text = " ".join(text.split())

    if len(text) > 500:
        text = text[:500].rsplit(" ", 1)[0] + "..."

    return text


def main():
    chunks = load_chunks()

    print("FlightWise AI - Free Aviation Information Retrieval")
    print("No OpenAI API credits required.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Ask an aviation question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        results = retrieve_information(question, chunks)

        if not results:
            print("\nNo relevant passages found in the handbook.\n")
            continue

        generate_simple_answer(question, results)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
