
import os
import json
from pathlib import Path
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "aviation_chunks.json"

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY is missing. Set it in your terminal first."
    )

client = OpenAI()


def load_chunks():
    """Load the prepared aviation handbook chunks."""
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def retrieve_chunks(question, chunks, top_k=5):
    """Find handbook passages relevant to the question."""
    documents = [chunk.get("text", "") for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english")
    document_vectors = vectorizer.fit_transform(documents)
    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector, document_vectors
    ).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    for index in top_indices:
        if scores[index] > 0:
            results.append(chunks[index])

    return results


def answer_question(question):
    """Generate an answer using retrieved handbook passages."""
    chunks = load_chunks()
    results = retrieve_chunks(question, chunks)

    if not results:
        return "No relevant information was found in the aviation handbook."

    context_parts = []

    for chunk in results:
        source = chunk.get("source", "Aviation handbook")
        page = chunk.get("page", "Unknown")
        text = chunk.get("text", "")

        context_parts.append(
            f"Source: {source}, PDF page: {page}\n{text}"
        )

    context = "\n\n".join(context_parts)

    response = client.responses.create(
        model="gpt-5.5",
        instructions=(
            "You are FlightWise AI, an aviation operations information "
            "assistant. Answer using only the supplied handbook passages. "
            "Explain concepts in simple language. If the passages do not "
            "contain enough information, clearly say so. Do not invent "
            "aviation rules or operational instructions. Include source "
            "filenames and PDF page numbers when available. This handbook "
            "is reference material, not live flight or weather information."
        ),
        input=(
            f"Question: {question}\n\n"
            f"Retrieved handbook passages:\n{context}"
        ),
    )

    return response.output_text


if __name__ == "__main__":
    print("FlightWise AI - Aviation RAG Assistant")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Ask an aviation question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            answer = answer_question(question)
            print("\nAnswer:\n")
            print(answer)
            print("\n" + "-" * 60 + "\n")
        except Exception as error:
            print(f"\nError: {error}\n")
