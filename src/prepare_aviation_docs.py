
from pathlib import Path
from pypdf import PdfReader
import json

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_DIR / "aviation_docs"
OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "aviation_chunks.json"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []

    for pdf_file in DOCS_DIR.glob("*.pdf"):
        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text or not text.strip():
                continue

            # Split each page into smaller text sections
            words = text.split()
            chunk_size = 200
            overlap = 40
            start = 0

            while start < len(words):
                chunk_text = " ".join(words[start:start + chunk_size])

                if chunk_text.strip():
                    chunks.append({
                        "source": pdf_file.name,
                        "page": page_number,
                        "text": chunk_text
                    })

                start += chunk_size - overlap

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print("Document preparation completed!")
    print(f"Total text chunks: {len(chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
