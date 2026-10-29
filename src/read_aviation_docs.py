
from pathlib import Path
from pypdf import PdfReader

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_DIR / "aviation_docs"

def read_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    text = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text)

def main():
    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in aviation_docs.")
        return

    for pdf_file in pdf_files:
        print(f"\nReading: {pdf_file.name}")

        try:
            document_text = read_pdf(pdf_file)

            print(f"Extracted characters: {len(document_text)}")
            print("\nFirst 1000 characters:\n")
            print(document_text[:1000])

        except Exception as error:
            print(f"Could not read PDF: {error}")

if __name__ == "__main__":
    main()
