import fitz
from pathlib import Path


def read_pdfs(folder):
    documents = []

    files = list(Path(folder).glob("*.pdf"))

    for file in files:
        pdf = fitz.open(file)

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text()

            if text.strip():
                documents.append({
                    "document": file.name,
                    "page": page_number,
                    "text": text
                })

        pdf.close()

    return documents