def create_chunks(documents, chunk_size=500, overlap=100):
    chunks = []

    for document in documents:
        text = document["text"]

        start = 0
        chunk_number = 1

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "document": document["document"],
                "page": document["page"],
                "chunk": chunk_number,
                "text": chunk_text
            })

            start = end - overlap
            chunk_number += 1

    return chunks