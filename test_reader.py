from src.pdf_reader import read_pdfs
from src.chunker import create_chunks
from src.embedder import create_embeddings

documents = read_pdfs("data")

print("Total pages:", len(documents))

chunks = create_chunks(documents)

print("Total chunks:", len(chunks))

embeddings = create_embeddings(chunks)

print("Embeddings created:", len(embeddings))
print("Embedding size:", len(embeddings[0]))