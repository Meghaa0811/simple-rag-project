from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_pinecone

pdf_path = "./data/sample.pdf"


def process_pdf(pdf_path):

    pages = read_pdf(pdf_path)

    chunks = chunk_pages(pages)

    embeddings = embed_chunks(chunks)

    store_in_pinecone(chunks, embeddings)

    return {
        "pages": len(pages),
        "chunks": len(chunks)
    }


def run():

    result = process_pdf(pdf_path)

    print(result)


if __name__ == "__main__":
    run()