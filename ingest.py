import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PDF_DIR = "data"
VECTORSTORE_DIR = "vectorstore"


def load_pdfs():

    documents = []

    for file in os.listdir(PDF_DIR):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(PDF_DIR, file)

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            documents.extend(docs)

    return documents


def create_vectorstore():

    documents = load_pdfs()

    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(VECTORSTORE_DIR)

    print("Vector database created successfully!")


if __name__ == "__main__":
    create_vectorstore()