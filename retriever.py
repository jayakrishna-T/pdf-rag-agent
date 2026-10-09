from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


VECTORSTORE_DIR = "vectorstore"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)



def retrieve_documents(query):

    documents = vectorstore.similarity_search(
        query,
        k=4
    )

    return documents