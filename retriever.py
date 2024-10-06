from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings

def create_chroma_db(documents, model_name="sentence-transformers/all-MiniLM-L6-v2", persist_dir="your_directory_here"):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    chroma_db = Chroma.from_documents(
        documents, embedding=embedding_model, persist_directory=persist_dir
    )
    return chroma_db

def load_chroma_db(persist_dir, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    chroma_db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    return chroma_db
