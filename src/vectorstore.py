import os
from langchain_community.vectorstores import Chroma
from src.embeddings import get_embedding_model
from langchain.schema import Documen
for wrapping user inputs

DB_PATH = os.path.join(os.path.dirname(__file__), "db")
COLLECTION_NAME = "documents"

def create_vectorstore(documents: list[Document]):
    """
    Create a new Chroma vector store from the provided documents and persist it.
    """
    try:
        os.makedirs(DB_PATH, exist_ok=True)
        embedding_model = get_embedding_model()
        vectordb = Chroma.from_documents(
            documents,
            embedding=embedding_model,
            persist_directory=DB_PATH,
            collection_name=COLLECTION_NAME
        )
        vectordb.persist()
        print(f"Vector store saved at {DB_PATH}.")
        return vectordb
    except Exception as e:
        raise ValueError(f"Failed to create vector store: {str(e)}")

def upsert_documents(documents: list[Document]):
    """
    Embed and add new documents into the existing Chroma collection, then persist updates.
    """
    try:
        os.makedirs(DB_PATH, exist_ok=True)
        embedding_model = get_embedding_model()  # REUSED: get embedding model
        vectordb = Chroma(
            
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=DB_PATH
        )
        vectordb.add_documents(documents)        # ADDED: add new document embeddings
        vectordb.persist()                      # ADDED: persist updated vector store

        return vectordb
    except Exception as e:
        raise ValueError(f"Failed to upsert into vector store: {str(e)}")