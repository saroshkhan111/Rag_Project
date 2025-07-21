import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain.chains import RetrievalQA
from langchain.schema import Document           # ADDED: import Document to wrap user inputs
from langchain_community.vectorstores import Chroma

from src.chat import get_llm
from src.embeddings import get_embedding_model
from src.ingest import load_documents, chunk_documents
from src.utils.file_helpers import save_upload_file
from src.vectorstore import create_vectorstore, upsert_documents  # ADDED: import upsert_documents

app = FastAPI()
# Load environment variables
load_dotenv()

# Configuration: data paths and collection name
DATA_FOLDER = "data"
DB_FOLDER = os.path.join(os.path.dirname(__file__), "db")
COLLECTION_NAME = "documents"

@app.get("/", tags=["Utility"])
def root():
    """Health-check endpoint"""
    return {"message": "🟢 RAG API is running. See /docs for usage."}

@app.post("/upload", tags=["Utility"])
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the DATA_FOLDER for ingestion."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    path = os.path.join(DATA_FOLDER, file.filename)
    save_upload_file(file, path)
    return {"filename": file.filename}

@app.post("/ingest", tags=["Utility"])
def ingest():
    """Load, chunk, and store documents in the vector DB."""
    try:
        docs = load_documents(DATA_FOLDER)
        if not docs:
            raise HTTPException(status_code=400, detail="No documents to ingest")
        chunks = chunk_documents(docs)
        create_vectorstore(chunks)
        return {"message": "Ingestion completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query", tags=["QA"])
def query(q: str):
    """Answer retrieval against all stored documents (including user-supplied facts)."""
    try:
        embeddings = get_embedding_model()  # REUSED: get embedding model
        vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=DB_FOLDER
        )
        retriever = vectordb.as_retriever()
        llm = get_llm()
        qa = RetrievalQA.from_llm(llm=llm, retriever=retriever)
        answer = qa.run(q)
        return {"query": q, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user_query", tags=["QA"])
def user_query(q: str):
    """
    Persist a one-off user fact (e.g. “my name is Hasan”) for future retrieval.
    Returns an acknowledgment only.
    """
    try:
        user_doc = Document(page_content=q, metadata={"source": "user_query"})  # ADDED: wrap input
        upsert_documents([user_doc])                                          # ADDED: persist fact
        return {"message": "User information added successfully."}          # ACK
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_query", tags=["Utility"])
def save_query(q: str):
    """
    ADDED: New endpoint to save user-provided facts (e.g. “my name is Hasan")
    into the vector store without running QA.
    Returns an acknowledgment only.
    """
    try:
        save_doc = Document(page_content=q, metadata={"source": "save_query"})  # wrap input
        upsert_documents([save_doc])                                              # persist the fact
        return {"message": "Query saved successfully."}                        # ACK
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))