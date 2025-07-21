import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(data_dir: str):
    documents = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")  # Check if data_dir exists
    
    for file_path in Path(data_dir).rglob("*"):
        print(f"Loading file: {file_path}")  # Debugging print statement
        if file_path.is_file():

            try:
                if file_path.suffix.lower() == ".txt":
                    loader = TextLoader(str(file_path))
                    documents.extend(loader.load())
                    
                elif file_path.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                    documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")  # Log file load errors
    if not documents:
        raise ValueError("No documents were loaded from the provided directory.")
    return documents

def chunk_documents(documents):
    # Split documents into chunks for better retrieval
    if not documents:
        raise ValueError("No documents available to chunk.")  # Ensure documents exist
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)
