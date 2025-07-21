from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model():
    """Ensure the embedding model is loaded correctly."""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        return embedding_model
    except Exception as e:
        raise ValueError(f"Error loading embedding model: {str(e)}")

