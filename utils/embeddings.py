from sentence_transformers import SentenceTransformer

# Initialize embedding model globally to avoid repeated initialization
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = model.encode(chunk["chunk_text"], convert_to_tensor=False).tolist()
        embeddings.append({
            "page_number": chunk["page_number"],
            "chunk_text": chunk["chunk_text"],
            "embedding": embedding
        })
    return embeddings

def embed_query(query_text):
    try:
        # trim if the query is too long
        if len(query_text) > 512:
            query_text = query_text[:512]
        return model.encode(query_text, convert_to_tensor=False).tolist()
    except Exception as e:
        print(f"Error embedding query: {e}")
        # return empty string if embedding fails
        return []