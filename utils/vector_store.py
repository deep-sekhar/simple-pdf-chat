import os
from pinecone import Pinecone, ServerlessSpec
import time
import numpy as np

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
index_name = "pdf-embeddings"
# Namespace for vectors
namespace = "unsoiled_assgn" 

# Create Pinecone client
pc = Pinecone(api_key=api_key)

# Create or retrieve an index
if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # Match embedding size of the model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()

# Store embeddings in Pinecone
def store_embeddings_in_pinecone(embeddings, file_name, chunk_id=0, session_id="0"):
    vectors = []
    for idx, emb in enumerate(embeddings):
        unique_id = f"{file_name}_chunk_{chunk_id}_{idx}"
        # print(unique_id, "== >", emb["chunk_text"])
        vectors.append({
            "id": unique_id,
            "values": emb["embedding"],  # Vector embedding
            "metadata": {
                "page_number": emb["page_number"],
                "text": emb["chunk_text"],
                "file_name": file_name
            }
        })
    # Upsert vectors into Pinecone
    index.upsert(vectors=vectors, namespace=session_id)

# Query Pinecone
def query_embeddings_pinecone(query_embedding, session_id="0"):
    try:
        print(index.describe_index_stats())  
        # Query the index
        results = index.query(
            namespace=session_id,
            vector=query_embedding,
            top_k=6,
            include_metadata=True
        )
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

