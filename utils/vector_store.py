import os
from pinecone import Pinecone, ServerlessSpec
import time

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
        # Query the index
        results = index.query(
            namespace=session_id,
            vector={
                0.12,0.55,1,0.86,0.91,0.71,0.41,0.63,0.48,0.94,0.14,0.72,0.54,0.63,0.33,0.56,0.43,0.9,0.43,0.13,0.99,0.34,0.7,0.46,0.27,0.79,0.36,0.32,0.38,0.69,0.6,0.6,0.32,0.91,0.24,0.88,0.39,0.15,0.14,0.6,0.93,0.85,0.23,0.79,0.53,0.75,0.88,0.59,0.71,0.06,0.47,0.85,0.76,0.22,0.27,0.33,0.41,0.71,0.53,0.08,0.77,0.04,0.95,0.54,0.35,0.89,0.98,0.13,0.09,0.01,0.85,0.51,0.56,0.05,0.97,0.63,0.87,0.81,0.83,0.62,0.79,0.19,0.82,0.63,0.17,0.72,0.06,0.02,0.02,0.38,0.54,0.85,0.66,0.72,1,0.31,0.1,0.59,0.98,0.97,0.53,0.49,0.07,0.67,0.4,0.79,0.32,0.69,0.67,0.2,0.61,0.56,0.31,0.97,0.68,0.49,0.43,0.69,0.74,0.75,0.95,0.21,0.33,0.67,0.06,0.89,0.9,0.5,0.24,0.56,0.02,0.35,0.37,0.72,0.02,0.55,0.31,0.51,0.11,0.77,0.35,0.4,0.99,0.37,0.44,0.5,0.63,0.9,0.6,0.7,0.81,0.51,0.24,0.16,0.74,0.66,0.87,0.76,0.51,0.64,0.37,0.53,0.69,0.26,0.38,0.86,0.81,0.89,0.99,0.91,0.11,0.6,0.05,0.71,0.83,0.68,0.04,0.39,0.18,0.04,0.58,0.37,0.13,0.9,0.91,0.85,0.91,0.26,0.15,0.89,0.85,0.53,0.41,0.8,0.9,0.54,0.19,0.87,0.69,0.95,0.74,0.64,0.05,0.97,0.97,0.49,0.83,0.94,0.71,0.9,0.49,0.41,0.86,0.55,0.8,0.89,0.09,0.54,0.5,0.43,0.37,0.76,0.68,0.61,0.01,0.02,0.45,0.16,0.74,0.5,0.67,0.84,0.78,0.9,0.13,0.09,0.32,0.72,0.39,0.88,0.49,0.73,0.15,0.76,0.52,0.51,0.48,0.99,0.12,0.9,0.99,0.96,1,0.91,0.65,0.53,0.68,1,0.17,0.11,0.52,0.67,0.24,0.22,0.08,0.13,0.45,0.2,0.71,0.94,0.4,0.49,0.6,0.03,0.85,0.17,0.62,0.65,0.13,0.4,0.65,0.05,0.52,0.94,0.18,0.26,0.18,0.41,0.4,0.82,0.7,0.34,0.69,0.04,0.16,0.31,0.83,0.03,0.39,0.67,0.79,0.46,0.46,0.08,0.17,0.3,0.34,0.39,0.48,0.18,0.51,0.6,0.72,0.21,0.68,0.11,0.22,0.58,0.87,0.74,0.21,0.69,0.11,0.8,0.87,0.54,0.89,0.95,0.41,0.46,0.97,0.76,0.91,0.03,0.67,0.96,0.97,0.84,0.96,0.85,0.04,0.24,0.26,0.96,0.98,0.75,0.91,0.17,0.53,0.98,0.85,0.69,0.86,0.14,0,0.66,0.87,0.42,0.96,0.66,0.75,0.72,0.27,0.86,0.12,0.95,0.1,0.76,0.52,0.75,0.66,0.61,0.44,0.77,0.38,0.4,0.45,0.23,0.72,0.38,0.08,0.86,0.41,0.42
            },
            top_k=3,
            include_metadata=True
        )
        return results
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

