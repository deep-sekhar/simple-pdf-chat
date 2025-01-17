from dotenv import load_dotenv
# Load environment variables
load_dotenv()
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.pdf_extraction import batch_extract_and_save
from utils.text_chunking import chunk_text
from utils.embeddings import generate_embeddings, embed_query
from utils.vector_store import store_embeddings_in_pinecone, query_embeddings_pinecone
from datetime import datetime
import json
from fastapi.encoders import jsonable_encoder

# Initialize FastAPI app
app = FastAPI()

# Ensure necessary directories exist
if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")
if not os.path.exists("parsed_files"):
    os.makedirs("parsed_files")

# Initialize Flan-T5 model for answering queries
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.get("/")
async def root():
    return {"message": "Welcome to the PDF Query API!"}

@app.post("/upload/")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    try:
        # create a unique filename using timestamp, session_id, and original filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # remove spaces and file extension i.e .pdf
        file_name = file.filename.replace(" ", "_").rsplit(".", 1)[0] 
        unique_filename = f"{file_name}_{timestamp}"

        # Save uploaded file
        file_path = f"uploaded_files/{unique_filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract text from the PDF
        parsed_file_path = f"parsed_files/{unique_filename}.json"
        total_pages = batch_extract_and_save(file_path, parsed_file_path)
        print("Total pages extracted:", total_pages)

        # process each page stored as json files in parsed_files directory
        for i in range(1, total_pages + 1):
            index_file_path = f"parsed_files/{unique_filename}_page_{i}.json"
            with open(index_file_path, "r") as f:
                page_data = json.load(f)
                # Chunk text
                chunks = chunk_text([page_data])
                # Generate embeddings for the chunks
                embeddings = generate_embeddings(chunks)
                # Store embeddings in Pinecone
                store_embeddings_in_pinecone(embeddings, unique_filename, i, session_id)

        return {"message": f"File {unique_filename} uploaded and processed successfully."}
    except Exception as e:
        # Log the error
        print(f"Error during file upload: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/query/")
async def query(question: str, session_id: str):
    try:
        # if query size is greater than 512, trim it
        if len(question) > 100:
            question = question[:100]

        # Generate embedding for the user's query
        query_embedding = embed_query(question)

        # Query the Pinecone vector database
        results = query_embeddings_pinecone(query_embedding, session_id)

        # print(results)
        # return {
        #     "results": "working"
        # }

        # Combine chunks as context
        context = "\n".join(
            [f"{res['metadata']['text']}" for res in results["matches"]]
        )

        # print("Context:", context)
        token_count = len(tokenizer.encode(context))
        print("Token count:", token_count)

        # Handle context overflow
        # Maximum input size for the model
        max_tokens = 512  
        if token_count > max_tokens:
            print(f"Context exceeds {max_tokens} tokens. Trimming...")
            trimmed_context = summarize_context(results, max_tokens, tokenizer, model)
        else:
            trimmed_context = context

        print("trimmed_context", trimmed_context)

        # Generate an answer using the local Flan-T5 model
        inputs = tokenizer(f"Query: {question}\nAnswer using context: {trimmed_context}", return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=400)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Answer", answer)

        # Return the answer and citations
        return {
            "query": question,
            "answer": answer,
            "citations": [
                {"page": res["metadata"]["page_number"], "text": res["metadata"]["text"]}
                for res in results["matches"]
            ]
        }
    except Exception as e:
        # Log and return error details
        print(f"Error during query: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

def summarize_context(results, max_tokens, tokenizer, model):
    summarized_chunks = []
    total_length = 0
    try:
        #  we keep adding until we reach the max_tokens limit
        for res in results['matches']:
            text = res['metadata']['text']
            if total_length + len(tokenizer.encode(text)) < max_tokens:
                summarized_chunks.append(text)
                total_length += len(tokenizer.encode(text))
        
        # Combine the summarized chunks
        summarized_context = "\n".join(summarized_chunks)
        return summarized_context
    
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return "No summary generated."