def chunk_text(pages: list, max_tokens: int = 512):
    chunks = []
    for page in pages:
        page_number = page["page_number"]
        text = page["text"]
        words = text.split()
        
        # Create chunks
        for i in range(0, len(words), max_tokens):
            chunk = " ".join(words[i:i + max_tokens])
            chunks.append({"page_number": page_number, "chunk_text": chunk})
    return chunks
