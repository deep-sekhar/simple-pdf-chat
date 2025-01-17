import os
import json
from PyPDF2 import PdfReader

# Batch extract text from a PDF and save to a JSON file
def batch_extract_and_save(file_path, output_file, batch_size=10):
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    file_name = os.path.basename(file_path)

    # Process the PDF in batches
    for i in range(0, total_pages, batch_size):
        if i + batch_size > total_pages:
            batch = reader.pages[i:]
        else:
            batch = reader.pages[i:i + batch_size]
        for page_num, page in enumerate(batch, start=i + 1):
            text = page.extract_text().strip()
            if text:
                extracted_data = {
                    "file_name": file_name,
                    "page_number": page_num,
                    "text": text
                }
                # add page number before ".json" in the output file name 
                output_file_name = f"{output_file.rsplit('.', 1)[0]}_page_{page_num}.json"
                with open(output_file_name, "w") as out_file:
                    json.dump(extracted_data, out_file)

    return total_pages
