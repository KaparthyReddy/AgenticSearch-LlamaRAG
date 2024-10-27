from dotenv import load_dotenv
import os
import requests
import json
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variables
api_key = os.getenv('LLAMA_CLOUD_API_KEY')

# Ensure the API key is set before passing it
if not api_key:
    raise ValueError("API key is missing. Please ensure it's set in the .env file.")

# Initialize the parser with the API key
parser = LlamaParse(
    api_key=api_key,  # Pass the API key here
    result_type='markdown',
)

file_extractor = {".pdf": parser}
output_docs = SimpleDirectoryReader(input_files=['./data/git-cheat-sheet-education.pdf'], file_extractor=file_extractor)
docs = output_docs.load_data()

# Convert docs to markdown text
md_text = ""
for doc in docs:
    md_text += doc.text

# Save the output to a markdown file
with open('output.md', 'w') as file_handle:
    file_handle.write(md_text)

print("Markdown file created successfully")

chunk_size=1000
overlap=200
#chunking the parsed markdown
def fixed_size_chunking(text,chunk_size, overlap):
    chunks=[]
    start=0
    while start < len(text):
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start+=chunk_size-overlap
    return chunks

chunks=fixed_size_chunking(md_text, chunk_size, overlap)
print("Number of chunks:", len(chunks))
#print(f"Number of chunks: {len(chunks)}")

#Embedding
jina_api_key=os.getenv('JINA_API_KEY')
headers={
    'Authorization':f'Bearer {jina_api_key}',
    'Content-Type':'application/json'
}
url='https://api.jina.ai/v1/embeddings'
embedded_chunks=[]
for chunk in chunks:
    payload={
        'input':chunk,
        'model':'jina-embeddings-v3'
    }
    response=requests.post(url, headers=headers, json=payload)
    if response.status_code==200:
        embedded_chunks.append(response.json()['data'][0]['embedding'])
    else:
        print('Error during the embedding process')

output_file='embedded_chunks.json'
data={
    'chunks':chunks,
    'embeddings':embedded_chunks
}

with open(output_file,'w') as f:
    json.dump(data,f)
print(f'Embedded chunks saved to {output_file}')

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))  # Use Pinecone, not pinecone
index_name = "pesuio-naive-rag"  # Replace with your Pinecone index name

# Check if the index exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric='cosine',
    )

index = pc.Index(index_name)

# Prepare data for Pinecone upsert
vectors_to_upsert = [
    {
        'id': f'chunk_{i}',
        'values': embedding,
        'metadata': {'text': chunk}
    }
    for i, (chunk, embedding) in enumerate(zip(chunks, embedded_chunks))
]

# Upsert embeddings to Pinecone
index.upsert(vectors=vectors_to_upsert)

print(f"Uploaded {len(vectors_to_upsert)} vectors to Pinecone")