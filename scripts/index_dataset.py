import json
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def index_dataset():
    # 1. Load the JSON file
    file_path = './data/alpaca_dataset.json'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    # 2. Convert JSON entries into LangChain Documents
    documents = []
    for entry in data:
        # We combine instruction and input for the search 'brain'
        content = f"Question: {entry['instruction']}\nInput: {entry['input']}"
        # We store the output as metadata so we can retrieve it easily
        metadata = {"answer": entry['output']}
        documents.append(Document(page_content=content, metadata=metadata))

    # 3. Initialize the same embedding model
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Add to Chroma DB (This merges with your existing knowledge_base index)
    print(f"Indexing {len(documents)} Q&A pairs into chroma_db...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Success! BFSI Dataset indexed.")

if __name__ == "__main__":
    index_dataset()