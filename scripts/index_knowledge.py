import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

def index_docs():
    # 1. Load the documents from your knowledge_base folder
    print("Loading documents...")
    loader = DirectoryLoader('./knowledge_base', glob="./*.md", loader_cls=TextLoader)
    docs = loader.load()

    # 2. Split long text into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # 3. Initialize the embedding model (This is the one you asked about!)
    print("Initializing sentence-transformers/all-MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Create and save the Vector Database
    print("Creating vector database in chroma_db/...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Success! Knowledge base indexed.")

if __name__ == "__main__":
    index_docs()