from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    print("1. Initializing Embedding Model (this may download a model file)...")
    # This uses a small, fast model to test the installation
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("2. Testing ChromaDB creation...")
    # This creates a temporary in-memory database
    vectorstore = Chroma.from_texts(
        texts=["LangChain is awesome!", "ChromaDB is a vector store."],
        embedding=embeddings
    )

    print("3. Testing Search...")
    results = vectorstore.similarity_search("What is ChromaDB?", k=1)
    
    print(f"\nSuccess! Found match: {results[0].page_content}")

except Exception as e:
    print(f"\nAn error occurred: {e}")