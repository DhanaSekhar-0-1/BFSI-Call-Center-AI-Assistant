import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class BFSIAgent:
    def __init__(self):
        self.db_path = "./chroma_db"
        print("Initializing BFSI Assistant...")

        # --- Tier 1 & 3: Embeddings ---
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)

        # --- Tier 2: Local SLM (Phi-3-mini) ---
        print("Loading Tier 2: Local SLM (Phi-3-mini)...")
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # FIX: Set trust_remote_code=False to use the native, stable library code
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False 
        )
        self.slm_pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def ask(self, query):
        # TIER 1: Dataset Similarity Check
        results = self.db.similarity_search_with_relevance_scores(query, k=1)
        
        if results:
            doc, score = results[0]
            if score > 0.75 and "answer" in doc.metadata:
                print("[Tier 1: Dataset Match]")
                return doc.metadata["answer"]

        # TIER 3: RAG Retrieval (Policy/Rates/EMI)
        keywords = ["rate", "policy", "emi", "interest", "limit", "eligibility", "status"]
        if any(word in query.lower() for word in keywords):
            print("[Tier 3: RAG Retrieval]")
            if results:
                doc, score = results[0]
                if "answer" not in doc.metadata:
                    return f"According to our current policy: {doc.page_content}"

        # TIER 2: Local SLM Generation (Professional Fallback)
        print("[Tier 2: Local SLM Generation]")
        
        # Improved Professional Teller System Prompt
        system_instructions = (
            "You are a professional BFSI virtual assistant. Your tone is formal and helpful. "
            "NEVER invent interest rates or financial numbers. If you do not have specific "
            "data, politely direct the user to a branch or the mobile app."
        )
        
        prompt = f"<|system|>{system_instructions}<|end|>\n<|user|>{query}<|end|>\n<|assistant|>"
        
        outputs = self.slm_pipeline(
            prompt, 
            max_new_tokens=150, 
            do_sample=True, 
            temperature=0.4, # Lower temperature for higher professional consistency
            top_p=0.9
        )
        return outputs[0]['generated_text'].split("<|assistant|>")[-1].strip()

if __name__ == "__main__":
    agent = BFSIAgent()
    print("\n--- BFSI Assistant Ready ---")
    while True:
        user_query = input("\n[You]: ")
        if user_query.lower() in ["exit", "quit"]: break
        print(f"\n[Agent]: {agent.ask(user_query)}")