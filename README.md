# BFSI-Call-Center-AI-Assistant
# AI AGENT FOR CHATBOT
  **1. Executive Summary**
The BFSI Call Center AI Assistant is a production-grade, local-first agentic AI system designed to
handle common Banking, Financial Services, and Insurance (BFSI) queries. It follows a strict
three-tier response architecture prioritizing safety, accuracy, and zero hallucination of financial data.
  **2. Project Scope and Objectives**
Primary Objective: Deliver fast, standardized, and compliant responses to customer inquiries.
• Loan eligibility and application tracking
• EMI schedules and detailed calculations
• Current interest rate schedules
• Payment status and transaction history
• General account support and customer service
  **3. System Architecture (Tiered Response Logic)**
Tier 1 – Dataset Match: Uses semantic similarity (threshold > 0.85). If matched, approved response
is returned directly. Tier 2 – Local SLM: Triggered when Tier 1 fails. Fine-tuned local model
generates professional compliant responses. Tier 3 – RAG Retrieval: Activated for complex queries
requiring policy lookup or specific data breakdown.
  **4. Technical Stack**
• Model: Phi-3-mini-4k-instruct
• Orchestration: LangChain
• Vector Database: ChromaDB
• Embeddings: sentence-transformers/all-MiniLM-L6-v2
• Environment: Windows with WSL2/Ubuntu
  **5. Core Deliverables**
• Alpaca Dataset: 150+ BFSI conversation samples. • Fine-tuned SLM: Specialized for financial
professional tone. • Knowledge Base: Markdown files with policies, EMI formulas, interest rates. •
Guardrails: Prevents guessing numbers and exposing PII.
  **6. Implementation and Safety**
The system follows a Deterministic-First design. It prioritizes dataset-based answers, uses local
model generation only when required, and grounds complex responses in structured knowledge. All
processing runs locally to ensure sensitive financial data never leaves the environment.
