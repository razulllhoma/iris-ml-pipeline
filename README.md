# Iris ML Pipeline — Gen AI Portfolio

ML Engineer portfolio project demonstrating end-to-end Gen AI skills.
Built by Homa Rasouli — ML/CV Engineer with 8+ years experience in medical imaging and sensor fusion.

---

## Projects

### 1. Semantic Search
Embedding-based similar case retrieval using sentence-transformers.
Given a new Iris sample, finds the 3 most similar cases from training data using cosine similarity.
The same retrieval principle used in RAG pipelines.

### 2. RAG Pipeline
Retrieval-Augmented Generation pipeline using:
- LangChain text splitter for document chunking
- sentence-transformers for embedding
- ChromaDB as vector database
- Confidence threshold to avoid out-of-domain answers
- Source attribution with metadata
- Interactive query interface
- Evaluation using embedding similarity scoring

Knowledge base: medical imaging, sensor fusion, and ML domain data.

### 3. Prompt Engineering
Examples of key prompting techniques:
- Zero-shot prompting
- Few-shot prompting
- Chain-of-thought prompting
- System prompts for persona and domain restriction

### 4. Function Calling
ML assistant combining:
- Function calling for structured data retrieval (model accuracy, dataset info)
- RAG for conceptual questions
- Fallback for out-of-domain queries

### 5. Fine-Tuning Experiment
Fine-tuned `facebook/opt-125m` using LoRA (Low-Rank Adaptation) on domain-specific
ML and medical imaging data.

**Key Results:**
- Base model: 125,534,208 parameters
- LoRA trainable parameters: 294,912 (0.23% of total)
- Training loss: 9.09 → 4.80 across 3 epochs
- Adapter size: 1.1MB vs 250MB full model

**Why LoRA:**
Instead of updating all 125M weights, LoRA adds small adapter matrices A and B
to the attention layers. Only these adapters are trained — making fine-tuning
feasible on a single GPU without expensive infrastructure.

**When to Fine-Tune vs RAG:**
- Fine-tune when the task is stable and well-defined
- RAG when knowledge changes frequently or source attribution is needed

---

## Tech Stack
Python · PyTorch · sentence-transformers · ChromaDB · LangChain · 
Hugging Face Transformers · PEFT/LoRA · Docker · GitHub Actions · MLflow

---

## Background
Built as part of a structured Gen AI upskilling plan covering:
LLM fundamentals · Embeddings · RAG pipelines · Prompt engineering · 
Function calling · Fine-tuning with LoRA · Production evaluation# Iris ML Pipeline 
