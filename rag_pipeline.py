from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import numpy as np
import os

# Your document
document = """
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Neural networks are computational models inspired by the human brain.
Deep learning uses multiple layers of neural networks to learn complex patterns.
Convolutional neural networks are widely used for image recognition tasks.
Transfer learning allows models pretrained on large datasets to be fine-tuned on smaller ones.
Embeddings are dense vector representations that capture semantic meaning of data.
RAG combines retrieval with language model generation to produce grounded answers.
Large language models are trained on massive text corpora to predict the next token.
Fine-tuning adapts a pretrained model to a specific domain using additional training data.
Docker containers package applications with their dependencies for consistent deployment.
GitHub Actions automates testing and deployment workflows using CI/CD pipelines.
Medical imaging uses convolutional neural networks to detect diseases in scans.
Sensor fusion combines data from multiple sensors to improve prediction accuracy.
"""

# Step 1 - Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_text(document)
print(f"Number of chunks: {len(chunks)}")

# Step 2 - Embed and store in ChromaDB
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.create_collection("rag_knowledge_base")

print("Embedding chunks...")
embeddings = model.encode(chunks)
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)
print(f"Stored {collection.count()} chunks in ChromaDB")

# Step 3 - Retrieve
def retrieve(query, top_k=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results['documents'][0]

# Step 4 - Generate
generator = pipeline("text-generation", model="distilgpt2")

def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Context: {context}
Question: {query}
Answer:"""
    result = generator(prompt, max_new_tokens=100, do_sample=False)
    return result[0]['generated_text'].split("Answer:")[-1].strip()

# Step 5 - Full RAG pipeline
def rag(query):
    context_chunks = retrieve(query)
    answer = generate_answer(query, context_chunks)
    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}")

rag("how do neural networks learn from data?")
rag("what is used for medical image analysis?")