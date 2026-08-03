# Note: using distilgpt2 for local development —
# replace with GPT-4/Claude/Mistral API for production-quality generation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import numpy as np
import os

# Your document
document = """
Convolutional neural networks extract hierarchical features from images by applying learned filters across spatial dimensions.
Medical image segmentation divides an image into regions corresponding to different tissues or pathological structures.
Hematopoietic stem cell classification uses deep learning to distinguish cell types from microscopy images.
Semi-supervised learning reduces the need for labeled data by combining small labeled datasets with large unlabeled ones.
Ladder networks use denoising autoencoders to propagate labels through unlabeled data during training.
Anomaly detection identifies samples that deviate significantly from the learned distribution of normal data.
Sensor fusion combines measurements from multiple sensors to produce more accurate predictions than any single sensor.
The Skillflow smart glove uses flexible sensor arrays to capture hand gesture data for classification tasks.
Edge deployment runs machine learning models directly on embedded hardware without cloud connectivity.
Transfer learning fine-tunes pretrained models on domain-specific data to achieve high accuracy with limited samples.
Proteomics research uses mass spectrometry data to identify and quantify proteins in biological samples.
Cell morphology classification distinguishes healthy from pathological cells based on shape and texture features.
MLflow tracks machine learning experiments including parameters, metrics, and model artifacts.
Docker containers package models with their dependencies for reproducible deployment across environments.
GitHub Actions automates testing and deployment pipelines for machine learning projects.
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
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"source": "knowledge_base", "chunk_index": i}
               for i in range(len(chunks))]
)
print(f"Stored {collection.count()} chunks in ChromaDB")

# Step 3 - Retrieve
def retrieve(query, top_k=3, threshold=1.3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )

    documents = results['documents'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    filtered = [
        (doc, dist, meta)
        for doc, dist, meta in zip(documents, distances, metadatas)
        if dist < threshold
    ]

    if not filtered:
        return None

    print(f"\nRetrieved {len(filtered)} chunks:")
    for doc, dist, meta in filtered:
        print(f"  Distance: {dist:.3f} | Source: {meta['source']} | Chunk: {meta['chunk_index']}")

    return [doc for doc, dist, meta in filtered]
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
import time
import json
from datetime import datetime


def rag(query, log=True):
    start_time = time.time()

    print(f"\nUser: {query}")

    # Retrieve
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "distances", "metadatas"]
    )

    documents = results['documents'][0]
    distances = results['distances'][0]

    # Filter by threshold
    filtered = [
        (doc, dist) for doc, dist in zip(documents, distances)
        if dist < 1.3
    ]

    # Generate
    if not filtered:
        answer = "I don't have enough information to answer this."
        context_chunks = []
    else:
        context_chunks = [doc for doc, dist in filtered]
        answer = generate_answer(query, context_chunks)

    latency = time.time() - start_time
    avg_confidence = sum(dist for doc, dist in filtered) / len(filtered) if filtered else None

    # Log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer[:100],
        "chunks_retrieved": len(filtered),
        "avg_distance": round(avg_confidence, 3) if avg_confidence else None,
        "latency_seconds": round(latency, 3),
        "guardrail_triggered": len(filtered) == 0
    }

    if log:
        with open("rag_logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(f"Answer: {answer[:150]}")
    print(f"Latency: {latency:.2f}s | Chunks: {len(filtered)} | Guardrail: {log_entry['guardrail_triggered']}")

    return answer


# Test
rag("what is transfer learning?")
rag("how does the Skillflow glove work?")
rag("what is the weather in Munich?")
#rag("how do neural networks learn from data?")
#rag("what is used for medical image analysis?")


def retrieve(query, top_k=3, threshold=1.3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    documents = results['documents'][0]
    distances = results['distances'][0]

    # Filter by threshold — remember lower distance = more similar
    filtered = [
        doc for doc, dist in zip(documents, distances)
        if dist < threshold
    ]

    if not filtered:
        return None

    return filtered


def rag(query):
    context_chunks = retrieve(query)

    if context_chunks is None:
        print(f"\nQuestion: {query}")
        print("Answer: I don't have enough information to answer this.")
        return

    answer = generate_answer(query, context_chunks)
    print(f"\nQuestion: {query}")
    print(f"Answer: {answer}")


# Test with relevant and irrelevant queries
#rag("how do neural networks learn from data?")
#rag("what is used for medical image analysis?")
#rag("what is the weather in Munich today?")

#rag("how do you classify stem cells from microscopy images?")
#rag("what sensors does the Skillflow glove use?")
#rag("how do you deploy a model on edge hardware?")
#rag("what is semi-supervised learning?")


def run_interactive():
    print("\n" + "=" * 50)
    print("RAG Pipeline — Domain: ML/CV/Medical Imaging")
    print("Type your question and press Enter.")
    print("Type 'quit' to exit.")
    print("=" * 50)

    while True:
        print()
        query = input("Your question: ").strip()

        if query.lower() == 'quit':
            print("Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        rag(query)


# Replace the hardcoded rag() calls with this
#run_interactive()