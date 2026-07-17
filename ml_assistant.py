import json
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")


# ── Function definitions ──────────────────────────────────
def get_model_accuracy(model_name: str, dataset: str) -> dict:
    results = {
        ("CNN", "chest_xray"): {"accuracy": 0.94, "f1": 0.93},
        ("ResNet", "pathology"): {"accuracy": 0.91, "f1": 0.90},
        ("LSTM", "sensor_data"): {"accuracy": 0.87, "f1": 0.86},
    }
    key = (model_name, dataset)
    return results.get(key, {"error": f"No results for {model_name} on {dataset}"})


def get_dataset_info(dataset_name: str) -> dict:
    datasets = {
        "chest_xray": {"size": 5856, "classes": 2, "type": "medical imaging"},
        "pathology": {"size": 294912, "classes": 2, "type": "medical imaging"},
        "sensor_data": {"size": 10000, "classes": 5, "type": "sensor fusion"},
    }
    return datasets.get(dataset_name, {"error": f"Dataset {dataset_name} not found"})


# ── Router ────────────────────────────────────────────────
def route_query(query: str):
    query_lower = query.lower()

    if "accuracy" in query_lower or "performance" in query_lower:
        model = "CNN" if "cnn" in query_lower else "ResNet"
        dataset = "chest_xray" if "xray" in query_lower else "pathology"
        result = get_model_accuracy(model, dataset)
        return f"Model performance: {json.dumps(result)}"

    elif "dataset" in query_lower or "data" in query_lower:
        dataset = "chest_xray" if "xray" in query_lower else "sensor_data"
        result = get_dataset_info(dataset)
        return f"Dataset info: {json.dumps(result)}"

    return None


# ── RAG knowledge base ────────────────────────────────────
knowledge_base = [
    "Convolutional neural networks extract features from images using learned filters.",
    "Medical image segmentation divides images into regions of different tissues.",
    "Transfer learning fine-tunes pretrained models on domain-specific data.",
    "Semi-supervised learning combines labeled and unlabeled data during training.",
    "RAG retrieves relevant documents and passes them to an LLM for grounded generation.",
]


def retrieve_knowledge(query: str) -> str:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    model = SentenceTransformer('all-MiniLM-L6-v2')
    kb_embeddings = model.encode(knowledge_base)
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    top_idx = np.argmax(similarities)

    if similarities[top_idx] > 0.3:
        return knowledge_base[top_idx]
    return None


# ── Main assistant ────────────────────────────────────────
SYSTEM_PROMPT = """You are an ML engineering assistant specialising in 
medical imaging and sensor fusion. Answer concisely based on the context provided."""


def assistant(query: str):
    print(f"\nUser: {query}")

    # Step 1 — Try function calling first
    function_result = route_query(query)
    if function_result:
        print(f"[Function call result] {function_result}")
        return

    # Step 2 — Try RAG
    context = retrieve_knowledge(query)
    if context:
        prompt = f"{SYSTEM_PROMPT}\n\nContext: {context}\n\nUser: {query}\nAssistant:"
        result = generator(prompt, max_new_tokens=80, do_sample=False)
        answer = result[0]['generated_text'].split("Assistant:")[-1].strip()
        print(f"[RAG answer] {answer[:150]}")
        return

    # Step 3 — Fallback
    print("[Fallback] I don't have enough information to answer this.")


# Test
assistant("What is the accuracy of CNN on chest xray?")
assistant("Tell me about the sensor data dataset")
assistant("What is transfer learning?")
assistant("What is the weather in Munich?")