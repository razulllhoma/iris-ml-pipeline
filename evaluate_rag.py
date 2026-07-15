from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rag_pipeline import retrieve, generate_answer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Evaluation dataset
eval_data = [
    {
        "question": "what is semi-supervised learning?",
        "ground_truth": "Semi-supervised learning combines small labeled datasets with large unlabeled ones to reduce the need for labeled data."
    },
    {
        "question": "how does the Skillflow glove work?",
        "ground_truth": "The Skillflow smart glove uses flexible sensor arrays to capture hand gesture data for classification tasks."
    },
    {
        "question": "what is transfer learning?",
        "ground_truth": "Transfer learning fine-tunes pretrained models on domain-specific data to achieve high accuracy with limited samples."
    },
    {
        "question": "what is the capital of France?",
        "ground_truth": "Paris is the capital of France."
    },
]


def evaluate(eval_data):
    print("\nRAG Pipeline Evaluation")
    print("=" * 60)

    scores = []
    for item in eval_data:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Retrieve and generate
        chunks = retrieve(question)

        if chunks is None:
            answer = "I don't have enough information to answer this."
        else:
            answer = generate_answer(question, chunks)

        # Measure answer relevancy using embedding similarity
        answer_emb = model.encode([answer])
        truth_emb = model.encode([ground_truth])
        relevancy = cosine_similarity(answer_emb, truth_emb)[0][0]

        scores.append(relevancy)

        print(f"\nQuestion: {question}")
        print(f"Answer:   {answer[:100]}...")
        print(f"Expected: {ground_truth[:100]}")
        print(f"Relevancy score: {relevancy:.3f}")

    print("\n" + "=" * 60)
    print(f"Average relevancy score: {sum(scores) / len(scores):.3f}")


evaluate(eval_data)