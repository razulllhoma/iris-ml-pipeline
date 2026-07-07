from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# Load Iris data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)


df['species'] = [iris.target_names[i] for i in iris.target]
print(iris.target_names)
print(df.head())
# Convert each row to a text description
def row_to_text(row):
    return (
        f"sepal length {row['sepal length (cm)']:.1f}cm, "
        f"sepal width {row['sepal width (cm)']:.1f}cm, "
        f"petal length {row['petal length (cm)']:.1f}cm, "
        f"petal width {row['petal width (cm)']:.1f}cm, "
        f"species {row['species']}"
    )

documents = [row_to_text(row) for _, row in df.iterrows()]
print(len(documents))
print(documents[0])
print(documents[1])

# Embed all documents once
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding training data...")
doc_embeddings = model.encode(documents, show_progress_bar=True)
print(len(doc_embeddings))

def find_similar_cases(query_row, top_k=3, threshold=0.85):
    query_text = row_to_text(query_row)
    #print("query_text")
    #print(query_text)
    query_embedding = model.encode([query_text])
    #print("query_embedding")
    # print(query_embedding)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # skip index 0 (itself)

    print(f"\nQuery: {query_text}")
    print("Most similar past cases:")
    for idx in top_indices:
        print(f"  {similarities[idx]:.2f}  |  {documents[idx]}")

# Test with first 3 samples
#for i in range(3):
    #find_similar_cases(df.iloc[i])

find_similar_cases(df.iloc[50])