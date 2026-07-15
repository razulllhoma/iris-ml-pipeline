from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

def prompt(text, max_tokens=80):
    result = generator(text, max_new_tokens=max_tokens, do_sample=False)
    output = result[0]['generated_text'].strip()
    print(f"\nPrompt: {text}")
    print(f"Output: {output}")
    print("-"*50)

# 1. Zero-shot
prompt("Classify this sentence as positive or negative: 'I love this product'")

# 2. Few-shot
prompt("""Classify sentiment as positive or negative.

'I hate this' -> negative
'This is amazing' -> positive
'I love this product' ->""")

# 3. Chain-of-thought
prompt("""Is 17 a prime number? Think step by step.
Step 1:""")

# 4. System prompt style
prompt("""You are a medical imaging expert. Answer only about medical imaging topics.
Question: What is image segmentation?
Answer:""")