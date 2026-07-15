import json
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")


def extract_json(text):
    try:
        # Find the first { and last } and extract everything between
        start = text.index('{')
        end = text.rindex('}') + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return None


def structured_prompt(prompt_text, max_tokens=100):
    result = generator(prompt_text, max_new_tokens=max_tokens, do_sample=False)
    output = result[0]['generated_text']

    print(f"\nPrompt: {prompt_text}")
    print(f"Raw output: {output}")

    parsed = extract_json(output)
    if parsed:
        print(f"Parsed JSON: {parsed}")
    else:
        print("Could not parse JSON from output")

    return parsed


# Test 1 — Extract entities
structured_prompt("""Extract information as JSON.
Text: "John Smith is a 35 year old doctor from Munich"
JSON: {"name":""")

# Test 2 — Classify with confidence
structured_prompt("""Classify the sentiment as JSON.
Text: "I love machine learning"
JSON: {"sentiment":""")

# Test 3 — Medical imaging structured output
structured_prompt("""Extract medical imaging info as JSON.
Text: "CNN model achieved 94% accuracy on chest X-ray classification"
JSON: {"model":""")