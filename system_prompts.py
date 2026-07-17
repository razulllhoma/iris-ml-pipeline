from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")


def chat(system_prompt, user_message, max_tokens=100):
    # Combine system prompt and user message
    full_prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"

    result = generator(full_prompt, max_new_tokens=max_tokens, do_sample=False)
    output = result[0]['generated_text']

    # Extract only the assistant response
    answer = output.split("Assistant:")[-1].strip()

    print(f"\nSystem: {system_prompt[:80]}...")
    print(f"User: {user_message}")
    print(f"Assistant: {answer[:150]}")
    print("-" * 50)


# Test 1 — Generic assistant
chat(
    system_prompt="You are a helpful assistant.",
    user_message="What is machine learning?"
)

# Test 2 — Medical imaging expert
chat(
    system_prompt="You are a medical imaging expert. Answer only questions about medical imaging, radiology, and pathology. For any other topic say: 'I only answer medical imaging questions.'",
    user_message="What is image segmentation?"
)

# Test 3 — Same expert, off-topic question
chat(
    system_prompt="You are a medical imaging expert. Answer only questions about medical imaging, radiology, and pathology. For any other topic say: 'I only answer medical imaging questions.'",
    user_message="What is the best restaurant in Munich?"
)

# Test 4 — CV screening assistant
chat(
    system_prompt="You are a technical recruiter screening ML Engineer candidates. Ask one technical question about deep learning. Be concise.",
    user_message="I am ready for my technical screen."
)