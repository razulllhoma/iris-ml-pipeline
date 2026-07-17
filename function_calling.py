import json


# Step 1 — Define your functions
def get_model_accuracy(model_name: str, dataset: str) -> dict:
    """Returns accuracy metrics for a given model and dataset."""
    # Simulated database of results
    results = {
        ("CNN", "chest_xray"): {"accuracy": 0.94, "f1": 0.93},
        ("ResNet", "pathology"): {"accuracy": 0.91, "f1": 0.90},
        ("LSTM", "sensor_data"): {"accuracy": 0.87, "f1": 0.86},
    }
    key = (model_name, dataset)
    if key in results:
        return results[key]
    return {"error": f"No results found for {model_name} on {dataset}"}


def get_dataset_info(dataset_name: str) -> dict:
    """Returns information about a dataset."""
    datasets = {
        "chest_xray": {"size": 5856, "classes": 2, "type": "medical imaging"},
        "pathology": {"size": 294912, "classes": 2, "type": "medical imaging"},
        "sensor_data": {"size": 10000, "classes": 5, "type": "sensor fusion"},
    }
    if dataset_name in datasets:
        return datasets[dataset_name]
    return {"error": f"Dataset {dataset_name} not found"}


# Step 2 — Define function schemas for the LLM
function_schemas = [
    {
        "name": "get_model_accuracy",
        "description": "Get accuracy metrics for a ML model on a specific dataset",
        "parameters": {
            "model_name": "string — name of the model e.g. CNN, ResNet, LSTM",
            "dataset": "string — name of the dataset e.g. chest_xray, pathology"
        }
    },
    {
        "name": "get_dataset_info",
        "description": "Get information about a dataset including size and type",
        "parameters": {
            "dataset_name": "string — name of the dataset"
        }
    }
]


# Step 3 — Simple function router
def call_function(function_name: str, arguments: dict):
    if function_name == "get_model_accuracy":
        return get_model_accuracy(**arguments)
    elif function_name == "get_dataset_info":
        return get_dataset_info(**arguments)
    return {"error": "Unknown function"}


# Step 4 — Simulate LLM function calling decision
def simulate_llm_function_call(user_query: str):
    print(f"\nUser query: {user_query}")

    # In production this decision is made by GPT-4/Claude
    # Here we simulate it with simple keyword matching
    if "accuracy" in user_query.lower() or "performance" in user_query.lower():
        # Extract model and dataset from query
        model = "CNN" if "cnn" in user_query.lower() else "ResNet"
        dataset = "chest_xray" if "xray" in user_query.lower() else "pathology"

        function_call = {
            "function": "get_model_accuracy",
            "arguments": {"model_name": model, "dataset": dataset}
        }
    elif "dataset" in user_query.lower() or "data" in user_query.lower():
        dataset = "chest_xray" if "xray" in user_query.lower() else "sensor_data"

        function_call = {
            "function": "get_dataset_info",
            "arguments": {"dataset_name": dataset}
        }
    else:
        print("No relevant function found for this query")
        return

    print(f"LLM decided to call: {function_call['function']}")
    print(f"With arguments: {function_call['arguments']}")

    result = call_function(function_call['function'], function_call['arguments'])
    print(f"Function result: {result}")

    return result


# Test
simulate_llm_function_call("What is the accuracy of CNN on chest xray?")
simulate_llm_function_call("Tell me about the sensor data dataset")
simulate_llm_function_call("What is the weather today?")