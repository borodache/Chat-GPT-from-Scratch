import json
import psutil
import torch
import urllib.request
from tqdm import tqdm


from gpt import GPTModel
from from_OpenAI import download_and_load_gpt2
# from instruction_fine_tuning import format_input_alpaca
# from instruction_fine_tuning import (test_data, format_input_alpaca, generate, text_to_token_ids, tokenizer, device,
#                                      token_ids_to_text)

""" Code copied from instruction_fine_tuning.py - It wasn't imported to avoid the long training process over there"""

def format_input_alpaca(entry):
    instruction_text = (f"Below is an instruction that describes a task. "
                        f"Write a response that appropriately completes the request."
                        f"\n\n### Instruction:\n{entry['instruction']}")

    input_text = (f"\n\n### Input:\n{entry['input']}" if entry["input"] else "")

    return instruction_text + input_text


BASE_CONFIG = {"vocab_size": 50257, # Vocabulary size
               "context_length": 1024, # Context length
               "drop_rate": 0.0, # Dropout rate
               "qkv_bias": True # Query-key-value bias
               }

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)

model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

with open("instruction-data-with-response.json", "r") as file:
    test_data = json.load(file)

print(test_data[:3])


def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break

    return running


ollama_running = check_if_running("ollama")
if not ollama_running:
    raise RuntimeError(
    "Ollama not running. Launch ollama before proceeding."
    )
print("Ollama running:", check_if_running("ollama"))


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break

            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
        f"Given the input `{format_input_alpaca(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry[json_key]}`"
        f" on a scale from 0 to 100, where 100 is the best score. "
        f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")

print("Finished!")
