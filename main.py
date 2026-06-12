import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv

# 1. Safe Authentication
# Loads the variables from the .env file into the environment
load_dotenv()

# Retrieves the token safely without hardcoding it
hf_api_key = os.getenv("HF_TOKEN")

if hf_api_key:
    login(token=hf_api_key)
    print("Authenticated successfully via .env file.")
else:
    print(
        "Warning: No HF_TOKEN found in .env. Ensure you've authenticated via 'huggingface-cli login' in your terminal.")

# 2. Hardware-Optimized Quantization
# 4-bit NF4 Quantization ensures the model and KV cache fits easily in your 8GB VRAM [cite: 4]
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 3. Model Loading
model_id = "google/gemma-4-E4B-it"

print(f"Downloading and loading {model_id} in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16
)
print("Model loaded successfully into VRAM!\n")

# 4. Interactive Chat Loop
messages = []

print("=== Gemma 4 E4B Interactive Session ===")
print("Type 'exit' or 'quit' to end the session.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Ending session.")
        break

    messages.append({"role": "user", "content": user_input})

    # Instruct apply_chat_template to return a raw string, bypassing the tensor shape bug [cite: 7, 8]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the raw string into standard tensors
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate the response
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # Slice the output to exclude the prompt tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"\nGemma: {response.strip()}\n")

    messages.append({"role": "assistant", "content": response.strip()})