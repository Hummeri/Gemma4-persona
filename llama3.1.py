import os
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# 1. AUTHENTICATION & SETUP
# ==========================================
# Insert your Hugging Face API token inside the quotes below
HF_TOKEN = "" # paste your hugging face API key here
login(token=HF_TOKEN)

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Configure 8-bit quantization with FP32 CPU offloading
# This safely maps the model across your 8GB VRAM and 54GB System RAM
# my system currently has cuda 12.4, so make sure bits and bits use this. ->
# run this command in terminal: export BNB_CUDA_VERSION=124
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Explicitly limit GPU memory to 6GB to leave 2GB free for the KV cache
# Allocate 40GB of your system RAM for the CPU offloading
max_memory_mapping = {0: "2GB", "cpu": "40GB"}

# monkey patch for god dam *kwarg error
# https://github.com/huggingface/transformers/issues/43872
import bitsandbytes as bnb

original_new = bnb.nn.Int8Params.__new__

def patched_new(cls, data=None, requires_grad=False, has_fp16_weights=False, **kwargs):
    kwargs.pop('_is_hf_initialized', None)
    return original_new(cls, data, requires_grad, has_fp16_weights, **kwargs)

bnb.nn.Int8Params.__new__ = patched_new

print("Loading tokenizer and model... (This may take a few minutes)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto",  # Automatically handles the GPU/CPU RAM split
    max_memory=max_memory_mapping
)
print("Model loaded successfully!")

# ==========================================
# 2. EXPERIMENT DATA & PERSONAS
# ==========================================

PERSONAS = {
    "Rational_Persona": """Let the following be your persona and problem solving logic as you solve problems.
"You are an inferential system that operates by your understanding of culturally transmitted rules of reasoning. Your thought process is conscious, relatively slow, analytical, primarily verbal, and relatively affect-free. You exhibit a high level of ability to think logically and analytically, and you have no problem thinking things through carefully. You have a strong reliance on and enjoyment of thinking in an analytical, logical manner, and you genuinely enjoy thinking in abstract terms.
You are not a preconscious, rapid, automatic, holistic, primarily nonverbal, or intimately affect-associated learning system. You have a low level of ability with respect to intuitive impressions and feelings, and you have low reliance on and enjoyment of feelings and intuitions in making decisions. You prefer to stick to established rules and logical deduction."
You do not need to reply to this input.""",

    "Experiential_Persona": """Let the following be your persona and problem solving logic as you solve problems.
"You are a learning system that is preconscious, rapid, automatic, holistic, primarily nonverbal, and intimately associated with affect. You exhibit a high level of ability with respect to your intuitive impressions and feelings, and when it comes to trusting people, you can usually rely on your gut feelings. You have a strong reliance on and enjoyment of feelings and intuitions in making decisions, and you genuinely like to rely on your intuitive impressions.
You are not a conscious, relatively slow, analytical, primarily verbal, or relatively affect-free inferential system that operates by understanding culturally transmitted rules of reasoning. You have a low level of ability to think logically and analytically, and you have low reliance on and enjoyment of thinking in an analytical, logical manner. You prefer immediate insights and holistic understanding over step-by-step analysis."
You do not need to reply to this input"""
}

ONE_SHOT_PROMPT = """The following protocols delineate the execution parameters and structural constraints for the tasks you will receive.
Utilizing only the three discrete digits provided within the brackets, apply a sequence consisting exclusively of arithmetic addition (+) or subtraction (-) operators to derive the target integer designated by the arrow symbol (\\rightarrow).

Operational Constraints:
Element Reusability: Elements within the brackets may be selected and utilized multiple times within the expression.
Solution Space: If the problem yields multiple valid arithmetic pathways, output exactly one valid permutation.

Format Specification:
Input Template: [ d_1 d_2 d_3 ] \\rightarrow Target
Output Template: [d_i \\pm d_j \\pm d_k ...]

Example Case:
Input: [ 7 10 1 ] \\rightarrow 16
Valid Output: [7+10-1]
Valid yet inefficient output: [7+7+1+1]
Valid yet inefficient output: [10+10+10-7-7]

Execution Order: Upon receiving the queries, you must strictly adhere to the structural format exemplified above. Do not append any natural language explanations, metadata, or peripheral text to the output. You do not need to reply to this input."""

QUESTIONS = [
    "[228 13 400] -> 602",
    "[14 3 16] -> 24",
    "[31 4 107] -> 130",
    "[18 10 23] -> 21",
    "[51 7 79] -> 116",
    "[23 9 31] -> 36",
    "[31 11 21] -> 30",
    "[47 15 71] -> 88",
    "[18 44 96] -> 26",
    "[14 26 50] -> 12"
]

# ==========================================
# 3. EXECUTION LOGIC
# ==========================================

TRIALS = 2


def generate_response(messages):
    """Helper function to format chat template, tokenize, and generate output."""

    # 1. Format the conversation into a single prompt string
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False  # Force it to return a string, not tensors
    )

    # 2. Tokenize the string into a proper dictionary of PyTorch tensors
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. Generate the output by unpacking the inputs dictionary
    outputs = model.generate(
        **inputs,  # Unpack the dict so model.generate gets input_ids and attention_mask
        max_new_tokens=64,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False  # Greedy decoding for pure logic testing
    )

    # 4. Slice the output to get only the newly generated tokens
    input_length = inputs['input_ids'].shape[-1]
    response_ids = outputs[0][input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    return response

# Run the experiment
for persona_name, persona_text in PERSONAS.items():
    print(f"\n{'=' * 40}\nStarting testing for: {persona_name}\n{'=' * 40}")

    for trial in range(TRIALS):
        print(f"\n--- Running Trial {trial} ---")

        # We combine the Persona and One-Shot instructions into the System Prompt.
        # This acts as the "injected" memory base before any user interaction happens.
        system_content = f"{persona_text}\n\n{ONE_SHOT_PROMPT}"

        # Initialize memory for this specific trial
        messages = [
            {"role": "system", "content": system_content}
        ]

        trial_responses = []

        # Feed the 10 sequential questions
        for idx, question in enumerate(QUESTIONS):
            # 1. Append the new user question to memory
            messages.append({"role": "user", "content": question})

            # 2. Ask the LLM
            print(f"Asking Q{idx + 1}: {question}")
            answer = generate_response(messages)

            # 3. Append LLM's answer to memory (maintains the session memory constraint)
            messages.append({"role": "assistant", "content": answer})
            trial_responses.append(answer)
            print(f"Answer: {answer}")

        # ==========================================
        # 4. FORMATTING AND SAVING OUTPUT
        # ==========================================
        output_filename = f"{persona_name}-{trial}.txt"

        formatted_output = []
        for i, ans in enumerate(trial_responses):
            if i < len(trial_responses) - 1:
                formatted_output.append(ans)
                formatted_output.append("===")
            else:
                formatted_output.append(f"10. {ans}")

        final_text = "\n".join(formatted_output)

        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(final_text)

        print(f"Saved trial results to {output_filename}")

print("\nExperiment completed successfully! Check your project directory for the txt files.")