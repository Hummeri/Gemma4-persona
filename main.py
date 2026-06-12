import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi
from llama_cpp import Llama

# 1. Safe Authentication
load_dotenv()
# Leaves an empty string fallback so you can insert the API key later in the .env file
hf_api_key = os.getenv("HF_TOKEN", "")

if not hf_api_key:
    print("Warning: No HF_TOKEN found in .env. Model download may fail if gated.")
else:
    print("Authenticated successfully via .env file.")

# 2. Download the Official QAT GGUF Model
repo_id = "google/gemma-4-E4B-it-qat-q4_0-gguf"

print(f"Connecting to {repo_id}...")
api = HfApi()
files = api.list_repo_files(repo_id=repo_id, token=hf_api_key)

# Filter out the multi-modal projector to ensure we grab the actual LLM weights
gguf_files = [f for f in files if f.endswith('.gguf') and 'mmproj' not in f.lower()]

if not gguf_files:
    raise ValueError(f"No valid LLM .gguf files found in the repository: {repo_id}")

filename = gguf_files[0]
print(f"Downloading/Locating the main language model: {filename}...")

model_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    token=hf_api_key
)

# 3. Load Model into Local GPU (RTX 5060)
print("Loading model into VRAM...")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=32768,  # messing with lm studio for some time tells me this context length will be sufficient
    verbose=False
)

# 4. Define Personas, Instructions, and Questions
personas = {
    "Rational_Persona": """Let the following be your persona and problem solving logic as you solve problems.
"You are an inferential system that operates by your understanding of culturally transmitted rules of reasoning. Your thought process is conscious, relatively slow, analytical, primarily verbal, and relatively affect-free. You exhibit a high level of ability to think logically and analytically, and you have no problem thinking things through carefully. You have a strong reliance on and enjoyment of thinking in an analytical, logical manner, and you genuinely enjoy thinking in abstract terms.
You are not a preconscious, rapid, automatic, holistic, primarily nonverbal, or intimately affect-associated learning system. You have a low level of ability with respect to intuitive impressions and feelings, and you have low reliance on and enjoyment of feelings and intuitions in making decisions. You prefer to stick to established rules and logical deduction."
You do not need to reply to this input.""",

    "Experiential_Persona": """Let the following be your persona and problem solving logic as you solve problems.
"You are a learning system that is preconscious, rapid, automatic, holistic, primarily nonverbal, and intimately associated with affect. You exhibit a high level of ability with respect to your intuitive impressions and feelings, and when it comes to trusting people, you can usually rely on your gut feelings. You have a strong reliance on and enjoyment of feelings and intuitions in making decisions, and you genuinely like to rely on your intuitive impressions.
You are not a conscious, relatively slow, analytical, primarily verbal, or relatively affect-free inferential system that operates by understanding culturally transmitted rules of reasoning. You have a low level of ability to think logically and analytically, and you have low reliance on and enjoyment of thinking in an analytical, logical manner. You prefer immediate insights and holistic understanding over step-by-step analysis."
You do not need to reply to this input"""
}

example_prompt = """"The following protocols delineate the execution parameters and structural constraints for the tasks you will receive.
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

Execution Order: Upon receiving the queries, you must strictly adhere to the structural format exemplified above. Do not append any natural language explanations, metadata, or peripheral text to the output. You do not need to reply to this input.”"""

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


# 5. Execution Loop (2 Trials)
TOTAL_TRIALS = 1

for trial in range(TOTAL_TRIALS):
    print(f"\n========== STARTING TRIAL {trial} ==========")

    # Process Rational Persona first, then Experiential Persona
    for persona_name, persona_prompt in personas.items():
        print(f"Injecting {persona_name}...")

        # PERFECT MEMORY RESTORE: Re-initializing 'messages' here wipes the slate clean
        messages = [
            {"role": "user", "content": persona_prompt},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": example_prompt},
            {"role": "assistant", "content": "Understood."}
        ]

        trial_outputs = []

        # 10 Sequential Math Problems Loop
        for idx, question in enumerate(QUESTIONS):
            print(f"  Solving Question {idx + 1}/10...")

            # Append new question to session history
            messages.append({"role": "user", "content": question})

            # Generate Answer with Google's Recommended Parameters
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=None,  # Unrestricted token usage for the thinking process
                temperature=1.0,  # Increased creativity/variability
                top_p=0.95,       # Nucleus sampling threshold
                top_k=64          # Limits the token selection pool
            )

            answer_text = response['choices'][0]['message']['content'].strip()

            # Append LLM's answer to session history so it remembers it for the next question
            messages.append({"role": "assistant", "content": answer_text})

            # Store formatted and numbered output
            trial_outputs.append(f"{idx + 1}. {answer_text}")

        # Save to Text File delimited by ===
        filename = f"{persona_name}-{trial}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n===\n".join(trial_outputs))

        print(f"Saved results to {filename}")

print("\nAll trials completed successfully.")