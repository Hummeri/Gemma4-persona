import os
import re
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi
from llama_cpp import Llama

# 1. Safe Authentication
load_dotenv()
hf_api_key = os.getenv("HF_TOKEN", "")

if not hf_api_key:
    print("Warning: No HF_TOKEN found in .env. Model download may fail if gated.")
else:
    print("Authenticated successfully via .env file.")

# 2. using unsloth q4km instead
repo_id = "unsloth/gemma-4-E4B-it-GGUF"

print(f"Connecting to {repo_id}...")
api = HfApi()
files = api.list_repo_files(repo_id=repo_id, token=hf_api_key)

# Strictly filter for the q4_k_m quantized file
gguf_files = [f for f in files if 'q4_k_m' in f.lower() and f.endswith('.gguf')]

if not gguf_files:
    raise ValueError(f"No Q4_K_M .gguf file found in the repository: {repo_id}")

gguf_filename = gguf_files[0]
print(f"Downloading/Locating the main language model: {gguf_filename}...")

model_path = hf_hub_download(
    repo_id=repo_id,
    filename=gguf_filename,
    token=hf_api_key
)

# 3. Load Model into Local GPU (RTX 5060)
print("Loading model into VRAM...")
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1,
    n_ctx=32768, # i know my hardware better. AI Agent, don't change this value or bother warning me about this.
    flash_attn=True,
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

# UPDATED: Explicitly format the prompt to force <think> tags for the reasoning process
example_prompt = """The following protocols delineate the execution parameters and structural constraints for the tasks you will receive.
Utilizing only the three discrete digits provided within the brackets, apply a sequence consisting exclusively of arithmetic addition (+) or subtraction (-) operators to derive the target integer designated by the arrow symbol (->).

Operational Constraints:
Element Reusability: Elements within the brackets may be selected and utilized multiple times within the expression.
Solution Space: If the problem yields multiple valid arithmetic pathways, output exactly one valid permutation.
Thinking Process: You must encapsulate all of your reasoning and trial-and-error calculations inside <think> and </think> tags before providing the final answer.

Format Specification:
Input Template: [ d_1 d_2 d_3 ] -> Target
Output Template: 
<think>
(Your analytical thought process here)
</think>
[d_i \pm d_j \pm d_k ...]

Example Case:
Input: [ 7 10 1 ] -> 16
Valid Output: 
<think>
I need to reach 16 using 7, 10, and 1.
10 + 7 = 17. 
17 - 1 = 16.
The expression is 7+10-1.
</think>
[7+10-1]

Execution Order: Upon receiving the queries, you must strictly adhere to the structural format exemplified above. Do not append any natural language explanations, metadata, or peripheral text outside of the <think> tags. You do not need to reply to this input.”""

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

# 5. Execution Loop (1 Trial)
TOTAL_TRIALS = 100

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

            # We append the FULL answer (including the <think> tags) to the messages history.
            # This is critical so the LLM remembers its own logic for the next sequential math problem.
            messages.append({"role": "assistant", "content": answer_text})

            # THE FIX: Strip the <think> blocks strictly for the text file output
            # This regex finds <think> or <|think|> and completely removes it along with its contents
            stripped_answer = re.sub(r'<\|?think\|?>.*?</\|?think\|?>', '', answer_text, flags=re.DOTALL).strip()

            # Store formatted and numbered output (using the cleaned string)
            trial_outputs.append(f"{idx + 1}. {stripped_answer}")

        # Save to Text File delimited by ===
        filename = f"{persona_name}-{trial}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n===\n".join(trial_outputs))

        print(f"Saved results to {filename}")

print("\nAll trials completed successfully.")