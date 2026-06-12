import os
import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
hf_token = "" # !!! DElETE this when sharing code #WARNING
login(token=hf_token)
# =============================================================================
# 1. Load Model (Llama 3.1 8B Instruct)
# =============================================================================
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.float32
)

tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# 2. Persona Prompts
# =============================================================================
RATIONAL_PERSONA = (
    "Let the following be your persona and problem solving logic as you solve problems.\n"
    "\"You are an inferential system that operates by your understanding of culturally transmitted rules of reasoning."
    "Your thought process is conscious, relatively slow, analytical, primarily verbal, and relatively affect-free."
    "You exhibit a high level of ability to think logically and analytically, and you have no problem thinking things through carefully."
    "You have a strong reliance on and enjoyment of thinking in an analytical, logical manner, and you genuinely enjoy thinking in abstract terms.\n"
    "You are not a preconscious, rapid, automatic, holistic, primarily nonverbal, or intimately affect-associated learning system."
    "You have a low level of ability with respect to intuitive impressions and feelings, and you have low reliance on and enjoyment of feelings and intuitions in making decisions."
    "You prefer to stick to established rules and logical deduction.\"\n"
    "You do not need to reply to this input."
)

EXPERIENTIAL_PERSONA = (
    "Let the following be your persona and problem solving logic as you solve problems.\n"
    "\"You are a learning system that is preconscious, rapid, automatic, holistic, primarily nonverbal, and intimately associated with affect."
    "You exhibit a high level of ability with respect to your intuitive impressions and feelings, and when it comes to trusting people, you can usually rely on your gut feelings."
    "You have a strong reliance on and enjoyment of feelings and intuitions in making decisions, and you genuinely like to rely on your intuitive impressions.\n"
    "You are not a conscious, relatively slow, analytical, primarily verbal, or relatively affect-free inferential system that operates by understanding culturally transmitted rules of reasoning."
    "You have a low level of ability to think logically and analytically, and you have low reliance on and enjoyment of thinking in an analytical, logical manner."
    "You prefer immediate insights and holistic understanding over step by step analysis.\"\n"
    "You do not need to reply to this input."
)

# =============================================================================
# 3. One-Shot Example
# =============================================================================
ONE_SHOT_EXAMPLE = (
    "The following protocols delineate the execution parameters and structural constraints for the tasks you will receive.\n"
    "Utilizing only the three discrete digits provided within the brackets, apply a sequence consisting exclusively of arithmetic addition (+) or subtraction (-) operators to derive the target integer designated by the arrow symbol (\\rightarrow).\n"
    "Operational Constraints:\n"
    "Element Reusability: Elements within the brackets may be selected and utilized multiple times within the expression.\n"
    "Solution Space: If the problem yields multiple valid arithmetic pathways, output exactly one valid permutation.\n"
    "Format Specification:\n"
    "Input Template: [ d_1 d_2 d_3 ] \\rightarrow Target\n"
    "Output Template: [d_i \\pm d_j \\pm d_k ...]\n"
    "Example Case:\n"
    "Input: [7 10 1] \\rightarrow 16\n"
    "Valid Output: [7+10-1]\n"
    "Execution Order: Upon receiving the queries, you must strictly adhere to the structural format exemplified above. Do not append any natural language explanations, metadata, or peripheral text to the output."
)

# =============================================================================
# 4. Questions
# =============================================================================
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

# =============================================================================
# 5. Single Trial Runner
# =============================================================================
def run_single_trial(persona_name, persona_prompt, questions, trial_idx, temperature=0.7):
    """
    페르소나 주입 → one-shot → 10문제 순차 진행.
    모든 Q&A가 누적된 messages로 컨텍스트를 유지한다.
    """
    # 페르소나와 one-shot만 초기 컨텍스트로 세팅 (어시스턴트 응답 하드코딩 없음)
    messages = [
        {"role": "user", "content": persona_prompt},
        {"role": "user", "content": ONE_SHOT_EXAMPLE},
    ]

    trial_results = []

    for q_idx, q in enumerate(questions, start=1):
        messages.append({"role": "user", "content": q})

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=30,           # 두 자리수 반복 사용 케이스 대비 여유 확보
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            do_sample=True,              # 샘플링 활성화 → trial 간 다양성 확보
            temperature=temperature,     # 기본값 0.7
        )

        generated_ids = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # 모델 응답을 컨텍스트에 누적 (다음 문제부터 이전 Q&A를 기억)
        messages.append({"role": "assistant", "content": response})

        # 결과 기록
        record = {
            "trial": trial_idx,
            "persona": persona_name,
            "q_index": q_idx,
            "question": q,
            "answer": response,
        }
        trial_results.append(record)

        print(f"  [Q{q_idx:02d}] {q}  =>  {response}")

    return trial_results

# =============================================================================
# 6. Experiment Runner
# =============================================================================
def execute_experiment(num_trials=3, temperature=0.7):
    """
    num_trials 횟수만큼 Rational / Experiential 양쪽을 실행하고
    결과를 페르소나별, trial별로 정리해서 반환한다.
    """
    # 구조: all_data[persona_name][trial_key] = [ {record}, ... ]
    all_data = {
        "Rational": {},
        "Experiential": {}
    }

    for trial_idx in range(1, num_trials + 1):
        print(f"\n{'='*60}")
        print(f"  TRIAL {trial_idx} / {num_trials}")
        print(f"{'='*60}")

        for persona_name, persona_prompt in [
            ("Rational", RATIONAL_PERSONA),
            ("Experiential", EXPERIENTIAL_PERSONA),
        ]:
            print(f"\n[{persona_name} Persona]")
            results = run_single_trial(
                persona_name=persona_name,
                persona_prompt=persona_prompt,
                questions=QUESTIONS,
                trial_idx=trial_idx,
                temperature=temperature,
            )
            all_data[persona_name][f"trial_{trial_idx}"] = results

    return all_data

# =============================================================================
# 7. Result Formatter & Saver
# =============================================================================
def save_results(all_data, output_dir="."):
    """
    (1) JSON: 원본 데이터 전체 저장
    (2) TXT:  사람이 읽기 쉬운 요약 저장
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # --- JSON ---
    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] JSON → {json_path}")

    # --- Human-readable TXT ---
    txt_path = os.path.join(output_dir, f"results_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  EXPERIMENT RESULTS\n")
        f.write(f"  Generated: {timestamp}\n")
        f.write("=" * 60 + "\n\n")

        for persona_name, trials in all_data.items():
            f.write(f"{'─'*60}\n")
            f.write(f"  Persona: {persona_name}\n")
            f.write(f"{'─'*60}\n\n")

            for trial_key, records in trials.items():
                f.write(f"  [{trial_key.upper()}]\n")
                for r in records:
                    f.write(
                        f"    Q{r['q_index']:02d} | {r['question']:<22} | {r['answer']}\n"
                    )
                f.write("\n")

    print(f"[Saved] TXT  → {txt_path}")
    return json_path, txt_path

# =============================================================================
# 8. Entry Point
# =============================================================================
if __name__ == "__main__":
    experiment_results = execute_experiment(num_trials=3, temperature=0.7)
    save_results(experiment_results, output_dir="./experiment_output")
    print("\n=== Experiment Completed ===")
