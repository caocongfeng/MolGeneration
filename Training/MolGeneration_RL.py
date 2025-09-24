import os
import json
import argparse
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from datasets import Dataset

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

SYSTEM_PROMPT = (
    "You are an expert in info-chemistry, especially in SMILES generation, "
    "professional and rigorous."
)

# -------------------------------
# Data utils
# -------------------------------

def read_cot_data(cot_path: str) -> List[dict]:
    """Read JSONL CoT data with keys: smile, explain, cot."""
    out = []
    with open(cot_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            out.append({"answer": data["smile"], "des": data["explain"], "cot": data["cot"]})
    return out


def build_QA_dict(molecule_data: List[dict], zero_shot: bool = True) -> List[dict]:
    if zero_shot:
        q_tmpl = (
            "You are an expert in info-chemistry, especially in SMILES interpretation and translation. "
            "Could you please provide the SMILES representation based on this molecular description: {} "
            "1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. "
            "2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description."
        )
    else:
        q_tmpl = (
            "You are an expert in info-chemistry, especially in SMILES interpretation and translation. "
            "1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. "
            "2. Double-check it within the <Thinking></Thinking> tags and ensure that the SMILES you generate is completely correct. "
            "3. Make sure your answer is the best possible answer for the molecular description. "
            "Could you please provide the SMILES representation based on this molecular description: {} "
        )
    return [{"question": q_tmpl.format(i["des"]), "answer": i["answer"]} for i in molecule_data]


def get_molecule_questions(ds: Dataset) -> Dataset:
    def _map(x):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["answer"],
        }

    return ds.map(_map)


# -------------------------------
# Parsing helpers
# -------------------------------

ANSWER_RE = re.compile(r"<Answer>(.*?)</Answer>", re.DOTALL)
THINK_RE = re.compile(r"<Thinking>(.*?)</Thinking>", re.DOTALL)


def extract_answer(text: str) -> str:
    m = list(ANSWER_RE.finditer(text))
    return m[-1].group(1).strip() if m else ""


def extract_last_tags(text: str) -> Tuple[str, str]:
    a = list(ANSWER_RE.finditer(text))
    t = list(THINK_RE.finditer(text))
    answer = a[-1].group(1).strip() if a else ""
    thinking = t[-1].group(1).strip() if t else ""
    return answer, thinking


# -------------------------------
# SMILES utils
# -------------------------------

def is_SMILES(smiles: str) -> str | bool:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else False


def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fp1, fp2 = gen.GetFingerprint(mol1), gen.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def exact_match(smiles1: str, smiles2: str) -> float:
    return 1.0 if tanimoto_similarity(smiles1, smiles2) == 1.0 else 0.0


# -------------------------------
# Rewards
# -------------------------------

_WS_ONLY_RE = re.compile(r"^\s*$")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def strict_format_reward_func(completions, **kwargs) -> List[float]:
    responses = [c[0]["content"] for c in completions]
    STRICT_RE = re.compile(r"\A\s*<Thinking>\s*(.*?)\s*</Thinking>\s*<Answer>\s*(.*?)\s*</Answer>\s*\Z", re.DOTALL)
    NEAR_RE = re.compile(r"<Thinking>\s*(.*?)\s*</Thinking>\s*<Answer>\s*(.*?)\s*</Answer>", re.DOTALL)
    HAS_BOTH_RE = re.compile(r"<Thinking>.*?</Thinking>.*?<Answer>.*?</Answer>", re.DOTALL)

    scores = []
    for r in responses:
        t = _normalize(r)
        if STRICT_RE.match(t):
            scores.append(1.0)
        elif NEAR_RE.search(t):
            scores.append(0.5)
        elif HAS_BOTH_RE.search(t):
            scores.append(0.2)
        else:
            scores.append(0.0)
    return scores


def single_tag_occurrence_reward(*, prompts, completions, **kwargs) -> List[float]:
    ans = re.compile(r"<Answer>\s*(.*?)\s*</Answer>", re.DOTALL | re.IGNORECASE)
    thk = re.compile(r"<Thinking>\s*(.*?)\s*</Thinking>", re.DOTALL | re.IGNORECASE)
    out = []
    for c in completions:
        text = c[0]["content"]
        out.append(0.2 if len(list(thk.finditer(text))) == 1 and len(list(ans.finditer(text))) == 1 else 0.0)
    return out


def is_SMILES_reward(completions, **kwargs) -> List[float]:
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_answer(r) for r in responses]
    return [1.0 if is_SMILES(s) else 0.0 for s in extracted]


def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_answer(r) for r in responses]
    pred = [is_SMILES(r) for r in extracted]
    gold = [is_SMILES(a) for a in answer]
    return [0.0 if p is False else 4.0 * exact_match(p, g) for p, g in zip(pred, gold)]


def len_reward(completions, max_len: int = 1200, **kwargs) -> List[float]:
    scores = []
    for c in completions:
        l = len(c[0]["content"])
        scores.append((l / max_len) if l <= max_len else max(0.0, 1 - (l - max_len) / max_len))
    return scores


def soft_format_reward_func(completions, **kwargs) -> List[float]:
    pattern = re.compile(r"<Thinking>.*?</Thinking>\s*<Answer>.*?</Answer>", re.DOTALL)
    matches = [bool(pattern.search(c[0]["content"])) for c in completions]
    return [0.2 if m else 0.0 for m in matches]


def stack_reward(completions, answer, **kwargs) -> List[float]:
    pattern = re.compile(r"<Thinking>.*?</Thinking>\s*<Answer>.*?</Answer>", re.DOTALL)
    rewards: List[float] = []
    max_len = 1200
    for i, c in enumerate(completions):
        resp = c[0]["content"]
        if not pattern.search(resp):
            rewards.append(0.0)
            continue
        pred = extract_answer(resp)
        pred_canon = is_SMILES(pred)
        if not pred_canon:
            rewards.append(0.0)
            continue
        gold_canon = is_SMILES(answer[i])
        em = exact_match(pred_canon, gold_canon) if gold_canon else 0.0
        if em < 1.0:
            rewards.append(0.0)
            continue
        l = len(resp)
        length_bonus = (l / max_len) if l <= max_len else max(0.0, 1 - (l - max_len) / max_len)
        rewards.append(2.0 * em + 1.0 + length_bonus)
    return rewards


# -------------------------------
# Inference helpers
# -------------------------------

def build_test_messages(test_data: List[dict]) -> List[dict]:
    msgs = []
    for i in test_data:
        messages = [
            {"role": "system", "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation, professional and rigorous."},
            {"role": "user", "content": (
                "You are an expert in info-chemistry, especially in SMILES interpretation and translation. "
                f"Could you please provide the SMILES representation based on this molecular description: {i['des']} "
                "1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. "
                "2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description."
            )},
        ]
        msgs.append({"messages": messages})
    return msgs


# -------------------------------
# Main training / evaluation
# -------------------------------

def Qwen_RL(
    projectname: str = "Qwen_base_full_RL",
    wandbname: str = "Qwen_RL_non_conditional_non_KL",
    tags: List[str] | None = None,
    finetune_name: str = "Qwen/Qwen2.5-7B",
    finetune_tags: List[str] | None = None,
    output_dir: str = "Qwen_base_full_RL_KL_softformat_conditional_similarity",
    conditional_reward_style: bool = True,
    KL: float = 0.0,
    temperature: float = 0.7,
    lr: float = 5e-6,
    per_device_train_batch_size: int = 4,
    num_generations: int = 2,
    max_prompt_length: int = 2048,
    max_completion_length: int = 2048,
    num_train_epochs: int = 3,
    save_steps: int = 100,
    zero_shot: bool = True,
    BATCH_SIZE: int = 4,
    test_path: str = "./RL_data/test_data.txt",
) -> None:

    tags = tags or ["Qwen", "RL", "full"]
    finetune_tags = finetune_tags or ["molecule", "full tuning", "RL", "Qwen"]

    wandb.login()
    wandb.init(project=projectname, entity="byfrfy", name=wandbname, tags=tags)

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # Read data
    train_cot_path = "./RL_data/train_data.jsonl"
    val_cot_path = "./RL_data/val_data.jsonl"
    test_cot_path = "./RL_data/test_data.jsonl"

    train_data = read_cot_data(train_cot_path)
    validation_data = read_cot_data(val_cot_path)
    test = read_cot_data(test_cot_path)

    print(f"Data sizes -> train: {len(train_data)}, val: {len(validation_data)}, test: {len(test)}")

    train_data = get_molecule_questions(Dataset.from_list(build_QA_dict(train_data, zero_shot=zero_shot)))
    validation_data = get_molecule_questions(Dataset.from_list(build_QA_dict(validation_data, zero_shot=zero_shot)))

    # Model
    os.environ["TRANSFORMERS_CACHE"] = "./"
    tokenizer = AutoTokenizer.from_pretrained(finetune_name)
    model = AutoModelForCausalLM.from_pretrained(finetune_name, device_map="auto")

    # Training config
    training_args = GRPOConfig(
        learning_rate=lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_train_epochs=num_train_epochs,
        save_strategy="epoch",
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=output_dir,
        repetition_penalty=1.2,
        log_completions=True,
        temperature=temperature,
        beta=KL,
        epsilon=1.2,
        epsilon_high=2.8,
        loss_type="dr_grpo",
        mask_truncated_completions=True,
    )

    if conditional_reward_style:
        reward_funcs = [strict_format_reward_func, stack_reward, single_tag_occurrence_reward]
    else:
        reward_funcs = [
            correctness_reward_func,
            len_reward,
            strict_format_reward_func,
            soft_format_reward_func,
            is_SMILES_reward,
            single_tag_occurrence_reward,
        ]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
    )

    trainer.train()

    # --------------------
    # Evaluation on test_path (tab-separated: answer \t description per line)
    # --------------------
    raw_test_data, answers = [], []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                ans, des = parts[0], parts[1]
                raw_test_data.append({"des": des, "answer": ans})
                answers.append(ans)

    test_dataset = Dataset.from_list(build_test_messages(raw_test_data))
    messages = test_dataset["messages"]
    dataloader = DataLoader(messages, batch_size=BATCH_SIZE, collate_fn=list)

    results_dir = Path(output_dir) / "inference_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = (wandb.run.name or "run").replace("/", "_")
    run_id = wandb.run.id if wandb.run is not None else "noid"
    jsonl_path = results_dir / f"{run_name}_{run_id}.jsonl"
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    EM, similarity_score, validation_score = [], [], []

    is_qwen3 = ("Qwen3" in finetune_name) or ("qwen3" in finetune_name)

    for i, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        if is_qwen3:
            texts = [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True, enable_thinking=False) for item in batch]
        else:
            texts = [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for item in batch]

        model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for j, response in enumerate(responses):
            global_idx = i * BATCH_SIZE + j
            ref_answer = answers[global_idx]
            pred_answer, thinking = extract_last_tags(response)

            if pred_answer and is_SMILES(pred_answer):
                sim = tanimoto_similarity(is_SMILES(pred_answer), is_SMILES(ref_answer))
                similarity_score.append(sim)
                validation_score.append(1.0)
                EM.append(1.0 if sim == 1.0 else 0.0)
            else:
                similarity_score.append(0.0)
                validation_score.append(0.0)
                EM.append(0.0)

            # Write JSONL record
            des_text = raw_test_data[global_idx]["des"]
            cot_block = f"<Thinking>\n{thinking}\n</Thinking>\n<Answer>{pred_answer}</Answer>"
            record = {
                "index": global_idx,
                "smile": pred_answer,
                "explain": des_text,
                "cot": cot_block,
                "ref_smile": ref_answer,
                "em": EM[-1],
                "similarity": similarity_score[-1],
                "valid": validation_score[-1],
                "raw_response": response,
            }
            jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("========== Final Evaluation ==========")
    print("Mean EM:", float(np.mean(EM) if EM else 0.0))
    print("Mean Similarity:", float(np.mean(similarity_score) if similarity_score else 0.0))
    print("Mean Validation Rate:", float(np.mean(validation_score) if validation_score else 0.0))

    wandb.log({
        "mean EM": float(np.mean(EM) if EM else 0.0),
        "mean similarity": float(np.mean(similarity_score) if similarity_score else 0.0),
        "mean validation": float(np.mean(validation_score) if validation_score else 0.0),
    })

    jsonl_f.close()
    print(f"[Saved] Per-run JSONL results to: {jsonl_path}")


# -------------------------------
# CLI
# -------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Qwen_RL")
    parser.add_argument("--projectname", type=str, default="Qwen_base_full_RL")
    parser.add_argument("--wandbname", type=str, default="Qwen_RL_non_conditional_non_KL")
    parser.add_argument("--finetune_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output_dir", type=str, default="Qwen_base_full_RL_KL_softformat_conditional_similarity")
    parser.add_argument("--conditional_reward_style", type=str2bool, default=False)
    parser.add_argument("--KL", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--zero_shot", type=str2bool, default=True)
    args = parser.parse_args()

    Qwen_RL(
        projectname=args.projectname,
        wandbname=args.wandbname,
        finetune_name=args.finetune_name,
        output_dir=args.output_dir,
        conditional_reward_style=args.conditional_reward_style,
        KL=args.KL,
        temperature=args.temperature,
        lr=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        zero_shot=args.zero_shot,
    )


if __name__ == "__main__":
    main()
