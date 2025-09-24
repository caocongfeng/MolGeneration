import os
import json
import re
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from datasets import Dataset

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

# ================= Chemistry helpers =================

def is_SMILES(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def exact_match(smiles_a: str | None, smiles_b: str | None) -> bool:
    if not smiles_a or not smiles_b:
        return False
    return smiles_a == smiles_b


def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fp1, fp2 = gen.GetFingerprint(mol1), gen.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ================= IO helpers =================

def read_cot_data(cot_path: str) -> List[Dict]:
    data = []
    with open(cot_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            data.append({
                "answer": j.get("smile", ""),
                "des": j.get("explain", ""),
                "cot": j.get("cot", ""),
            })
    return data


def append_jsonl(record: Dict, path: str, overwrite_if_new: bool = False) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    if overwrite_if_new and (not hasattr(append_jsonl, "_touched") or path not in append_jsonl._touched):
        open(path, "w", encoding="utf-8").close()
        append_jsonl._touched = getattr(append_jsonl, "_touched", set())
        append_jsonl._touched.add(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_special_tokens(tokenizer, model) -> None:
    added = False
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        added = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        added = True
    if added:
        model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id


# ================= Builders =================

_ANS_RE = re.compile(r"<\s*(?:Answer|answer)\s*>(.*?)<\s*/\s*(?:Answer|answer)\s*>", re.DOTALL)


def split_cot_answer(cot_text: str, fallback_answer: str = "") -> Tuple[str, str]:
    if cot_text is None:
        cot_text = ""
    matches = list(_ANS_RE.finditer(cot_text))
    if matches:
        last = matches[-1]
        answer_text = last.group(1).strip()
        thinking_text = (cot_text[:last.start()] + cot_text[last.end():]).strip()
    else:
        answer_text = (fallback_answer or "").strip()
        thinking_text = cot_text.strip()
    thinking_text = _ANS_RE.sub("", thinking_text).strip()
    return thinking_text, answer_text


def build_messages_from_cot(cot_data: List[Dict], tokenizer) -> List[Dict]:
    eos = tokenizer.eos_token or "</s>"
    out = []
    for i in cot_data:
        des = i.get("des", "").strip()
        cot_raw = (i.get("cot") or "").replace("<answer>", "<Answer>").replace("</answer>", "</Answer>")
        thinking_text, answer_text = split_cot_answer(cot_raw, fallback_answer=i.get("answer", ""))
        if not thinking_text:
            thinking_text = cot_raw.strip()
        assistant_content = (
            f"<Thinking>\n{thinking_text}\n</Thinking>\n<Answer>\n{answer_text}\n</Answer>{eos}"
        )
        messages = [
            {"role": "system", "content": (
                "You are an expert in info-chemistry, especially in SMILES interpretation and translation, professional and rigorous."
            )},
            {"role": "user", "content": (
                "You are an expert in info-chemistry, especially in SMILES interpretation and translation. "
                f"Could you please provide the SMILES representation based on this molecular description: {des} \n "
                "1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. "
                "2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description."
            )},
            {"role": "assistant", "content": assistant_content},
        ]
        out.append({"messages": messages})
    return out


def build_test_messages(test_data: List[Dict]) -> List[Dict]:
    out = []
    for i in test_data:
        messages = [
            {"role": "system", "content": (
                "You are an expert in info-chemistry, especially in SMILES interpretation and translation, professional and rigorous."
            )},
            {"role": "user", "content": (
                "You are an expert in info-chemistry, especially in SMILES interpretation and translation. "
                f"Could you please provide the SMILES representation based on this molecular description: {i['des']} \n "
                "1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. "
                "2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description."
            )},
        ]
        out.append({"messages": messages})
    return out


# ================= Tag extraction (eval) =================

def extract_last_tags(text: str) -> Tuple[str, str]:
    ans_re = re.compile(r"<[Aa]nswer>(.*?)</[Aa]nswer>", re.DOTALL)
    thk_re = re.compile(r"<[Tt]hinking>(.*?)</[Tt]hinking>", re.DOTALL)
    answers = list(ans_re.finditer(text))
    thinkings = list(thk_re.finditer(text))
    ans = answers[-1].group(1).strip() if answers else ""
    thk = thinkings[-1].group(1).strip() if thinkings else ""
    return ans, thk


# ================= Best-of-N testing =================

def run_test_and_log(
    model,
    tokenizer,
    test_path: str,
    BATCH_SIZE: int = 4,
    model_type: str | None = None,
    save_jsonl_path: str | None = None,
    include_curation_fields: bool = True,
    default_curation_score: int = 3,
    lower_answer_tag: bool = True,
    best_of_n: int = 16,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    log_to_wandb: bool = True,
) -> Dict[int, Dict[str, float]]:
    # Build schedule: N, N/2, ..., 1
    n_values: List[int] = []
    n = best_of_n
    while n >= 1:
        n_values.append(n)
        n //= 2

    all_metrics: Dict[int, Dict[str, float]] = {}

    for cur_n in n_values:
        print(f"\n{'='*15} Running Test with Best-of-{cur_n} {'='*15}\n")
        if save_jsonl_path:
            append_jsonl({"_meta": f"test_predictions_begin_bestof_{cur_n}"}, save_jsonl_path, overwrite_if_new=(cur_n == best_of_n))

        raw_test_data, answers = [], []
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ans, des = line.strip().split("\t")
                    raw_test_data.append({"des": des, "answer": ans})
                    answers.append(ans)
                except ValueError:
                    continue

        dataset = Dataset.from_list(build_test_messages(raw_test_data))
        messages = dataset["messages"]
        dataloader = DataLoader(messages, batch_size=BATCH_SIZE, collate_fn=list)

        sample_EM: List[float], sample_SIM: List[float], sample_VAL: List[float] = [], [], []
        gid_base = 0

        for batch in tqdm(dataloader, desc=f"Running inference (test, best-of-{cur_n})"):
            if model_type == "Qwen3":
                batch_texts = [
                    tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    for item in batch
                ]
            else:
                batch_texts = [
                    tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
                    for item in batch
                ]

            for j, prompt_text in enumerate(batch_texts):
                gid = gid_base + j
                if gid >= len(answers):
                    continue
                ref_canon = is_SMILES(answers[gid])

                best_sim, any_valid, any_em = 0.0, False, False
                best_pred, best_think = "", ""

                model_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
                do_sample = cur_n > 1

                for _ in range(cur_n):
                    with torch.no_grad():
                        gen_ids = model.generate(
                            **model_inputs,
                            max_new_tokens=2048,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            do_sample=do_sample,
                            temperature=temperature if do_sample else 1.0,
                            top_p=top_p if do_sample else 1.0,
                            top_k=top_k if do_sample else 0,
                            use_cache=True,
                        )
                    resp = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    pred_raw, thinking = extract_last_tags(resp)

                    if pred_raw:
                        cand_canon = is_SMILES(pred_raw)
                        if cand_canon is not None:
                            any_valid = True
                            if exact_match(cand_canon, ref_canon):
                                any_em = True
                                best_pred, best_think = cand_canon, thinking
                            sim = tanimoto_similarity(cand_canon, ref_canon) if ref_canon else 0.0
                            if sim > best_sim:
                                best_sim, best_pred, best_think = sim, cand_canon, thinking

                if any_em:
                    em, sim, val = 1.0, 1.0, 1.0
                else:
                    em, sim, val = 0.0, best_sim, 1.0 if any_valid else 0.0

                sample_EM.append(em)
                sample_SIM.append(sim)
                sample_VAL.append(val)

            gid_base += len(batch_texts)

        metrics = {
            "bestofN": float(cur_n),
            "mean_EM": float(np.mean(sample_EM)) if sample_EM else 0.0,
            "mean_similarity": float(np.mean(sample_SIM)) if sample_SIM else 0.0,
            "mean_validation": float(np.mean(sample_VAL)) if sample_VAL else 0.0,
        }
        all_metrics[cur_n] = metrics

        print(f"\n{'='*10} Final Test Evaluation (best-of-{cur_n}) {'='*10}")
        print("Mean EM:", metrics["mean_EM"])
        print("Mean Similarity:", metrics["mean_similarity"])
        print("Mean Validation Rate:", metrics["mean_validation"])\

        if log_to_wandb and wandb.run is not None:
            wandb.log({
                f"test/bestofN_{cur_n}/mean_EM": metrics["mean_EM"],
                f"test/bestofN_{cur_n}/mean_similarity": metrics["mean_similarity"],
                f"test/bestofN_{cur_n}/mean_validation": metrics["mean_validation"],
            })

    return all_metrics


# ================= Per-epoch test callback =================

class TestPerEpochCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        test_path: str,
        batch_size: int,
        model_type: str | None,
        pred_save_dir: str | None = None,
        best_of_n: int = 16,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> None:
        self.tokenizer = tokenizer
        self.test_path = test_path
        self.batch_size = batch_size
        self.model_type = model_type
        self.pred_save_dir = pred_save_dir
        self.best_of_n = best_of_n
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        epoch_int = int(state.epoch) if state.epoch is not None else 0

        # Only run at final epoch end (example logic); adjust if needed
        if epoch_int != 3:
            return control

        save_path = None
        if self.pred_save_dir:
            os.makedirs(self.pred_save_dir, exist_ok=True)
            save_path = os.path.join(self.pred_save_dir, f"epoch_{epoch_int}.jsonl")

        print(f"\n===== Running TEST (best-of-{self.best_of_n}) at end of epoch {state.epoch} =====")
        metrics = run_test_and_log(
            model,
            self.tokenizer,
            self.test_path,
            BATCH_SIZE=self.batch_size,
            model_type=self.model_type,
            save_jsonl_path=save_path,
            include_curation_fields=True,
            default_curation_score=3,
            lower_answer_tag=True,
            best_of_n=self.best_of_n,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        if wandb.run is not None:
            payload = {"epoch": float(state.epoch) if state.epoch is not None else -1}
            for n, m in metrics.items():
                payload.update({
                    f"test/bestofN_{n}/mean_EM": m["mean_EM"],
                    f"test/bestofN_{n}/mean_similarity": m["mean_similarity"],
                    f"test/bestofN_{n}/mean_validation": m["mean_validation"],
                })
            wandb.log(payload)
        return control


# ================= Training =================

def SFT(
    projectname: str,
    wandbname: str,
    finetune_name: str,
    output_dir: str,
    lr: float,
    Full_tuning: bool,
    lora_rank: int,
    lora_dropout: float,
    num_train_epochs: int,
    BATCH_SIZE: int = 4,
    test_path: str = "./RL_data/test_data.txt",
    best_of_n: int = 16,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> None:

    wandb.login()
    wandb.init(project=projectname, entity="byfrfy", name=wandbname)

    lower = finetune_name.lower()
    is_llama = "llama" in lower
    is_mistral = ("mistral" in lower) or ("mistralai" in lower)
    is_qwen = "qwen" in lower
    is_qwen3 = ("qwen3" in lower) or ("/qwen3-" in lower)

    if is_llama or is_mistral:
        tokenizer = AutoTokenizer.from_pretrained(finetune_name, use_fast=True, padding_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            finetune_name,
            attn_implementation="sdpa",
            device_map="balanced",
            torch_dtype=torch.bfloat16,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(finetune_name, trust_remote_code=True, padding_side="left")
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            finetune_name,
            attn_implementation="sdpa",
            device_map="balanced",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    ensure_special_tokens(tokenizer, model)

    peft_config = None
    if not Full_tuning:
        if is_llama or is_mistral:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ]
        else:
            target_modules = "all-linear"
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

    sft_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="adamw_torch_fused",
        learning_rate=lr,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        push_to_hub=False,
        report_to="wandb",
        max_length=2048,
    )

    train_data = Dataset.from_list(
        build_messages_from_cot(read_cot_data("./Mix_SMolInstruct/Mix_SMolInstruct_train.jsonl"), tokenizer)
    )
    val_data = Dataset.from_list(
        build_messages_from_cot(read_cot_data("./Mix_SMolInstruct/Mix_SMolInstruct_val.jsonl"), tokenizer)
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
    )

    pred_save_dir = os.path.join(output_dir, "test_preds")
    os.makedirs(pred_save_dir, exist_ok=True)

    if is_qwen3:
        model_type = "Qwen3"
    elif is_qwen:
        model_type = "Qwen2.5"
    elif is_llama:
        model_type = "LLaMA"
    elif is_mistral:
        model_type = "Mistral"
    else:
        model_type = None

    test_cb = TestPerEpochCallback(
        tokenizer=tokenizer,
        test_path=test_path,
        batch_size=BATCH_SIZE,
        model_type=model_type,
        pred_save_dir=pred_save_dir,
        best_of_n=best_of_n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    trainer.add_callback(test_cb)

    trainer.train()


# ================= CLI =================

def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "y", "1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projectname", type=str, default="Qwen3_SFT")
    parser.add_argument("--wandbname", type=str, default="Qwen3_8B_SFT_FULL")
    parser.add_argument("--finetune_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="./SFT_models/Qwen3_8B_Full_SFT")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--Full_tuning", type=str2bool, default=True)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.06)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--test_path", type=str, default="./RL_data/test_data.txt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--best_of_n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()

    SFT(
        projectname=args.projectname,
        wandbname=args.wandbname,
        finetune_name=args.finetune_name,
        output_dir=args.output_dir,
        lr=args.lr,
        Full_tuning=args.Full_tuning,
        lora_rank=args.lora_rank,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.num_train_epochs,
        BATCH_SIZE=args.batch_size,
        test_path=args.test_path,
        best_of_n=args.best_of_n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
