import os
import torch
import json
import re
import numpy as np
from datasets import Dataset

import wandb

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from torch.utils.data import DataLoader
from tqdm import tqdm

RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')



# ============= Chemistry helpers =============
def is_SMILES(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def exact_match(smiles_a, smiles_b):
    # 与当前调用保持一致：传入的 cand_canon / ref_canon 可能是规范化后的字符串或 None
    if not smiles_a or not smiles_b:
        return False
    return smiles_a == smiles_b




def are_smiles_similar(smiles1, smiles2, threshold=1.0):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ============= Data IO =============
def read_cot_data(cot_path):
    cot_data = []
    with open(cot_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            cot_data.append({
                "answer": data.get("smile", ""),
                "des": data.get("explain", ""),
                "cot": data.get("cot", "")
            })
    return cot_data


# ============= JSONL writer =============
def append_jsonl(record, path: str, overwrite_if_new: bool = False):
    """Append one JSON object per line to a .jsonl file."""
    dirn = os.path.dirname(path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    if overwrite_if_new and (not hasattr(append_jsonl, "_touched") or path not in append_jsonl._touched):
        with open(path, "w", encoding="utf-8") as _:
            pass
        append_jsonl._touched = getattr(append_jsonl, "_touched", set())
        append_jsonl._touched.add(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============= EOS / special tokens =============
def ensure_special_tokens(tokenizer, model):
    """
    Ensure eos/pad exist and align config; resize embeddings if we add tokens.
    """
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


# ============= Builders (training / testing) =============
_ANS_RE = re.compile(r"<\s*(?:Answer|answer)\s*>(.*?)<\s*/\s*(?:Answer|answer)\s*>", re.DOTALL)

def split_cot_answer(cot_text: str, fallback_answer: str = ""):
    """
    From a COT block that may contain <answer>...</answer>, extract the last answer
    and return (thinking_without_answer, answer_text).
    If no <answer> is found, use fallback_answer.
    """
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


def build_messages_from_cot(cot_data, tokenizer):
    """
    Convert records with keys {des, cot, answer} into chat training examples:
      system, user(prompt), assistant: <Thinking>...</Thinking><Answer>...</Answer><eos>
    """
    eos = tokenizer.eos_token or "</s>"
    messages_list = []

    for i in cot_data:
        des = i.get("des", "").strip()
        cot_raw = (i.get("cot") or "").replace("<answer>", "<Answer>").replace("</answer>", "</Answer>")
        thinking_text, answer_text = split_cot_answer(cot_raw, fallback_answer=i.get("answer", ""))

        if not thinking_text:
            thinking_text = cot_raw.strip()

        assistant_content = (
            f"<Thinking>\n{thinking_text}\n</Thinking>\n"
            f"<Answer>\n{answer_text}\n</Answer>{eos}"
        )

        messages = [
            {"role": "system",
             "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation, professional and rigorous."},
            {"role": "user",
             "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation. "
                        f"Could you please provide the SMILES representation based on this molecular description: {des} \n "
                        "1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. "
                        "2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description."},
            {"role": "assistant", "content": assistant_content},
        ]
        messages_list.append({"messages": messages})

    return messages_list


def build_test_messages(test_data):
    messages_list = []
    for i in test_data:
        messages = [
            {"role": "system",
             "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation, professional and rigorous."},
            {"role": "user",
             "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation. "
                        "Could you please provide the SMILES representation based on this molecular description: {} \n "
                        "1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. "
                        "2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description."
                        .format(i["des"])},
        ]
        messages_list.append({"messages": messages})
    return messages_list


# ============= Tag extraction for evaluation =============
def extract_last_tags(text):
    answer_pattern = re.compile(r"<[Aa]nswer>(.*?)</[Aa]nswer>", re.DOTALL)
    thinking_pattern = re.compile(r"<[Tt]hinking>(.*?)</[Tt]hinking>", re.DOTALL)
    answers = list(re.finditer(answer_pattern, text))
    thinkings = list(re.finditer(thinking_pattern, text))
    answer = answers[-1].group(1).strip() if answers else ""
    thinking = thinkings[-1].group(1).strip() if thinkings else ""
    return answer, thinking


# # ============= Testing (Best-of-N + JSONL saving) =============
# def run_test_and_log(model, tokenizer, test_path, BATCH_SIZE=4, model_type=None,
#                      save_jsonl_path: str = None,
#                      include_curation_fields: bool = True,
#                      default_curation_score: int = 3,
#                      lower_answer_tag: bool = True,
#                      best_of_n: int = 16,
#                      temperature: float = 1.0,
#                      top_p: float = 0.9,
#                      top_k: int = 50):
#     """
#     Best-of-N 推理评估：
#       - 每条测试样本生成 N 次
#       - 若任一次 EM==1 -> 该样本(EM, Similarity, Validation) = (1,1,1)
#       - 否则：Similarity = N次中的最大值；Validation = 任一次有效即1，否则0；EM=0
#       - 最后对样本级指标取均值
#     """
#     if save_jsonl_path:
#         append_jsonl({"_meta": f"test_predictions_begin_bestof_{best_of_n}"}, save_jsonl_path, overwrite_if_new=True)

#     # 读取测试集（answer \t des）
#     raw_test_data, answers = [], []
#     with open(test_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 answer, des = line.strip().split('\t')
#                 raw_test_data.append({"des": des, "answer": answer})
#                 answers.append(answer)
#             except ValueError:
#                 continue

#     # 构造对话模板后分 batch
#     test_dataset = build_test_messages(raw_test_data)
#     test_dataset = Dataset.from_list(test_dataset)
#     messages = test_dataset["messages"]
#     dataloader = DataLoader(messages, batch_size=BATCH_SIZE, collate_fn=list)

#     # 汇总的样本级指标
#     sample_level_EM = []
#     sample_level_SIM = []
#     sample_level_VAL = []

#     global_idx_base = 0
#     for batch in tqdm(dataloader, desc="Running inference (test, best-of-N)"):
#         # 按模型族应用模板
#         if model_type == "Qwen3":
#             batch_texts = [
#                 tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True, enable_thinking=False)
#                 for item in batch
#             ]
#         else:
#             batch_texts = [
#                 tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
#                 for item in batch
#             ]

#         for local_j, prompt_text in enumerate(batch_texts):
#             global_idx = global_idx_base + local_j
#             if global_idx >= len(answers):
#                 continue
#             ref_answer = answers[global_idx]

#             # 对该样本进行 N 次生成
#             best_sim = 0.0
#             any_valid = False
#             any_em = False
#             best_pred = ""
#             best_thinking = ""

#             # 预先编码（单样本），避免重复 tokenization 带来的细微差异
#             model_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)

#             for n_try in range(max(1, best_of_n)):
#                 with torch.no_grad():
#                     generated_ids = model.generate(
#                         **model_inputs,
#                         max_new_tokens=2048,
#                         pad_token_id=tokenizer.pad_token_id,
#                         eos_token_id=tokenizer.eos_token_id,
#                         do_sample=True if best_of_n > 1 else False,
#                         temperature=temperature if best_of_n > 1 else 1.0,
#                         top_p=top_p if best_of_n > 1 else 1.0,
#                         top_k=top_k if best_of_n > 1 else 0,
#                     )
#                 response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#                 pred_answer, thinking = extract_last_tags(response)

#                 # 评估本次
#                 if pred_answer:
#                     valid = is_SMILES(pred_answer)
#                     if valid:
#                         any_valid = True
#                         sim_score = are_smiles_similar(valid, is_SMILES(ref_answer))
#                         if sim_score == 1.0:
#                             any_em = True
#                         if sim_score > best_sim:
#                             best_sim = sim_score
#                             best_pred = pred_answer
#                             best_thinking = thinking

#             # 汇总该样本的最终结果（best-of-N 规则）
#             if any_em:
#                 em = 1.0
#                 sim = 1.0
#                 val = 1.0
#                 if not best_pred:
#                     best_pred = ""
#                     best_thinking = ""
#             else:
#                 em = 0.0
#                 sim = best_sim
#                 val = 1.0 if any_valid else 0.0

#             # 打印
#             print("-"*10 + f"Output Example {global_idx} (best-of-{best_of_n})" + "-"*10)
#             print("Reference Answer: ", ref_answer)
#             print("Predicted Answer (best): ", best_pred)
#             print("Best Similarity: ", sim)
#             print("Any Valid: ", any_valid, " | Any EM: ", any_em)
#             print("EM/Sim/Val (sample-level): ", em, sim, val)
#             print("-"*10 + "End Example" + "-"*10)

#             # 保存 JSONL：保存该样本的“最佳”一次（EM=1 则为那次，否者为相似度最大一次）
#             if save_jsonl_path:
#                 ans_tag_open = "<answer>" if lower_answer_tag else "<Answer>"
#                 ans_tag_close = "</answer>" if lower_answer_tag else "</Answer>"
#                 cot_block = f"<Thinking>\n{best_thinking}\n</Thinking>\n{ans_tag_open}{best_pred}{ans_tag_close}"
#                 record = {
#                     "smile": best_pred,
#                     "explain": raw_test_data[global_idx]["des"],
#                     "cot": cot_block
#                 }
#                 if include_curation_fields:
#                     record["curation_score"] = default_curation_score
#                     record["reason"] = ""
#                 append_jsonl(record, save_jsonl_path)

#             sample_level_EM.append(em)
#             sample_level_SIM.append(sim)
#             sample_level_VAL.append(val)

#         global_idx_base += len(batch_texts)

#     metrics = {
#         "bestofN": int(best_of_n),
#         "mean_EM": float(np.mean(sample_level_EM)) if sample_level_EM else 0.0,
#         "mean_similarity": float(np.mean(sample_level_SIM)) if sample_level_SIM else 0.0,
#         "mean_validation": float(np.mean(sample_level_VAL)) if sample_level_VAL else 0.0,
#     }
#     print("="*10 + f"Final Test Evaluation (best-of-{best_of_n})" + "="*10)
#     print("Mean EM: ", metrics["mean_EM"])
#     print("Mean Similarity: ", metrics["mean_similarity"])
#     print("Mean Validation Rate: ", metrics["mean_validation"])
#     print("_"*40)
#     if save_jsonl_path:
#         print(f"[Saved predictions to] {save_jsonl_path}")
#     return metrics

def run_test_and_log(
    model, tokenizer, test_path, BATCH_SIZE=4, model_type=None,
    save_jsonl_path: str = None,
    include_curation_fields: bool = True,
    default_curation_score: int = 3,
    lower_answer_tag: bool = True,
    best_of_n: int = 16,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    log_to_wandb: bool = True,
):
    """
    Best-of-N 推理评估 + 下采样评估：
      - 从 best_of_n 开始，依次 /2 直到 1
        例如 best_of_n=16，则依次测试 16, 8, 4, 2, 1
        for cur_n in [16, 8, 4, 2, 1]:
        - 每次 best-of-cur_n 评估逻辑同之前
        - 每次运行完整测试，计算 EM / SIM / VAL
      - 打印并记录到 wandb（若启用）
    """

    # 构造采样列表 [best_of_n, best_of_n/2, ..., 1]
    n_values = []
    n = best_of_n
    while n >= 1:
        n_values.append(n)
        n //= 2

    all_metrics = {}

    for cur_n in n_values:
        print(f"\n{'='*15} Running Test with Best-of-{cur_n} {'='*15}\n")

        # =========== 保持你原有的测试逻辑，只是用 cur_n 替代 best_of_n ===========
        if save_jsonl_path:
            append_jsonl({"_meta": f"test_predictions_begin_bestof_{cur_n}"}, save_jsonl_path, overwrite_if_new=(cur_n==best_of_n))

        raw_test_data, answers = [], []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    answer, des = line.strip().split('\t')
                    raw_test_data.append({"des": des, "answer": answer})
                    answers.append(answer)
                except ValueError:
                    continue

        test_dataset = build_test_messages(raw_test_data)
        test_dataset = Dataset.from_list(test_dataset)
        messages = test_dataset["messages"]
        dataloader = DataLoader(messages, batch_size=BATCH_SIZE, collate_fn=list)

        sample_level_EM, sample_level_SIM, sample_level_VAL = [], [], []
        global_idx_base = 0

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

            for local_j, prompt_text in enumerate(batch_texts):
                global_idx = global_idx_base + local_j
                if global_idx >= len(answers):
                    continue

                ref_answer_raw = answers[global_idx]
                ref_canon = is_SMILES(ref_answer_raw)

                best_sim, any_valid, any_em = 0.0, False, False
                best_pred, best_thinking = "", ""

                model_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
                do_sample = cur_n > 1

                for _ in range(cur_n):
                    with torch.no_grad():
                        generated_ids = model.generate(
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
                    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    pred_answer_raw, thinking = extract_last_tags(response)

                    if pred_answer_raw:
                        cand_canon = is_SMILES(pred_answer_raw)
                        if cand_canon is not None:
                            any_valid = True
                            if exact_match(cand_canon, ref_canon):
                                any_em = True
                                best_pred, best_thinking = cand_canon, thinking
                            sim_score = are_smiles_similar(cand_canon, ref_canon) if ref_canon else 0.0
                            if sim_score > best_sim:
                                best_sim, best_pred, best_thinking = sim_score, cand_canon, thinking

                if any_em:
                    em, sim, val = 1.0, 1.0, 1.0
                else:
                    em, sim, val = 0.0, best_sim, 1.0 if any_valid else 0.0

                sample_level_EM.append(em)
                sample_level_SIM.append(sim)
                sample_level_VAL.append(val)

            global_idx_base += len(batch_texts)

        metrics = {
            "bestofN": int(cur_n),
            "mean_EM": float(np.mean(sample_level_EM)) if sample_level_EM else 0.0,
            "mean_similarity": float(np.mean(sample_level_SIM)) if sample_level_SIM else 0.0,
            "mean_validation": float(np.mean(sample_level_VAL)) if sample_level_VAL else 0.0,
        }
        all_metrics[cur_n] = metrics

        print(f"\n{'='*10} Final Test Evaluation (best-of-{cur_n}) {'='*10}")
        print("Mean EM: ", metrics["mean_EM"])
        print("Mean Similarity: ", metrics["mean_similarity"])
        print("Mean Validation Rate: ", metrics["mean_validation"])
        print("_"*40)

        # wandb logging
        if log_to_wandb and wandb.run is not None:
            wandb.log({
                f"test/bestofN_{cur_n}/mean_EM": metrics["mean_EM"],
                f"test/bestofN_{cur_n}/mean_similarity": metrics["mean_similarity"],
                f"test/bestofN_{cur_n}/mean_validation": metrics["mean_validation"],
            })

    return all_metrics

# ============= Per-epoch test callback (save JSONL) =============
class TestPerEpochCallback(TrainerCallback):
    def __init__(self, tokenizer, test_path, batch_size, model_type,
                 pred_save_dir: str = None,
                 best_of_n: int = 16,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 top_k: int = 50):
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

        if epoch_int !=3:
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
            lower_answer_tag=True,  # 训练数据示例使用小写 <answer>
            best_of_n=self.best_of_n,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        # if wandb.run is not None:
        #     wandb.log({
        #         "epoch": float(state.epoch) if state.epoch is not None else -1,
        #         "test/bestofN": metrics["bestofN"],
        #         "test/mean EM": metrics["mean_EM"],
        #         "test/mean similarity": metrics["mean_similarity"],
        #         "test/mean validation": metrics["mean_validation"],
        #     })
        # return control
        if wandb.run is not None:
            # 修复：metrics 是 dict-of-dicts，这里改为遍历记录
            payload = {"epoch": float(state.epoch) if state.epoch is not None else -1}
            for n, m in metrics.items():
                payload.update({
                    f"test/bestofN_{n}/mean_EM": m["mean_EM"],
                    f"test/bestofN_{n}/mean_similarity": m["mean_similarity"],
                    f"test/bestofN_{n}/mean_validation": m["mean_validation"],
                })
            wandb.log(payload)
        return control

# ============= Training & wiring =============
def SFT(projectname, wandbname, finetune_name, output_dir, lr, Full_tuning,
        lora_rank, save_steps, zero_shot, lora_dropout, num_train_epochs,
        BATCH_SIZE=4, test_path="./RL_data/test_data.txt",
        best_of_n: int = 16, temperature: float = 1.0, top_p: float = 0.9, top_k: int = 50):

    wandb.login()
    wandb.init(
        project=projectname,
        entity="byfrfy",
        name=wandbname,
    )

    # ========= Tokenizer / Model（按模型族做差异化设置）=========
    finetune_lower = finetune_name.lower()
    is_llama    = "llama" in finetune_lower
    is_mistral  = ("mistral" in finetune_lower) or ("mistralai" in finetune_lower)
    is_qwen     = "qwen" in finetune_lower
    is_qwen3    = ("qwen3" in finetune_lower) or ("/qwen3-" in finetune_lower)

    # Mistral / LLaMA: 右侧 padding，PAD=EOS，无需 trust_remote_code
    # Qwen / Qwen3:    左侧 padding，trust_remote_code=True
    if is_llama or is_mistral:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=finetune_name,
            use_fast=True,
            padding_side='right',
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=finetune_name,
            attn_implementation="sdpa",
            device_map="balanced",
            torch_dtype=torch.bfloat16,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=finetune_name,
            trust_remote_code=True,
            padding_side='left',
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=finetune_name,
            attn_implementation="sdpa",
            device_map="balanced",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

    ensure_special_tokens(tokenizer, model)

    # ========= LoRA（仅当 Full_tuning=False 时启用）=========
    peft_config = None
    if not Full_tuning:
        if is_llama or is_mistral:
            target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        else:
            # Qwen / Qwen3：沿用 all-linear
            target_modules = "all-linear"

        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

    # ========= 训练配置 =========
    SFTargs = SFTConfig(
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

    # ========= 构建数据集（保持 COT → chat messages 逻辑，自动补 EOS）=========
    train_data = Dataset.from_list(
        build_messages_from_cot(read_cot_data("./Mix_SMolInstruct/Mix_SMolInstruct_train.jsonl"), tokenizer)
    )
    val_data = Dataset.from_list(
        build_messages_from_cot(read_cot_data("./Mix_SMolInstruct/Mix_SMolInstruct_val.jsonl"), tokenizer)
    )

    trainer = SFTTrainer(
        model=model,
        args=SFTargs,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
    )

    # ======= 为预测结果保存目录：<output_dir>/test_preds =======
    pred_save_dir = os.path.join(output_dir, "test_preds")
    os.makedirs(pred_save_dir, exist_ok=True)

    # ========= 每个 epoch 结束即测试 + 记录到 wandb + 保存 JSONL =========
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

    test_callback = TestPerEpochCallback(
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
    trainer.add_callback(test_callback)

    trainer.train()


# ============= CLI =============
if __name__ == '__main__':
    import argparse
    def str2bool(v):
        return v.lower() in ('yes', 'true', 't', 'y', '1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--projectname', type=str, default="Qwen3_SFT")
    parser.add_argument("--wandbname", type=str, default="Qwen3_8B_SFT_FULL")
    parser.add_argument("--finetune_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="./SFT_models/Qwen3_8B_Full_SFT")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--Full_tuning", type=str2bool, default=True)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--zero_shot", type=str2bool, default=True)
    parser.add_argument("--lora_dropout", type=float, default=0.06)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--test_path", type=str, default="./RL_data/test_data.txt")
    parser.add_argument("--batch_size", type=int, default=4)

    # Best-of-N 与采样超参
    parser.add_argument("--best_of_n", type=int, default=16, help="Best-of-N sampling per test example")
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
        save_steps=args.save_steps,
        zero_shot=args.zero_shot,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.num_train_epochs,
        BATCH_SIZE=args.batch_size,
        test_path=args.test_path,
        best_of_n=args.best_of_n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
