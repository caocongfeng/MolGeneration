#!/usr/bin/env python3
"""
Fast evaluation of MolReasoner using vLLM for batched inference.

Usage:
    python evaluate_molreasoner_fast.py \
        --test_path Mix_SMolInstruct_test.jsonl \
        --model_name guojianz/MolReasoner \
        --output_path results_molreasoner.jsonl
"""

import argparse
import json
import os
import re
import time

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
import selfies as sf

RDLogger.DisableLog('rdApp.*')


# ==================== Chemistry helpers ====================
def is_SMILES(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None


def clean_selfies(selfies_str):
    """Fix common malformed SELFIES patterns from the model."""
    s = selfies_str
    s = re.sub(r'\[Ring\]\[(\d)\]', r'[Ring\1]', s)
    s = re.sub(r'\[Branch\]\[(\d)\]', r'[Branch\1]', s)
    s = re.sub(r'\[=Ring\]\[(\d)\]', r'[=Ring\1]', s)
    s = re.sub(r'\[=Branch\]\[(\d)\]', r'[=Branch\1]', s)
    s = re.sub(r'\[#Ring\]\[(\d)\]', r'[#Ring\1]', s)
    s = re.sub(r'\[#Branch\]\[(\d)\]', r'[#Branch\1]', s)
    s = s.replace('[C@@H]', '[C]')
    s = s.replace('[C@H]', '[C]')
    return s


def selfies_to_smiles(selfies_str):
    for s in [selfies_str, clean_selfies(selfies_str)]:
        try:
            smiles = sf.decoder(s)
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            pass
    return None


def try_parse_to_canonical(raw_str):
    if not raw_str:
        return None
    raw_str = raw_str.strip()
    canon = is_SMILES(raw_str)
    if canon:
        return canon
    canon = selfies_to_smiles(raw_str)
    if canon:
        return canon
    for sep in ['\n', ' ', ',']:
        parts = raw_str.split(sep)
        if len(parts) > 1:
            first = parts[0].strip()
            canon = is_SMILES(first)
            if canon:
                return canon
            canon = selfies_to_smiles(first)
            if canon:
                return canon
    return None


def exact_match(smiles_a, smiles_b):
    if not smiles_a or not smiles_b:
        return False
    return smiles_a == smiles_b


def are_smiles_similar(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ==================== Tag extraction ====================
def extract_last_tags(text):
    answer_pattern = re.compile(r"<[Aa]nswer>(.*?)</[Aa]nswer>", re.DOTALL)
    answers = list(re.finditer(answer_pattern, text))
    answer = answers[-1].group(1).strip() if answers else ""
    return answer


# ==================== Main ====================
def evaluate(args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"{'='*60}")
    print(f"MolReasoner Fast Evaluation (vLLM)")
    print(f"Model: {args.model_name}")
    print(f"Subfolder: {args.subfolder}")
    print(f"Test: {args.test_path}")
    print(f"GPU count: {args.tensor_parallel_size}")
    print(f"{'='*60}\n")

    # ----- Load test data -----
    test_data = []
    with open(args.test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            test_data.append({
                "answer": record.get("smile", ""),
                "des": record.get("explain", ""),
            })
    print(f"Loaded {len(test_data)} test samples\n")

    # ----- Build model path -----
    # For subfolder models, we need to construct the full path
    # vLLM doesn't support subfolder directly, so we use snapshot_download
    from huggingface_hub import snapshot_download
    print(f"Downloading/caching model from {args.model_name}/{args.subfolder}...")
    model_path = snapshot_download(
        args.model_name,
        allow_patterns=[f"{args.subfolder}/*"],
    )
    full_model_path = os.path.join(model_path, args.subfolder)
    print(f"Model path: {full_model_path}\n")

    # ----- Load tokenizer (for chat template) -----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(full_model_path, trust_remote_code=True)

    # ----- Build all prompts -----
    print("Building prompts...")
    all_prompts = []
    for item in test_data:
        messages = [
            {"role": "system",
             "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation, professional and rigorous."},
            {"role": "user",
             "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation. "
                        "Could you please provide the SMILES representation based on this molecular description: {} \n "
                        "1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. "
                        "2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description."
                        .format(item["des"])},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt)
    print(f"Built {len(all_prompts)} prompts\n")

    # ----- Initialize vLLM engine -----
    print("Initializing vLLM engine...")
    llm = LLM(
        model=full_model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
    )

    sampling_params = SamplingParams(
        temperature=0,  # greedy
        max_tokens=args.max_new_tokens,
        stop=["<|endoftext|>", "<|im_end|>"],
    )

    # ----- Run batch inference -----
    print(f"\nRunning batch inference on {len(all_prompts)} prompts...")
    start_time = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.1f}s ({inference_time/len(all_prompts):.2f}s/sample)\n")

    # ----- Evaluate -----
    print("Evaluating results...")
    sample_EM, sample_SIM, sample_VAL = [], [], []

    out_f = open(args.output_path, 'w', encoding='utf-8') if args.output_path else None

    for idx, (output, data) in enumerate(zip(outputs, test_data)):
        response = output.outputs[0].text
        ref_answer_raw = data["answer"]
        ref_canon = is_SMILES(ref_answer_raw)

        pred_answer_raw = extract_last_tags(response)
        pred_canon = try_parse_to_canonical(pred_answer_raw)
        is_valid = pred_canon is not None

        if is_valid and exact_match(pred_canon, ref_canon):
            em, sim, val = 1.0, 1.0, 1.0
        elif is_valid:
            sim = are_smiles_similar(pred_canon, ref_canon) if ref_canon else 0.0
            em, val = 0.0, 1.0
        else:
            em, sim, val = 0.0, 0.0, 0.0

        sample_EM.append(em)
        sample_SIM.append(sim)
        sample_VAL.append(val)

        record = {
            "idx": idx,
            "reference": ref_answer_raw,
            "reference_canon": ref_canon,
            "prediction": pred_answer_raw,
            "prediction_canon": pred_canon,
            "exact_match": em,
            "similarity": sim,
            "valid": val,
        }
        if out_f:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{len(test_data)}] "
                  f"EM={np.mean(sample_EM):.4f}  SIM={np.mean(sample_SIM):.4f}  VAL={np.mean(sample_VAL):.4f}")

    if out_f:
        out_f.close()

    total_time = time.time() - start_time
    mean_EM = float(np.mean(sample_EM))
    mean_SIM = float(np.mean(sample_SIM))
    mean_VAL = float(np.mean(sample_VAL))

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS ({len(test_data)} samples)")
    print(f"{'='*60}")
    print(f"  mean_EM          : {mean_EM:.4f}")
    print(f"  mean_similarity  : {mean_SIM:.4f}")
    print(f"  mean_validation  : {mean_VAL:.4f}")
    print(f"  Inference time   : {inference_time:.1f}s")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  Avg time/sample  : {inference_time/len(test_data):.3f}s")
    print(f"{'='*60}")

    if args.output_path:
        print(f"\nPer-sample predictions saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast MolReasoner evaluation with vLLM")
    parser.add_argument("--test_path", type=str, default="Mix_SMolInstruct_test.jsonl")
    parser.add_argument("--model_name", type=str, default="guojianz/MolReasoner")
    parser.add_argument("--subfolder", type=str,
                        default="grpo/grpo_text_based_de_novo_molecule_generation")
    parser.add_argument("--output_path", type=str, default="results_molreasoner.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Max total sequence length (prompt + output)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism")
    args = parser.parse_args()
    evaluate(args)
