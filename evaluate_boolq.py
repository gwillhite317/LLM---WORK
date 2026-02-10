
from __future__ import annotations

import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# You need: pip install datasets
from datasets import load_dataset


PROMPT_A = """Read the passage and answer the question.
Answer only with "yes" or "no".
Passage:
{passage}
Question:
{question}
Answer:"""

PROMPT_B = """Answer the question.
Answer only with "yes" or "no".
Question:
{question}
Answer:"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2", help="HF model name or local path")
    p.add_argument("--n", type=int, default=100, help="Subset size N (e.g., 50-200)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument("--split", type=str, default="validation", help="BoolQ split: train/validation")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_yes_no(text: str) -> str:
    """
    Spec normalization:
    - lowercase
    - extract first occurrence of 'yes' or 'no'
    - if neither, return 'invalid'
    """
    t = text.lower()
    i_yes = t.find("yes")
    i_no = t.find("no")
    if i_yes == -1 and i_no == -1:
        return "invalid"
    if i_yes == -1:
        return "no"
    if i_no == -1:
        return "yes"
    return "yes" if i_yes < i_no else "no"


def score_candidates_next_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt: str,
    candidates: List[Tuple[str, str]],
) -> str:
    """
    Deterministically choose among candidates by scoring log-prob under the model.

    candidates: list of (label, text_to_append)
      e.g. [("yes", " yes"), ("no", " no")]
    """
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1, :]  # next-token logits

    best_label = None
    best_score = -float("inf")

    for label, append_text in candidates:
        cand_ids = tokenizer.encode(append_text, add_special_tokens=False)

        score = 0.0
        cur_input = input_ids
        cur_attn = attention_mask
        cur_logits = logits

        for tid in cand_ids:
            log_probs = torch.log_softmax(cur_logits, dim=-1)
            score += float(log_probs[tid].cpu())

            # advance context
            tid_tensor = torch.tensor([[tid]], device=device)
            cur_input = torch.cat([cur_input, tid_tensor], dim=1)
            if cur_attn is not None:
                one = torch.ones((1, 1), device=device, dtype=cur_attn.dtype)
                cur_attn = torch.cat([cur_attn, one], dim=1)

            with torch.no_grad():
                out2 = model(input_ids=cur_input, attention_mask=cur_attn)
                cur_logits = out2.logits[0, -1, :]

        if score > best_score:
            best_score = score
            best_label = label

    return best_label  # "yes" or "no"


def predict_yes_no(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    passage: str,
    question: str,
    with_passage: bool,
) -> str:
    """
    Returns predicted label "yes" or "no" using scoring.
    """
    if with_passage:
        prompt = PROMPT_A.format(passage=passage, question=question)
    else:
        prompt = PROMPT_B.format(question=question)

    # Leading space helps for GPT tokenizers
    candidates = [("yes", " yes"), ("no", " no")]
    pred = score_candidates_next_token(model, tokenizer, device, prompt, candidates)

    # Also apply the official normalization rule to what we'd *expect* from free text.
    # Here pred is already yes/no, but keeping normalization aligns with spec intent.
    return normalize_yes_no(pred)


def bool_to_yesno(answer_bool: bool) -> str:
    return "yes" if answer_bool else "no"


def run_experiment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    examples: List[Dict],
    with_passage: bool,
) -> Tuple[float, List[Dict]]:
    correct = 0
    incorrect_examples: List[Dict] = []

    for ex in examples:
        gold = bool_to_yesno(bool(ex["answer"]))
        pred = predict_yes_no(
            model=model,
            tokenizer=tokenizer,
            device=device,
            passage=ex["passage"],
            question=ex["question"],
            with_passage=with_passage,
        )

        if pred == gold:
            correct += 1
        else:
            if len(incorrect_examples) < 5:
                incorrect_examples.append(
                    {
                        "question": ex["question"],
                        "passage": ex["passage"] if with_passage else None,
                        "expected": gold,
                        "model_output": pred,
                    }
                )

    acc = correct / len(examples) if examples else 0.0
    return acc, incorrect_examples


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    # Load BoolQ
    ds = load_dataset("boolq")
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(ds.keys())}")

    split = ds[args.split]
    n = min(args.n, len(split))

    # Reproducible subset selection
    idxs = list(range(len(split)))
    rng = random.Random(args.seed)
    rng.shuffle(idxs)
    idxs = idxs[:n]
    subset = [split[i] for i in idxs]

    print(f"BoolQ subset size N = {n} (split='{args.split}', seed={args.seed})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    # Experiment A (with passage)
    acc_a, wrong_a = run_experiment(model, tokenizer, device, subset, with_passage=True)
    print(f"\nExperiment A (with passage) accuracy: {acc_a:.3f}")

    if wrong_a:
        print("\nExperiment A incorrect (up to 5):")
        for ex in wrong_a:
            print("-" * 60)
            print(f"question: {ex['question']}")
            print(f"expected: {ex['expected']}")
            print(f"model_output: {ex['model_output']}")
            print("passage:", ex["passage"])

    # Experiment B (without passage)
    acc_b, wrong_b = run_experiment(model, tokenizer, device, subset, with_passage=False)
    print(f"\nExperiment B (without passage) accuracy: {acc_b:.3f}")

    if wrong_b:
        print("\nExperiment B incorrect (up to 5):")
        for ex in wrong_b:
            print("-" * 60)
            print(f"question: {ex['question']}")
            print(f"expected: {ex['expected']}")
            print(f"model_output: {ex['model_output']}")

    # Optional: quick summary line
    print(f"\nSummary: N={n} | Acc(A)={acc_a:.3f} | Acc(B)={acc_b:.3f}")


if __name__ == "__main__":
    main()