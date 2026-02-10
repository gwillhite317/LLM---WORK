from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PROMPT_TEMPLATE = (
    'Decide whether the statement is true or false.\n'
    'Answer with only "true" or "false".\n'
    'Statement: {statement}\n'
    'Answer:'  # no trailing space/newline; helps next-token scoring
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name or local path")
    p.add_argument("--prompt", type=str, default=None, help="(Optional) override prompt template")
    p.add_argument("--data", type=str, default="wiki_tf.jsonl", help="Path to JSONL dataset")
    p.add_argument("--max_new_tokens", type=int, default=128, help="Unused in scoring-mode; kept for spec-compat")
    p.add_argument("--temperature", type=float, default=0.7, help="Unused in scoring-mode; kept for spec-compat")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
            items.append(obj)

    if len(items) != 10:
        raise ValueError(f"{path} must contain exactly 10 entries; found {len(items)}")

    for obj in items:
        for k in ("id", "article", "statement", "label"):
            if k not in obj:
                raise ValueError(f"Missing required key '{k}' in entry: {obj}")
        if obj["label"] not in ("true", "false"):
            raise ValueError(f"label must be exactly 'true' or 'false' (lowercase). Bad: {obj['label']}")

    return items


def predict_tf_by_scoring(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    statement: str,
    prompt_override: str | None = None,
) -> str:
    """
    Deterministic TF prediction by scoring candidate completions:
      P(" true" | prompt) vs P(" false" | prompt)
    This avoids generation failures that lead to "invalid" outputs.
    """
    template = prompt_override if prompt_override is not None else PROMPT_TEMPLATE
    prompt = template.format(statement=statement)

    # For GPT-style tokenizers, a leading space often makes these single tokens.
    candidates = [("true", " true"), ("false", " false")]

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Compute logits for the next token after the prompt
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1, :]  # (vocab,)

    best_label = None
    best_score = -float("inf")

    for label, cand_text in candidates:
        cand_ids = tokenizer.encode(cand_text, add_special_tokens=False)

        # Score multi-token candidates by rolling forward token-by-token
        score = 0.0
        cur_input = input_ids
        cur_attn = attention_mask
        cur_logits = logits

        for tid in cand_ids:
            log_probs = torch.log_softmax(cur_logits, dim=-1)
            score += float(log_probs[tid].cpu())

            # append this token to context
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

    # best_label will always be set because we score two options
    return best_label  # "true" or "false"


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    data = load_jsonl(args.data)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    correct = 0
    incorrect = []

    for ex in data:
        gold = ex["label"]
        pred = predict_tf_by_scoring(
            model=model,
            tokenizer=tokenizer,
            device=device,
            statement=ex["statement"],
            prompt_override=args.prompt,
        )

        if pred == gold:
            correct += 1
        elif len(incorrect) < 5:
            incorrect.append(
                {
                    "id": ex["id"],
                    "statement": ex["statement"],
                    "expected": gold,
                    "model_output": pred,
                }
            )

    acc = correct / len(data)
    print(f"Accuracy: {acc:.3f} ({correct}/{len(data)})")

    if incorrect:
        print("\nIncorrect (up to 5):")
        for ex in incorrect:
            print("-" * 60)
            print(f"id: {ex['id']}")
            print(f"statement: {ex['statement']}")
            print(f"expected: {ex['expected']}")
            print(f"model_output: {ex['model_output']}")
    
    if correct: 
        print("\n Correct (up to 5):")
        print(f"id: {ex['id']}")
        print(f"statement: {ex['statement']}")
        print(f"expected: {ex['expected']}")
        print(f"model_output: {ex['model_output']}")


if __name__ == "__main__":
    main()