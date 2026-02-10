#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW1 Part 1: prompt an LLM and print its response.")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name or local path")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to send to the model")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Avoid padding warnings for models like GPT-2
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # If temperature is 0 (or negative), do greedy decoding for determinism.
    do_sample = args.temperature > 0.0

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # Print only the completion (remove the prompt prefix if present)
    if decoded.startswith(args.prompt):
        print(decoded[len(args.prompt):].lstrip())
    else:
        print(decoded)


if __name__ == "__main__":
    main()
