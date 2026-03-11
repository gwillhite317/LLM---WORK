# compare_base_chat.py

import argparse
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str):
    """
    Load a tokenizer and causal language model from Hugging Face.

    torch_dtype='auto' lets Transformers choose a sensible dtype.
    device_map='auto' places the model on CPU or GPU automatically.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # Some models do not define a pad token; reuse eos_token if needed.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_base_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float
) -> str:
    """
    Generate from the base model using the raw prompt exactly as given.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = temperature > 0

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Keep only newly generated tokens, not the original prompt tokens.
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_chat_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float
) -> str:
    """
    Generate from the chat/instruction model.

    The *user prompt content* is the same as for the base model, but here
    we wrap it in the tokenizer's chat template because instruction/chat
    models are trained to expect structured role-based conversation input.
    """
    messages = [
        {"role": "user", "content": prompt}
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    do_sample = temperature > 0

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Compare a base LLM and a chat/instruction-tuned LLM on the same prompt."
    )

    parser.add_argument("--base_model", type=str, required=True,
                        help="Hugging Face model ID for the base model")
    parser.add_argument("--chat_model", type=str, required=True,
                        help="Hugging Face model ID for the instruction-tuned/chat model")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to send to both models")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"Chat model: {args.chat_model}")
    print()

    base_tokenizer, base_model = load_model_and_tokenizer(args.base_model)
    chat_tokenizer, chat_model = load_model_and_tokenizer(args.chat_model)

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(args.prompt)
    print()

    print("=" * 80)
    print("DECODING SETTINGS")
    print("=" * 80)
    print(f"max_new_tokens = {args.max_new_tokens}")
    print(f"temperature    = {args.temperature}")
    print(f"seed           = {args.seed}")
    print()

    base_output = generate_base_response(
        base_model,
        base_tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature
    )

    # Reset seed again so both generations start from the same RNG state.
    # This helps keep the comparison as fair/reproducible as possible.
    set_seed(args.seed)

    chat_output = generate_chat_response(
        chat_model,
        chat_tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature
    )

    print("=" * 80)
    print("BASE MODEL OUTPUT")
    print("=" * 80)
    print(base_output)
    print()

    print("=" * 80)
    print("CHAT / INSTRUCTION-TUNED MODEL OUTPUT")
    print("=" * 80)
    print(chat_output)
    print()


if __name__ == "__main__":
    main()