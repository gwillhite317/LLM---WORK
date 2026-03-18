import argparse
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def build_zero_shot_prompt(sentence1: str, sentence2: str) -> str:
    return (
        "Do the following two sentences have the same meaning?\n"
        'Answer with only "yes" or "no".\n\n'
        f"Sentence 1: {sentence1}\n"
        f"Sentence 2: {sentence2}\n"
        "Answer:"
    )


def build_few_shot_prompt(sentence1: str, sentence2: str) -> str:
    return (
        "Do the following two sentences have the same meaning?\n"
        'Answer with only "yes" or "no".\n\n'

        "Sentence 1: The cat sat on the mat.\n"
        "Sentence 2: A cat was sitting on a mat.\n"
        "Answer: yes\n\n"

        "Sentence 1: The sky is blue.\n"
        "Sentence 2: Grass is green.\n"
        "Answer: no\n\n"

        "Sentence 1: A man is riding a bicycle.\n"
        "Sentence 2: A person is biking.\n"
        "Answer: yes\n\n"

        "Sentence 1: She is cooking dinner.\n"
        "Sentence 2: She is driving to work.\n"
        "Answer: no\n\n"

        f"Sentence 1: {sentence1}\n"
        f"Sentence 2: {sentence2}\n"
        "Answer:"
    )


def normalize_prediction(text: str) -> str | None:
    text = text.strip().lower()

    yes_match = re.search(r"\byes\b", text)
    no_match = re.search(r"\bno\b", text)

    if yes_match and not no_match:
        return "yes"
    if no_match and not yes_match:
        return "no"

    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"

    return None


def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def evaluate(model, tokenizer, dataset, prompt_mode: str, device: str, max_new_tokens: int):
    correct = 0
    total = 0
    unparsable = 0

    for i, example in enumerate(dataset):
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        gold = "yes" if example["label"] == 1 else "no"

        if prompt_mode == "zero-shot":
            prompt = build_zero_shot_prompt(sentence1, sentence2)
        else:
            prompt = build_few_shot_prompt(sentence1, sentence2)

        response = generate_response(model, tokenizer, prompt, device, max_new_tokens)
        pred = normalize_prediction(response)

        if pred is None:
            unparsable += 1
        else:
            total += 1
            if pred == gold:
                correct += 1

        print(
            f"[{i+1}/{len(dataset)}] mode={prompt_mode:>9} "
            f"gold={gold:>3} pred={str(pred):>4} raw={response!r}"
        )

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total, unparsable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model name from Hugging Face",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=100,
        help="Number of validation examples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=5,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="few-shot",
        choices=["zero-shot", "few-shot", "both"],
        help="Whether to run zero-shot, few-shot, or both",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        model = model.to(device)

    dataset = load_dataset("nyu-mll/glue", "mrpc", split="validation")
    dataset = dataset.select(range(min(args.subset_size, len(dataset))))

    if args.mode in ["zero-shot", "both"]:
        print("\n=== Evaluating base model: zero-shot ===")
        zs_acc, zs_total, zs_unparsable = evaluate(
            model, tokenizer, dataset, "zero-shot", device, args.max_new_tokens
        )
        print("\n=== Zero-shot Results ===")
        print(f"Examples scored: {zs_total}")
        print(f"Unparsable outputs: {zs_unparsable}")
        print(f"Accuracy: {zs_acc:.4f}")

    if args.mode in ["few-shot", "both"]:
        print("\n=== Evaluating base model: few-shot ===")
        fs_acc, fs_total, fs_unparsable = evaluate(
            model, tokenizer, dataset, "few-shot", device, args.max_new_tokens
        )
        print("\n=== Few-shot Results ===")
        print(f"Examples scored: {fs_total}")
        print(f"Unparsable outputs: {fs_unparsable}")
        print(f"Accuracy: {fs_acc:.4f}")


if __name__ == "__main__":
    main()