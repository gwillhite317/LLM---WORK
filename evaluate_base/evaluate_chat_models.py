import argparse
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def build_prompt(sentence1: str, sentence2: str) -> list[dict]:
    """
    Return a chat-formatted prompt for paraphrase detection.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a careful text classification assistant. "
                "Answer with only one word: yes or no."
            ),
        },
        {
            "role": "user",
            "content": (
                "Do the following two sentences have the same meaning?\n"
                'Answer with only "yes" or "no".\n\n'
                f"Sentence 1: {sentence1}\n"
                f"Sentence 2: {sentence2}"
            ),
        },
    ]


def normalize_prediction(text: str) -> str | None:
    """
    Convert model output into a normalized binary label.
    Returns:
        'yes', 'no', or None if parsing fails.
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Instruction-tuned/chat model name from Hugging Face",
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
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model: {args.model_name}")
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

    correct = 0
    total = 0
    unparsable = 0

    for i, example in enumerate(dataset):
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        gold = "yes" if example["label"] == 1 else "no"

        messages = build_prompt(sentence1, sentence2)

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        pred = normalize_prediction(response)

        if pred is None:
            unparsable += 1
        else:
            total += 1
            if pred == gold:
                correct += 1

        print(
            f"[{i+1}/{len(dataset)}] "
            f"gold={gold:>3} pred={str(pred):>4} raw={response!r}"
        )

    accuracy = correct / total if total > 0 else 0.0

    print("\n=== Results ===")
    print(f"Model: {args.model_name}")
    print("Dataset: GLUE MRPC validation")
    print(f"Subset size requested: {args.subset_size}")
    print(f"Examples scored: {total}")
    print(f"Unparsable outputs: {unparsable}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

