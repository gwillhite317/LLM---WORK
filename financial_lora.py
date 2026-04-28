import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report

model_name = "distilgpt2"
lora_rank = 16
lora_alpha = 32
epochs = 5
LR = 2e-4
output_dir = "lora_output"
labels = ["positive","negative","neutral"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fn_data = pd.read_csv(
    "C:/Users/mowma/Downloads/all-data.csv",
    encoding="latin-1",
    header=None,
    names=["sentiment", "headline"]
)
print(fn_data.info())

df = fn_data.sample(frac=1, random_state=42).reset_index(drop=True)
n  = len(df)
train_ds = Dataset.from_pandas(df.iloc[:int(n * 0.8)])
test_ds  = Dataset.from_pandas(df.iloc[int(n * 0.8):])
 
def prepare(example):
    example["label_str"] = example["sentiment"]
    example["text"] = (
        f"### Financial News:\n{example['headline']}\n\n"
        f"### Sentiment:\n{example['sentiment']}"
    )
    return example
 
train_ds = train_ds.map(prepare)
test_ds  = test_ds.map(prepare)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
 
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.pad_token_id
 
lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = lora_rank,
    lora_alpha     = lora_alpha,
    lora_dropout   = 0.05,
    target_modules = ["c_attn"],   # GPT-2; use ["q_proj","v_proj"] for LLaMA
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
 
# ── Training ──────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model        = model,
    args         = SFTConfig(
        output_dir              = output_dir,
        num_train_epochs        = epochs,
        per_device_train_batch_size = 8,
        learning_rate           = LR,
        max_length          = 128,
        logging_steps           = 20,
        fp16                    = torch.cuda.is_available(),
        report_to               = "none",
        dataset_text_field      = "text",
    ),
    train_dataset = train_ds,
    tokenizer     = tokenizer,
)
trainer.train()
model.save_pretrained(f"{OUTPUT_DIR}/adapter")
 
# ── Evaluation ────────────────────────────────────────────────────────────────
def predict(mdl, sentences):
    mdl.eval()
    preds = []
    with torch.inference_mode():
        for sentence in sentences:
            prompt = (
                f"### Financial News:\n{sentence}\n\n"
                f"### Sentiment:\n"
            )
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=128).to(device)
            out    = mdl.generate(**inputs, max_new_tokens=5, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.decode(out[0], skip_special_tokens=True).lower()
            pred = next((l for l in LABELS if l in decoded.split("### sentiment:\n")[-1]), "neutral")
            preds.append(pred)
    return preds
 
sentences = test_ds["headline"]
golds     = test_ds["sentiment"]
 
# Base model (zero-shot)
base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
base_preds = predict(base_model, sentences)
 
# Fine-tuned model
ft_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("gpt2"), f"{OUTPUT_DIR}/adapter"
).to(device)
ft_preds = predict(ft_model, sentences)
 
# ── Results ───────────────────────────────────────────────────────────────────
for name, preds in [("Base (zero-shot)", base_preds), ("LoRA fine-tuned", ft_preds)]:
    print(f"\n── {name} ──")
    print(f"Accuracy: {accuracy_score(golds, preds):.3f}")
    print(f"Macro-F1: {f1_score(golds, preds, average='macro', labels=LABELS, zero_division=0):.3f}")
    print(classification_report(golds, preds, labels=LABELS, zero_division=0))
