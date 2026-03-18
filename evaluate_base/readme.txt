Part one:
- used qwen 2.5-.5b base and instruct model for this 
- the base model performed how I would expect, prettymuch tried to continue the sentence.  The instruction tuned model actually gave a coherent answer
prompt:
python compare_base_chat.py --base_model "Qwen/Qwen2.5-0.5B" --chat_model "Qwen/Qwen2.5-0.5B-Instruct" --prompt "Explain the purpose of masking in training an LLM in 3 sentences." --max_new_tokens 64 --temperature 0.7 --seed 0



part two:
- dataset chosen - glue mprc validation, a paraphrase detection dataset, subset size 100
- used deterministic decoding with a temperature of 0
- the accuracy score was .63
- model accuracy was a bit above baseline, tried to use a larger model, but failed to load.  
prompt:
 python evaluate_chat_models.py --model_name Qwen/Qwen2.5-0.5B-Instruct --subset_size 100                                                             


part three:
- few shot examples were basic questions like "the sky is blue", "the man is riding a bike" - "is a man riding a bike"
- the accuracy actually decreased from .63-.60 with few shot being implemented
- instruction tuned models typically perform better in the zero shot setting, this is because it was specifically trained to follow NL instructions.  The base model is trained for next token prediction so not very helpful.  Adding fewshot typically improves the base model because it shows prompt structure and lables for the model to follow
prompt:
python evaluate_base_fewshot.py --model_name Qwen/Qwen2.5-0.5B --subset_size 50 --mode both 

