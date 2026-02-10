# LLM---WORK
- Note I used the gpt2 model with 540m+ parameters because my computer has limited processing capability

For the first section all you must do is save the prompt_llm.py into a directory with an IDE or python working directory,  then run a prompt such as this in the terminal:
python .\llm_prompt.py --model gpt2 --prompt "Write a poem about Greece" --max_new_tokens 64 --temperature 0.7 --seed 0

For the Second Section, save the wiki.jsonl file and evaluate_wiki.py file into the same directory, then this prompt in the termnial:
python .\evaluate_wiki.py --model gpt2 --data ".\wiki_tf.jsonl" --seed 0 --temperature 0
- The accuracy score for this model was .5, so basically the same as a baseline dummy guessing model

For the Third and final section regarding boolq.  Make sure you have the datasets package downloaded so you can access bool q, save the evaluate_boolq.py file into a directory with an IDE, then run this command
python .\evaluate_boolq.py --model gpt2 --n 100 --seed 0

The accuracy for the boolq with passages was .550, and without passages the accuracy was actually higher at .580, with n = 100.  I imagine that this is because gpt2 often ignores longer passages and leans heavily on parametric information rather that the given information in the passage.  It would be interesing for the wiki questions if there was a way to format the statements simpler such as "Chicago Illinois", perhaps because this is so short it becomes kind f a word match
situation where it would see chicago and illinois together, and as they are so similar it would be more accurate.
