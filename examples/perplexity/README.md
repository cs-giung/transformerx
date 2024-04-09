# transformerx/examples/perplexity

The example demonstrates computing perplexity.

```bash
python examples/perplexity/perplexity.py
    --model_name meta-llama/Llama-2-7b-hf
    --data_name wikitext2
```

## LLaMa-1 Scoreboard

| Model                       | FP16  | Q8_0  | Q6_0  | Q5_0  | Q4_0  |
| :-                          | :-:   | :-:   | :-:   | :-:   | :-:   |
| `huggyllama/llama-7b`       | 5.673 | 5.676 | 5.709 | 5.884 | 6.842 | 
| `huggyllama/llama-13b`      | 5.087 | 
| `huggyllama/llama-30b`      | 
| `huggyllama/llama-65b`      | 

## LLaMa-2 Scoreboard

| Model                       | FP16  | Q8_0  | Q6_0  | Q5_0  | Q4_0  |
| :-                          | :-:   | :-:   | :-:   | :-:   | :-:   |
| `meta-llama/Llama-2-7b-hf`  | 5.468 | 5.471 | 5.524 | 5.677 | 7.134 |
| `meta-llama/Llama-2-13b-hf` | 4.880 | 4.883 | 4.899 | 4.973 | 5.398 |
| `meta-llama/Llama-2-70b-hf` | 
