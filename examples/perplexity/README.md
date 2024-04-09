# transformerx/examples/perplexity

The example demonstrates computing perplexity.

```bash
python examples/perplexity/perplexity.py
    --model_name meta-llama/Llama-2-7b-hf
    --data_name wikitext2
```

## LLaMa Scoreboard

### Symmetric Quantization

| Model                       | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                          | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`       | 5.673  | 5.676  | 5.709  | 5.884  | 6.842  |
| `huggyllama/llama-13b`      | 5.087  | 5.088  | 5.095  | 5.196  | 5.719  |
| `huggyllama/llama-30b`      | 4.098  | 4.102  | 4.121  | 4.208  | 4.743  |
| `huggyllama/llama-65b`      | 3.531  | 3.531  | 3.557  | 3.633  | 4.393  |
| `meta-llama/Llama-2-7b-hf`  | 5.468  | 5.471  | 5.524  | 5.677  | 7.134  |
| `meta-llama/Llama-2-13b-hf` | 4.880  | 4.883  | 4.899  | 4.973  | 5.398  |
| `meta-llama/Llama-2-70b-hf` | 3.317  | 

### Asymmetric Quantization

| Model                       | `FP16` | `Q8_1` | `Q6_1` | `Q5_1` | `Q4_1` |
| :-                          | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`       | 5.673  | 
| `huggyllama/llama-13b`      | 5.087  | 
| `huggyllama/llama-30b`      | 4.098  | 
| `huggyllama/llama-65b`      | 3.531  | 
| `meta-llama/Llama-2-7b-hf`  | 5.468  | 5.471  | 5.520  | 5.651  | 6.108  |
| `meta-llama/Llama-2-13b-hf` | 4.880  | 
| `meta-llama/Llama-2-70b-hf` | 3.317  |  
