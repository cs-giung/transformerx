# LLaMa Scoreboard

## Perplexity

The example demonstrates computing perplexity.

```bash
python examples/llama/perplexity.py
    --model_name huggyllama/llama-7b
    --data_name wikitext2
    --quantization {...}
```

| Model                       | `FP16` |     | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |     | `Q8_1` | `Q6_1` | `Q5_1` | `Q4_1` |
| :-                          | :-:    | :-: | :-:    | :-:    | :-:    | :-:    | :-: | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`       | 5.673  |     | 5.676  | 5.709  | 5.884  | 6.842  |     | 5.676  | 5.697  | 5.863  | 6.284  |
| `huggyllama/llama-13b`      | 5.087  |     | 5.088  | 5.095  | 5.196  | 5.719  |     | 5.086  | 5.098  | 5.176  | 5.523  |
| `huggyllama/llama-30b`      | 4.098  |     | 4.102  | 4.121  | 4.208  | 4.743  |     | 4.099  | 4.108  | 4.187  | 4.539  |
| `huggyllama/llama-65b`      | 3.531  |     | 3.531  | 3.557  | 3.633  | 4.393  |     | 3.531  | 3.551  | 3.606  | 3.918  |
| `meta-llama/Llama-2-7b-hf`  | 5.468  |     | 5.471  | 5.524  | 5.677  | 7.134  |     | 5.471  | 5.520  | 5.651  | 6.108  |
| `meta-llama/Llama-2-13b-hf` | 4.880  |     | 4.883  | 4.899  | 4.973  | 5.398  |     | 4.882  | 4.903  | 4.960  | 5.203  |
| `meta-llama/Llama-2-70b-hf` | 3.317  |     | 3.319  | 3.344  | 3.436  | 3.865  |     | 3.319  | 3.335  | 3.415  | 3.670  |

| Model                       | `FP16` | `Q8_1` | `Q6_1` | `Q5_1` | `Q4_1` |
| :-                          | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`       | 5.673  | 5.676  | 5.697  | 5.863  | 6.284  |
| `huggyllama/llama-13b`      | 5.087  | 5.086  | 5.098  | 5.176  | 5.523  |
| `huggyllama/llama-30b`      | 4.098  | 4.099  | 4.108  | 4.187  | 4.539  |
| `huggyllama/llama-65b`      | 3.531  | 3.531  | 3.551  | 3.606  | 3.918  |
| `meta-llama/Llama-2-7b-hf`  | 5.468  | 5.471  | 5.520  | 5.651  | 6.108  |
| `meta-llama/Llama-2-13b-hf` | 4.880  | 4.882  | 4.903  | 4.960  | 5.203  |
| `meta-llama/Llama-2-70b-hf` | 3.317  | 3.319  | 3.335  | 3.415  | 3.670  |
