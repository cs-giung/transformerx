# Perplexity

The example demonstrates computing perplexity.
```bash
python examples/perplexity/run.py --model meta-llama/Meta-Llama-3-8B
```

## WikiText-2

| `model`                               | `FP16`| `Q8_0`| `Q7_0`| `Q6_0`| `Q5_0`| `Q4_0`| `Q3_0`|
| :-                                    | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   |
| `huggyllama/llama-7b`                 | 5.873 | 5.862 | 5.854 | 5.921 | 6.009 | 6.264 | 10.72 |
| `meta-llama/Llama-2-7b-hf`            | 5.834 | 5.822 | 5.827 | 5.855 | 6.031 | 6.184 | 10.55 |
| `meta-llama/Llama-2-7b-chat-hf`       | 5.763 | 5.762 | 5.766 | 5.791 | 5.959 | 6.379 | 10.12 |
| `meta-llama/Meta-Llama-3-8B`          | 6.501 | 6.512 | 6.509 | 6.511 | 6.646 | 8.313 | 12.66 | 
| `meta-llama/Meta-Llama-3-8B-Instruct` | 6.726 | 6.747 | 6.752 | 6.746 | 6.887 | 8.606 | 13.79 |
| `microsoft/Phi-3-medium-4k-instruct`  | 5.324 | 5.321 | 5.320 | 5.325 | 5.371 | 5.325 | 9.600 |
| `microsoft/Phi-3-mini-4k-instruct`    | 5.547 | 5.555 | 5.553 | 5.545 | 5.559 | 5.712 | 13.85 |
| `mistralai/Mistral-7B-Instruct-v0.1`  | 5.503 | 5.504 | 5.510 | 5.517 | 5.535 | 5.689 | 8.784 |
| `mistralai/Mistral-7B-Instruct-v0.2`  | 5.982 | 5.996 | 6.086 | 6.878 | 7.174 | 7.091 | 9.630 |
| `mistralai/Mistral-7B-Instruct-v0.3`  | 5.793 | 5.778 | 5.827 | 6.222 | 6.837 | 6.830 | 9.628 |
