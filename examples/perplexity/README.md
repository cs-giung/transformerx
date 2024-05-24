# Perplexity

The example demonstrates computing perplexity.
```bash
python examples/perplexity/run.py --model meta-llama/Meta-Llama-3-8B
```

## WikiText-2

| `model`                               | `FP16`| `Q8_0`| `Q7_0`| `Q6_0`| `Q5_0`| `Q4_0`| `Q3_0`|
| :-                                    | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   |
| `huggyllama/llama-7b`                 | 5.873 | 5.862 | 5.854 | 5.921 | 6.009 | 6.264 | 10.72 |
| `meta-llama/Llama-2-7b-chat-hf`       | 5.763 | 5.762 | 5.766 | 5.791 | 5.959 | 6.379 | 10.12 |
| `meta-llama/Llama-2-7b-hf`            | 5.834 | 5.822 | 5.827 | 5.855 | 6.031 | 6.184 | 10.55 |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 6.726 | 6.747 | 6.752 | 6.746 | 6.887 | 8.606 | 13.79 |
| `meta-llama/Meta-Llama-3-8B`          | 6.501 | 6.512 | 6.509 | 6.511 | 6.646 | 8.313 | 12.66 |
| `microsoft/Phi-3-medium-4k-instruct`  | 5.324 | 5.321 | 5.320 | 5.325 | 5.371 | 5.325 | 9.600 |
| `microsoft/Phi-3-mini-4k-instruct`    | 5.547 | 5.555 | 5.553 | 5.545 | 5.559 | 5.712 | 13.85 |
| `mistralai/Mistral-7B-Instruct-v0.1`  | 5.503 | 5.504 | 5.510 | 5.517 | 5.535 | 5.689 | 8.784 |
| `mistralai/Mistral-7B-Instruct-v0.2`  | 5.982 | 5.996 | 6.086 | 6.878 | 7.174 | 7.091 | 9.630 |
| `mistralai/Mistral-7B-Instruct-v0.3`  | 5.793 | 5.778 | 5.827 | 6.222 | 6.837 | 6.830 | 9.628 |

## PTB

| `model`                               | `FP16`| `Q8_0`| `Q7_0`| `Q6_0`| `Q5_0`| `Q4_0`| `Q3_0`|
| :-                                    | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   | :-:   |
| `huggyllama/llama-7b`                 | 6.907 | 6.912 | 6.893 | 6.910 | 6.963 | 6.860 | 11.10 |
| `meta-llama/Llama-2-7b-chat-hf`       | 6.524 | 6.519 | 6.526 | 6.579 | 6.613 | 6.846 | 10.65 |
| `meta-llama/Llama-2-7b-hf`            | 6.680 | 6.677 | 6.680 | 6.691 | 6.768 | 6.769 | 11.98 |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 7.012 | 7.051 | 7.024 | 7.044 | 7.013 | 8.437 | 13.39 |
| `meta-llama/Meta-Llama-3-8B`          | 6.861 | 6.898 | 6.894 | 6.873 | 7.033 | 8.409 | 12.72 |
| `microsoft/Phi-3-medium-4k-instruct`  | 5.562 | 5.565 | 5.561 | 5.554 | 5.582 | 5.643 | 9.580 |
| `microsoft/Phi-3-mini-4k-instruct`    | 5.625 | 5.632 | 5.641 | 5.629 | 5.625 | 5.932 | 13.88 |
| `mistralai/Mistral-7B-Instruct-v0.1`  | 6.804 | 6.802 | 6.817 | 6.843 | 6.725 | 6.934 | 8.812 |
| `mistralai/Mistral-7B-Instruct-v0.2`  | 6.667 | 6.682 | 6.723 | 7.426 | 7.955 | 7.931 | 9.888 |
| `mistralai/Mistral-7B-Instruct-v0.3`  | 6.907 | 6.895 | 6.907 | 7.001 | 7.530 | 7.957 | 9.806 |
