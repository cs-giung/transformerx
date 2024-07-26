# Perplexity

> [!NOTE]
> Experiments are primarily conducted in TPU VM environments, with support from Cloud TPUs provided by Google's TPU Research Cloud (TRC).

The example demonstrates computing perplexity.
```bash
python examples/perplexity/run.py --model meta-llama/Meta-Llama-3-8B
```

## WikiText-2

| `model`                               | `FP16`| 
| :-                                    | :-:   | 
| `huggyllama/llama-7b`                 | 5.707 | 
| `meta-llama/Llama-2-7b-chat-hf`       | 6.997 | 
| `meta-llama/Llama-2-7b-hf`            | 5.497 | 
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8.240 | 
| `meta-llama/Meta-Llama-3-8B`          | 6.191 | 
| `microsoft/Phi-3-medium-4k-instruct`  | 4.307 | 
| `microsoft/Phi-3-mini-4k-instruct`    | 6.406 | 
| `mistral-community/Mistral-7B-v0.2`   | 5.380 | 
| `mistralai/Mistral-7B-Instruct-v0.1`  | 6.902 | 
| `mistralai/Mistral-7B-Instruct-v0.2`  | 6.130 | 
| `mistralai/Mistral-7B-Instruct-v0.3`  | 5.675 | 
| `mistralai/Mistral-7B-v0.1`           | 5.320 | 
| `mistralai/Mistral-7B-v0.3`           | 5.380 | 

## PTB

| `model`                               | `FP16`| 
| :-                                    | :-:   | 
| `huggyllama/llama-7b`                 | 46.56 | 
| `meta-llama/Llama-2-7b-chat-hf`       | 34.87 | 
| `meta-llama/Llama-2-7b-hf`            | 32.52 | 
| `meta-llama/Meta-Llama-3-8B-Instruct` | 14.40 | 
| `meta-llama/Meta-Llama-3-8B`          | 11.21 | 
| `microsoft/Phi-3-medium-4k-instruct`  | 10.52 | 
| `microsoft/Phi-3-mini-4k-instruct`    | 13.34 | 
| `mistral-community/Mistral-7B-v0.2`   | 37.35 | 
| `mistralai/Mistral-7B-Instruct-v0.1`  | 24.78 | 
| `mistralai/Mistral-7B-Instruct-v0.2`  | 31.87 | 
| `mistralai/Mistral-7B-Instruct-v0.3`  | 28.94 | 
| `mistralai/Mistral-7B-v0.1`           | 35.50 | 
| `mistralai/Mistral-7B-v0.3`           | 38.97 | 
