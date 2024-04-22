# LLaMa Scoreboard

## MMLU

The example demonstrates MMLU evaluation.

```bash
python examples/llama/mmlu.py
    --model_name huggyllama/llama-7b
    --task_name hendrycks_test
    --n_fewshot 5
    --quantization {...}
```

### Macro-average

| Model                         | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                            | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`         | 33.57  | 34.54  | 34.12  | 35.52  | 27.36  |
| `huggyllama/llama-13b`        | 47.60  | 47.85  | 46.92  | 46.22  | 43.45  |
| `huggyllama/llama-30b`        | 54.75  | 54.40  | 55.05  | 54.75  | 51.99  |
| `meta-llama/Llama-2-7b-hf`    | 47.05  | 46.24  | 45.95  | 43.51  | 38.45  |
| `meta-llama/Llama-2-13b-hf`   | 56.70  | 56.66  | 55.57  | 57.27  | 52.08  |
| `meta-llama/Meta-Llama-3-8B`  | 63.97  | 63.84  | 63.06  | 61.11  | 47.40  |

### Micro-average

| Model                         | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                            | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`         | 33.96  | 34.29  | 34.55  | 34.68  | 27.69  |
| `huggyllama/llama-13b`        | 47.16  | 46.96  | 46.18  | 45.33  | 42.52  |
| `huggyllama/llama-30b`        | 54.93  | 55.06  | 55.26  | 54.80  | 52.78  |
| `meta-llama/Llama-2-7b-hf`    | 45.20  | 44.48  | 44.28  | 42.52  | 37.17  |
| `meta-llama/Llama-2-13b-hf`   | 56.17  | 56.04  | 54.74  | 54.54  | 51.08  |
| `meta-llama/Meta-Llama-3-8B`  | 62.83  | 62.97  | 61.53  | 60.48  | 45.79  |

## Perplexity

The example demonstrates computing perplexity.

```bash
python examples/llama/perplexity.py
    --model_name huggyllama/llama-7b
    --data_name wikitext2
    --quantization {...}
```

| Model                         | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                            | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`         | 5.673  | 5.676  | 5.709  | 5.884  | 6.842  |
| `huggyllama/llama-13b`        | 5.087  | 5.088  | 5.095  | 5.196  | 5.719  |
| `huggyllama/llama-30b`        | 4.098  | 4.102  | 4.121  | 4.208  | 4.743  |
| `huggyllama/llama-65b`        | 3.531  | 3.531  | 3.557  | 3.633  | 4.393  |
| `meta-llama/Llama-2-7b-hf`    | 5.468  | 5.471  | 5.524  | 5.677  | 7.134  |
| `meta-llama/Llama-2-13b-hf`   | 4.880  | 4.883  | 4.899  | 4.973  | 5.398  |
| `meta-llama/Llama-2-70b-hf`   | 3.317  | 3.319  | 3.344  | 3.436  | 3.865  |
| `meta-llama/Meta-Llama-3-8B`  | 6.130  | 6.138  | 6.295  | 6.649  | 10.51  |
