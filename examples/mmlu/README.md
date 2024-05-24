# MMLU

The example demonstrates MMLU evaluation.
```bash
python examples/mmlu/run.py --model meta-llama/Meta-Llama-3-8B
```

## macro-average

| `model`                               | `FP16` | `Q8_0` | `Q7_0` | `Q6_0` | `Q5_0` | `Q4_0` | `Q3_0` |
| :-                                    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`                 | 0.3205 | 0.3166 | 0.3232 | 0.3202 | 0.3052 | 0.2283 | 0.0044 |
| `meta-llama/Llama-2-7b-chat-hf`       | 0.4688 | 0.4690 | 0.4668 | 0.4700 | 0.4495 | 0.3688 | 0.0000 |
| `meta-llama/Llama-2-7b-hf`            | 0.3941 | 0.3915 | 0.3958 | 0.3813 | 0.3941 | 0.2817 | 0.0133 |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 0.6700 | 0.6691 | 0.6718 | 0.6575 | 0.6411 | 0.5246 | 0.0011 |
| `meta-llama/Meta-Llama-3-8B`          | 0.6377 | 0.6352 | 0.6383 | 0.6267 | 0.6145 | 0.4603 | 0.0016 |
| `microsoft/Phi-3-medium-4k-instruct`  | 0.7675 | 0.7622 | 0.7567 | 0.7673 | 0.7519 | 0.6007 | 0.0017 |
| `microsoft/Phi-3-mini-4k-instruct`    | 0.6622 | 0.6625 | 0.6688 | 0.6650 | 0.6332 | 0.4106 | 0.0050 |
| `mistralai/Mistral-7B-Instruct-v0.1`  | 0.5452 | 0.5450 | 0.5469 | 0.5457 | 0.5482 | 0.5304 | 0.0000 |
| `mistralai/Mistral-7B-Instruct-v0.2`  | 0.6165 | 0.6155 | 0.6144 | 0.6139 | 0.6193 | 0.5795 | 0.0000 |
| `mistralai/Mistral-7B-Instruct-v0.3`  | 0.6170 | 0.6139 | 0.6146 | 0.6161 | 0.6139 | 0.5734 | 0.0003 |

## micro-average

| `model`                               | `FP16` | `Q8_0` | `Q7_0` | `Q6_0` | `Q5_0` | `Q4_0` | `Q3_0` |
| :-                                    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`                 | 0.3050 | 0.3044 | 0.3089 | 0.3207 | 0.2978 | 0.2162 | 0.0033 |
| `meta-llama/Llama-2-7b-chat-hf`       | 0.4572 | 0.4566 | 0.4553 | 0.4585 | 0.4415 | 0.3690 | 0.0000 |
| `meta-llama/Llama-2-7b-hf`            | 0.3703 | 0.3710 | 0.3730 | 0.3553 | 0.3690 | 0.2600 | 0.0118 |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 0.6408 | 0.6388 | 0.6427 | 0.6303 | 0.6133 | 0.5127 | 0.0007 |
| `meta-llama/Meta-Llama-3-8B`          | 0.6094 | 0.6074 | 0.6140 | 0.5976 | 0.5931 | 0.4481 | 0.0013 |
| `microsoft/Phi-3-medium-4k-instruct`  | 0.7740 | 0.7734 | 0.7681 | 0.7753 | 0.7590 | 0.5813 | 0.0020 |
| `microsoft/Phi-3-mini-4k-instruct`    | 0.6662 | 0.6669 | 0.6721 | 0.6656 | 0.6381 | 0.4167 | 0.0046 |
| `mistralai/Mistral-7B-Instruct-v0.1`  | 0.5284 | 0.5265 | 0.5278 | 0.5265 | 0.5291 | 0.5147 | 0.0000 |
| `mistralai/Mistral-7B-Instruct-v0.2`  | 0.6035 | 0.6055 | 0.6029 | 0.5996 | 0.6029 | 0.5617 | 0.0000 |
| `mistralai/Mistral-7B-Instruct-v0.3`  | 0.5983 | 0.5957 | 0.5983 | 0.5931 | 0.5963 | 0.5513 | 0.0007 |
