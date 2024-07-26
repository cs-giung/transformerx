# Llama Examples

> [!NOTE]
> Experiments are mainly carried out in TPU VM environments, with support for Cloud TPUs provided by Google's TPU Research Cloud (TRC). Although there may be minor differences in the results based on the specific execution environment, these variations are expected to be minimal.

## Perplexity

The `perplexity` example computes how well a language model predicts the next word in a text, with lower scores indicating better performance. Perplexity scores cannot be directly compared between models.

### WikiText-2

Following table summarizes `perplexity` for `seqlen={1024, 2048, 4096, 8192}`.

```bash
python examples/llama/perplexity.py --model microsoft/Phi-3-mini-4k-instruct --rope_type simple --seqlen 2048 --data wikitext2
```

| Model                                   | 1024   | 2048   | 4096   | 8192   |
| :-                                      | :-     | :-     | :-     | :-     |
| `mistralai/Mistral-7B-v0.3`             | 6.065  | 5.380  | 4.983  | 4.764  |
| `mistralai/Mistral-7B-Instruct-v0.3`    | 6.459  | 5.675  | 5.219  | 4.968  |
| `meta-llama/Meta-Llama-3-8B`            | 6.937  | 6.191  | 5.780  | 5.552  |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | 9.256  | 8.240  | 7.690  | 7.405  |
| `meta-llama/Meta-Llama-3.1-8B`          | 7.078  | 6.294  | 5.866  | 5.623  |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8.093  | 7.227  | 6.744  | 6.473  |
| `microsoft/Phi-3-mini-4k-instruct`      | 6.760  | 6.032  | 5.645  | 5.478  |

### PTB

Following table summarizes `perplexity` for `seqlen={1024, 2048, 4096, 8192}`.

```bash
python examples/llama/perplexity.py --model microsoft/Phi-3-mini-4k-instruct --rope_type simple --seqlen 2048 --data ptb
```

| Model                                   | 1024   | 2048   | 4096   | 8192   |
| :-                                      | :-     | :-     | :-     | :-     |
| `mistralai/Mistral-7B-v0.3`             | 37.18  | 38.97  | 48.37  | 73.29  |
| `mistralai/Mistral-7B-Instruct-v0.3`    | 33.21  | 28.94  | 27.62  | 28.64  |
| `meta-llama/Meta-Llama-3-8B`            | 12.40  | 11.21  | 10.55  | 10.18  |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | 15.92  | 14.40  | 13.59  | 13.14  |
| `meta-llama/Meta-Llama-3.1-8B`          | 12.33  | 11.16  | 10.52  | 10.16  |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 14.40  | 13.05  | 12.39  | 12.04  |
| `microsoft/Phi-3-mini-4k-instruct`      | 14.28  | 12.85  | 12.16  | 11.85  |

## Reasoning

The `reasoning` example assesses a language model's ability to use common sense or world knowledge to make inferences. Reasoning scores reflect the knowledge gained during pretraining by evaluating the models exclusively in zero-shot and few-shot scenarios.

### MMLU (0-shot)

Following table summarizes `total macro-average` / `total micro-average` for `maxlen={1024, 2048, 4096, 8192}`.

```bash
python examples/llama/reasoning_mmlu.py --model microsoft/Phi-3-mini-4k-instruct --rope_type simple --maxlen 2048 --shot 0
```

| Model                                   | 1024            | 2048            | 4096            | 8192            |
| :-                                      | :-              | :-              | :-              | :-              |
| `huggyllama/llama-7b`                   | 0.3156 / 0.3005 | 0.3171 / 0.3037 | 0.3173 / 0.3063 | 0.3178 / 0.3063 |
| `meta-llama/Llama-2-7b-hf`              | 0.3960 / 0.3717 | 0.3902 / 0.3677 | 0.3976 / 0.3703 | 0.3926 / 0.3697 |
| `meta-llama/Llama-2-7b-chat-hf`         | 0.4699 / 0.4579 | 0.4688 / 0.4572 | 0.4699 / 0.4579 | 0.4721 / 0.4592 |
| `meta-llama/Llama-2-13b-hf`             | 0.5127 / 0.5069 | 0.5095 / 0.5042 | 0.5094 / 0.5042 | 0.5098 / 0.5049 |
| `meta-llama/Llama-2-13b-chat-hf`        | 0.5333 / 0.5160 | 0.5368 / 0.5186 | 0.5372 / 0.5199 | 0.5383 / 0.5199 |
| `mistralai/Mistral-7B-v0.3`             | 0.5364 / 0.5245 | 0.5425 / 0.5304 | 0.5404 / 0.5291 | 0.5397 / 0.5284 |
| `mistralai/Mistral-7B-Instruct-v0.3`    | 0.6169 / 0.5990 | 0.6175 / 0.5983 | 0.6202 / 0.6009 | 0.6154 / 0.5970 |
| `meta-llama/Meta-Llama-3-8B`            | 0.6405 / 0.6133 | 0.6413 / 0.6120 | 0.6409 / 0.6120 | 0.6437 / 0.6140 |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | 0.6686 / 0.6388 | 0.6689 / 0.6395 | 0.6693 / 0.6408 | 0.6682 / 0.6381 |
| `meta-llama/Meta-Llama-3.1-8B`          | 0.6564 / 0.6414 | 0.6557 / 0.6401 | 0.6560 / 0.6414 | 0.6558 / 0.6401 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 0.6701 / 0.6747 | 0.6734 / 0.6760 | 0.6710 / 0.6754 | 0.6723 / 0.6760 |
| `microsoft/Phi-3-mini-4k-instruct`      | 0.6876 / 0.6871 | 0.6876 / 0.6871 | 0.6876 / 0.6871 | 0.6877 / 0.6871 |

### MMLU (5-shot)

Following table summarizes `total macro-average` / `total micro-average` for `maxlen={1024, 2048, 4096, 8192}`.

```bash
python examples/llama/reasoning_mmlu.py --model microsoft/Phi-3-mini-4k-instruct --rope_type simple --maxlen 2048 --shot 5
```

| Model                                   | 1024            | 2048            | 4096            | 8192            |
| :-                                      | :-              | :-              | :-              | :-              |
| `huggyllama/llama-7b`                   | 0.3246 / 0.3266 | 0.3307 / 0.3357 | 0.3187 / 0.3246 | 0.3185 / 0.3246 |
| `meta-llama/Llama-2-7b-hf`              | 0.4702 / 0.4526 | 0.4668 / 0.4500 | 0.4705 / 0.4526 | 0.4678 / 0.4507 |
| `meta-llama/Llama-2-7b-chat-hf`         | 0.4918 / 0.4716 | 0.4893 / 0.4709 | 0.4913 / 0.4729 | 0.4911 / 0.4716 |
| `meta-llama/Llama-2-13b-hf`             | 0.5656 / 0.5591 | 0.5658 / 0.5611 | 0.5656 / 0.5591 | 0.5672 / 0.5617 |
| `meta-llama/Llama-2-13b-chat-hf`        | 0.5465 / 0.5284 | 0.5474 / 0.5304 | 0.5486 / 0.5317 | 0.5485 / 0.5323 |
| `mistralai/Mistral-7B-v0.3`             | 0.6094 / 0.5976 | 0.6105 / 0.6016 | 0.6081 / 0.5976 | 0.6105 / 0.6003 |
| `mistralai/Mistral-7B-Instruct-v0.3`    | 0.6287 / 0.6238 | 0.6262 / 0.6166 | 0.6312 / 0.6212 | 0.6287 / 0.6179 |
| `meta-llama/Meta-Llama-3-8B`            | 0.6372 / 0.6277 | 0.6428 / 0.6303 | 0.6405 / 0.6316 | 0.6431 / 0.6316 |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | 0.6530 / 0.6427 | 0.6537 / 0.6427 | 0.6522 / 0.6414 | 0.6538 / 0.6427 |
| `meta-llama/Meta-Llama-3.1-8B`          | 0.6497 / 0.6421 | 0.6508 / 0.6421 | 0.6545 / 0.6453 | 0.6520 / 0.6440 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 0.6842 / 0.6715 | 0.6821 / 0.6728 | 0.6855 / 0.6741 | 0.6875 / 0.6741 |
| `microsoft/Phi-3-mini-4k-instruct`      | 0.7244 / 0.7146 | 0.7252 / 0.7178 | 0.7261 / 0.7191 | 0.7251 / 0.7185 |
