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

| Model                                   | 1024        | 2048        | 4096        | 8192        |
| :-                                      | :-          | :-          | :-          | :-          |
| `huggyllama/llama-7b`                   | 31.6 / 30.1 | 31.7 / 30.4 | 31.7 / 30.6 | 31.8 / 30.6 |
| `meta-llama/Llama-2-7b-hf`              | 39.6 / 37.2 | 39.0 / 36.8 | 39.8 / 37.0 | 39.3 / 37.0 |
| `meta-llama/Llama-2-7b-chat-hf`         | 47.0 / 45.8 | 46.9 / 45.7 | 47.0 / 45.8 | 47.2 / 45.9 |
| `meta-llama/Llama-2-13b-hf`             | 51.3 / 50.7 | 51.0 / 50.4 | 50.9 / 50.4 | 51.0 / 50.5 |
| `meta-llama/Llama-2-13b-chat-hf`        | 53.3 / 51.6 | 53.7 / 51.9 | 53.7 / 52.0 | 53.8 / 52.0 |
| `mistralai/Mistral-7B-v0.3`             | 53.6 / 52.5 | 54.3 / 53.0 | 54.0 / 52.9 | 54.0 / 52.8 |
| `mistralai/Mistral-7B-Instruct-v0.3`    | 61.7 / 59.9 | 61.8 / 59.8 | 62.0 / 60.1 | 61.5 / 59.7 |
| `meta-llama/Meta-Llama-3-8B`            | 64.1 / 61.3 | 64.1 / 61.2 | 64.1 / 61.2 | 64.4 / 61.4 |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | 66.9 / 63.9 | 66.9 / 64.0 | 66.9 / 64.1 | 66.8 / 63.8 |
| `meta-llama/Meta-Llama-3.1-8B`          | 65.6 / 64.1 | 65.6 / 64.0 | 65.6 / 64.1 | 65.6 / 64.0 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 67.0 / 67.5 | 67.3 / 67.6 | 67.1 / 67.5 | 67.2 / 67.6 |
| `microsoft/Phi-3-mini-4k-instruct`      | 68.8 / 68.7 | 68.8 / 68.7 | 68.8 / 68.7 | 68.8 / 68.7 |

### MMLU (5-shot)

Following table summarizes `total macro-average` / `total micro-average` for `maxlen={1024, 2048, 4096, 8192}`.

```bash
python examples/llama/reasoning_mmlu.py --model microsoft/Phi-3-mini-4k-instruct --rope_type simple --maxlen 2048 --shot 5
```

| Model                                    | 1024        | 2048        | 4096        | 8192        |
| :-                                       | :-          | :-          | :-          | :-          |
| `huggyllama/llama-7b`                    | 32.5 / 32.7 | 33.1 / 33.6 | 31.9 / 32.5 | 31.9 / 32.5 |
| `meta-llama/Llama-2-7b-hf`               | 47.0 / 45.3 | 46.7 / 45.0 | 47.1 / 45.3 | 46.8 / 45.1 |
| `meta-llama/Llama-2-7b-chat-hf`          | 49.2 / 47.2 | 48.9 / 47.1 | 49.1 / 47.3 | 49.1 / 47.2 |
| `meta-llama/Llama-2-13b-hf`              | 56.6 / 55.9 | 56.6 / 56.1 | 56.6 / 55.9 | 56.7 / 56.2 |
| `meta-llama/Llama-2-13b-chat-hf`         | 54.7 / 52.8 | 54.7 / 53.0 | 54.9 / 53.2 | 54.9 / 53.2 |
| `mistralai/Mistral-7B-v0.3`              | 60.9 / 59.8 | 61.1 / 60.2 | 60.8 / 59.8 | 61.1 / 60.0 |
| `mistralai/Mistral-7B-Instruct-v0.3`     | 62.9 / 62.4 | 62.6 / 61.7 | 63.1 / 62.1 | 62.9 / 61.8 |
| `meta-llama/Meta-Llama-3-8B`             | 63.7 / 62.8 | 64.3 / 63.0 | 64.1 / 63.2 | 64.3 / 63.2 |
| `meta-llama/Meta-Llama-3-8B-Instruct`    | 65.3 / 64.3 | 65.4 / 64.3 | 65.2 / 64.1 | 65.4 / 64.3 |
| `meta-llama/Meta-Llama-3.1-8B`           | 65.0 / 64.2 | 65.1 / 64.2 | 65.5 / 64.5 | 65.2 / 64.4 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct`  | 68.4 / 67.2 | 68.2 / 67.3 | 68.6 / 67.4 | 68.8 / 67.4 |
| `microsoft/Phi-3-mini-4k-instruct`       | 72.4 / 71.5 | 72.5 / 71.8 | 72.6 / 71.9 | 72.5 / 71.9 |
| `mistralai/Mistral-Nemo-Instruct-2407`   | -           |             | -           | -           |
| `mistralai/Mistral-Large-Instruct-2407`  | -           |             | -           | -           |
| `meta-llama/Meta-Llama-3-70B-Instruct`   | -           |             | -           | -           |
| `meta-llama/Meta-Llama-3.1-70B-Instruct` | -           |             | -           | -           |

### MMLU Related Results

The table below presents results from various sources. Note that the setups vary, so comparisons should be made cautiously.

| Model                  | 5-shot      | Source |
| :-                     | :-          | :-     |
| Llama 1 7B             | 35.1        | [Llama 2 Tech Report](https://arxiv.org/abs/2307.09288)
| Llama 1 13B            | 46.9        | [Llama 2 Tech Report](https://arxiv.org/abs/2307.09288)
| Llama 2 7B             | 45.3        | [Llama 2 Tech Report](https://arxiv.org/abs/2307.09288)
| Llama 2 7B             | 44.4        | [Mistral 7B Tech Report](https://arxiv.org/abs/2310.06825)
| Llama 2 7B             | 45.7        | [meta-llama/llama3](https://github.com/meta-llama/llama3)
| Llama 2 7B Instruct    | 34.1        | [meta-llama/llama3](https://github.com/meta-llama/llama3)
| Llama 2 13B            | 54.8        | [Llama 2 Tech Report](https://arxiv.org/abs/2307.09288)
| Llama 2 13B            | 55.6        | [Mistral 7B Tech Report](https://arxiv.org/abs/2310.06825)
| Llama 2 13B            | 53.8        | [meta-llama/llama3](https://github.com/meta-llama/llama3)
| Llama 2 13B Instruct   | 47.8        | [meta-llama/llama3](https://github.com/meta-llama/llama3)
| Llama 3 8B             | 66.6 / 65.4 | [meta-llama/llama3](https://github.com/meta-llama/llama3)
| Llama 3 8B Instruct    | 68.4 / 67.4 | [meta-llama/llama3](https://github.com/meta-llama/llama3)
| Llama 3 8B Instruct    | 66.5        | [Phi-3 Tech Report](https://arxiv.org/abs/2404.14219)
| Llama 3 70B            | 79.5 / 78.9 | [meta-llama/llama3](https://github.com/meta-llama/llama3)
| Llama 3 70B Instruct   | 82.0 / 82.0 | [meta-llama/llama3](https://github.com/meta-llama/llama3)
| Llama 3.1 8B           | 66.7 / 65.6 | [meta-llama/llama-models](https://github.com/meta-llama/llama-models)
| Llama 3.1 8B Instruct  | 69.4 / 69.4 | [meta-llama/llama-models](https://github.com/meta-llama/llama-models)
| Llama 3.1 70B          | 79.3 / 79.0 | [meta-llama/llama-models](https://github.com/meta-llama/llama-models)
| Llama 3.1 70B Instruct | 83.6 / 84.0 | [meta-llama/llama-models](https://github.com/meta-llama/llama-models)
| phi-3-mini             | 68.8        | [Phi-3 Tech Report](https://arxiv.org/abs/2404.14219)
| phi-3-small            | 75.7        | [Phi-3 Tech Report](https://arxiv.org/abs/2404.14219)
| phi-3-medium           | 78.0        | [Phi-3 Tech Report](https://arxiv.org/abs/2404.14219)
| Mistral-7b-v0.1        | 60.1        | [Mistral 7B Tech Report](https://arxiv.org/abs/2310.06825)
| Mistral-7b-v0.1        | 61.7        | [Phi-3 Tech Report](https://arxiv.org/abs/2404.14219)
| Mistral NeMo 12B       | 68.0        | [Mistral News](https://mistral.ai/news/mistral-nemo/)
| Mistral Large 2 (2407) | 84.0        | [Mistral News](https://mistral.ai/news/mistral-large-2407/)
