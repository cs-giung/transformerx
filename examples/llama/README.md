# Llama Examples

> [!NOTE]
> Experiments are mainly carried out in TPU VM environments, with support for Cloud TPUs provided by Google's TPU Research Cloud (TRC). Although there may be minor differences in the results based on the specific execution environment, these variations are expected to be minimal.

## Perplexity

The `perplexity` example computes how well a language model predicts the next word in a text, with lower scores indicating better performance. Perplexity scores cannot be directly compared between models.

## Reasoning

The `reasoning` example assesses a language model's ability to use common sense or world knowledge to make inferences. Reasoning scores reflect the knowledge gained during pretraining by evaluating the models exclusively in zero-shot and few-shot scenarios.

### MMLU (0-shot)

```bash
python examples/llama/reasoning_mmlu.py \
    --model microsoft/Phi-3-mini-4k-instruct
    --rope_type simple \
    --maxlen 2048 \
    --shot 0
```

#### Macro-average

| Model                                   | `FP16` |
| :-                                      | :-     |
| `meta-llama/Meta-Llama-3-8B`            | 0.6413 |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | 0.6689 |
| `meta-llama/Meta-Llama-3.1-8B`          | 0.6557 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 0.6734 |
| `microsoft/Phi-3-mini-4k-instruct`      | 0.6876 |

#### Micro-average

| Model                                   | `FP16` |
| :-                                      | :-     |
| `meta-llama/Meta-Llama-3-8B`            | 0.6120 |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | 0.6395 |
| `meta-llama/Meta-Llama-3.1-8B`          | 0.6401 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 0.6760 |
| `microsoft/Phi-3-mini-4k-instruct`      | 0.6871 |

### MMLU (5-shot)

```bash
python examples/llama/reasoning_mmlu.py \
    --model microsoft/Phi-3-mini-4k-instruct
    --rope_type simple \
    --maxlen 2048 \
    --shot 5
```

#### Macro-average

| Model                                   | `FP16` |
| :-                                      | :-     |
| `meta-llama/Meta-Llama-3-8B`            | 0.6428 |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | 0.6537 |
| `meta-llama/Meta-Llama-3.1-8B`          | 0.6508 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 0.6821 |
| `microsoft/Phi-3-mini-4k-instruct`      | 0.7252 |

#### Micro-average

| Model                                   | `FP16` |
| :-                                      | :-     |
| `meta-llama/Meta-Llama-3-8B`            | 0.6303 |
| `meta-llama/Meta-Llama-3-8B-Instruct`   | 0.6427 |
| `meta-llama/Meta-Llama-3.1-8B`          | 0.6421 |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 0.6728 |
| `microsoft/Phi-3-mini-4k-instruct`      | 0.7178 |
