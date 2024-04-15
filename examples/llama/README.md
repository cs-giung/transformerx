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

## Accuracy

The example demonstrates computing accuracy on zero-shot tasks.

```bash
python examples/llama/accuracy.py
    --model_name huggyllama/llama-7b
    --data_name arc_e
    --quantization {...}
```

### Classification accuracy (%)

| Model                       | `FP16` |     | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |     | `Q8_1` | `Q6_1` | `Q5_1` | `Q4_1` |
| :-                          | :-:    | :-: | :-:    | :-:    | :-:    | :-:    | :-: | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`       | 72.63  |     | 72.81  | 72.11  | 72.28  | 70.35  |     | 72.81  | 73.51  | 72.98  | 72.28  |
| `huggyllama/llama-13b`      | 75.26  |     | 75.61  | 75.79  | 75.79  | 72.46  |
| `huggyllama/llama-30b`      | 80.18  |     | 79.65  | 80.88  | 80.18  | 77.02  |
| `huggyllama/llama-65b`      |        |     |
| `meta-llama/Llama-2-7b-hf`  | 74.56  |     | 74.74  | 75.44  | 74.39  | 71.23  |
| `meta-llama/Llama-2-13b-hf` | 77.19  |     | 77.72  | 77.37  | 77.19  | 75.44  |
| `meta-llama/Llama-2-70b-hf` |        |     |

### Categorical negative log-likelihood

| Model                       | `FP16` |     | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |     | `Q8_1` | `Q6_1` | `Q5_1` | `Q4_1` |
| :-                          | :-:    | :-: | :-:    | :-:    | :-:    | :-:    | :-: | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`       | 1.100  |     | 1.104  | 1.117  | 1.119  | 1.368  | 
| `huggyllama/llama-13b`      | .9724  |     | .9702  | .9769  | .9879  | 1.117  |
| `huggyllama/llama-30b`      | .8401  |     | .8428  | .8267  | .8581  | .9274  |
| `huggyllama/llama-65b`      |        |     |
| `meta-llama/Llama-2-7b-hf`  | 1.081  |     | 1.082  | 1.078  | 1.111  | 1.328  |
| `meta-llama/Llama-2-13b-hf` | .9345  |     | .9362  | .9443  | .9410  | .9746  |
| `meta-llama/Llama-2-70b-hf` |        |     |

### Expected calibration error (%)

| Model                       | `FP16` |     | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |     | `Q8_1` | `Q6_1` | `Q5_1` | `Q4_1` |
| :-                          | :-:    | :-: | :-:    | :-:    | :-:    | :-:    | :-: | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-7b`       | 10.63  |     | 10.49  | 10.67  | 10.81  | 13.33  |
| `huggyllama/llama-13b`      | 8.412  |     | 9.252  | 8.279  | 8.350  | 10.33  |
| `huggyllama/llama-30b`      | 7.889  |     | 8.605  | 8.118  | 9.074  | 8.590  |
| `huggyllama/llama-65b`      |        |     |
| `meta-llama/Llama-2-7b-hf`  | 10.41  |     | 10.78  | 10.29  | 10.58  | 13.63  |
| `meta-llama/Llama-2-13b-hf` | 8.983  |     | 9.328  | 8.868  | 9.278  | 9.611  |
| `meta-llama/Llama-2-70b-hf` |        |     |