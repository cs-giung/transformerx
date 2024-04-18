# LLaMa Scoreboards

## Zero-shot Results

<details>
  <summary>See example prompt:</summary>

```python
>>> from transformerx.tasks import ARCEasy
>>> task = ARCEasy()
>>> 
>>> task.create_qa_prompt_choices_fewshot([], task.valid_docs()[0])
'The following are multiple choice questions (with answers) about common sense reasoning.\n\nWhich technology was developed most recently?\nA. cellular telephone\nB. television\nC. refrigerator\nD. airplane\nAnswer:'
>>> 
>>> print(task.create_qa_prompt_choices_fewshot([], task.valid_docs()[0]))
The following are multiple choice questions (with answers) about common sense reasoning.

Which technology was developed most recently?
A. cellular telephone
B. television
C. refrigerator
D. airplane
Answer:
>>> 
```
</details>

### Accuracy

| Model                  | Task             | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                     | :-               | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-13b` | `arc_e`          | 69.47  | 69.47  | 69.12  | 66.49  | 55.26  |
|                        | `arc_c`          | 47.16  | 46.82  | 46.82  | 42.81  | 32.78  |
|                        | `commonsense_qa` | 54.30  | 54.63  | 54.05  | 50.78  | 37.18  |
|                        | `piqa`           | 64.80  | 65.23  | 64.58  | 65.02  | 61.43  |
| `huggyllama/llama-30b` | `arc_e`          |        | 85.09  | 83.33  | 83.68  | 81.40  |
|                        | `arc_c`          |        | 68.23  | 68.23  | 66.89  | 60.54  |
|                        | `commonsense_qa` |        | 64.86  | 64.78  | 62.82  | 62.00  |
|                        | `piqa`           |        | 71.65  | 70.84  | 67.95  | 68.28  |

### Negative Log-Likelihood

| Model                  | Task             | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                     | :-               | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-13b` | `arc_e`          | .9329  | .9319  | .9391  | .9798  | 1.127  |
|                        | `arc_c`          | 1.238  | 1.237  | 1.240  | 1.249  | 1.385  |
|                        | `commonsense_qa` | 1.268  | 1.270  | 1.281  | 1.321  | 1.481  |
|                        | `piqa`           | .6223  | .6226  | .6215  | .6266  | .6434  |
| `huggyllama/llama-30b` | `arc_e`          |        | .5345  | .5260  | .5271  | .5939  |
|                        | `arc_c`          |        | .8838  | .8892  | .8996  | .9907  |
|                        | `commonsense_qa` |        | 1.014  | 1.007  | 1.043  | 1.091  |
|                        | `piqa`           |        | .5503  | .5511  | .5660  | .5667  |

### Expected Calibration Error

| Model                  | Task             | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                     | :-               | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-13b` | `arc_e`          | 21.88  | 21.76  | 21.49  | 21.00  | 16.83  |
|                        | `arc_c`          | 8.796  | 8.751  | 7.797  | 7.011  | 7.332  |
|                        | `commonsense_qa` | 17.12  | 17.59  | 17.31  | 15.66  | 7.607  |
|                        | `piqa`           | 3.981  | 4.663  | 4.032  | 6.574  | 2.515  |
| `huggyllama/llama-30b` | `arc_e`          |        | 15.42  | 12.82  | 11.81  | 13.02  |
|                        | `arc_c`          |        | 11.60  | 9.851  | 8.263  | 8.313  |
|                        | `commonsense_qa` |        | 11.62  | 10.19  | 9.189  | 11.65  |
|                        | `piqa`           |        | 3.135  | 2.470  | 3.291  | 2.733  |

## MMLU Results (5-shot)

<details>
  <summary>See example prompt:</summary>

```python
>>> from transformerx.tasks import HendrycksTest
>>> task = HendrycksTest(subject='formal_logic')
>>>
>>> task.create_qa_prompt_choices_fewshot(task.kshot_docs()[:5], task.valid_docs()[0])
"The following are multiple choice questions (with answers) about formal logic.\n\nSelect the best translation into predicate logic: No people drive on Mars.\nA. ~Pd\nB. (∀x)(Px ∨ ~Dx)\nC. (∀x)(Px ⊃ ~Dx)\nD. ~Dp\nAnswer: C\n\nSelect the best translation into predicate logic.George borrows Hector's lawnmower. (g: George; h: Hector; l: Hector's lawnmower; Bxyx: x borrows y from z)\nA. Blgh\nB. Bhlg\nC. Bglh\nD. Bghl\nAnswer: C\n\nSelect the best English interpretation of the given arguments in predicate logic.\nDm\n(∀x)(Wx ⊃ ~Dx)\n(∀x)Wx ∨ Ag\t/ (∃x)Ax\nA. Marina is a dancer. Some weaklings are not dancers. Either everything is aweakling or Georgia plays volleyball. So something plays volleyball.\nB. Marina is a dancer. No weakling is a dancer. Everything is either a weakling or plays volleyball. So something plays volleyball.\nC. Marina is a dancer. Some weaklings are not dancers. Everything is either a weakling or plays volleyball. So something plays volleyball.\nD. Marina is a dancer. No weakling is a dancer. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.\nAnswer: D\n\nConstruct a complete truth table for the following pairs of propositions. Then, using the truth tables, determine whether the statements are logically equivalent or contradictory. If neither, determine whether they are consistent or inconsistent. Justify your answers.\nE ⊃ (F · E) and ~E · F\nA. Logically equivalent\nB. Contradictory\nC. Neither logically equivalent nor contradictory, but consistent\nD. Inconsistent\nAnswer: C\n\nWhich of the given formulas of PL is the best symbolization of the following sentence?\nTurtles live long lives and are happy creatures, unless they are injured.\nA. (L • H) ≡ I\nB. (L • H) ∨ I\nC. L • (H ∨ I)\nD. L • (H ⊃ R)\nAnswer: B\n\nIdentify the antecedent of the following conditional proposition: If the Bees win their first game, then neither the Aardvarks nor the Chipmunks win their first games.\nA. The Aardvarks do not win their first game.\nB. The Bees win their first game.\nC. The Chipmunks do not win their first game.\nD. Neither the Aardvarks nor the Chipmunks win their first games.\nAnswer:"
>>>
>>> print(task.create_qa_prompt_choices_fewshot(task.kshot_docs()[:5], task.valid_docs()[0]))
The following are multiple choice questions (with answers) about formal logic.

Select the best translation into predicate logic: No people drive on Mars.
A. ~Pd
B. (∀x)(Px ∨ ~Dx)
C. (∀x)(Px ⊃ ~Dx)
D. ~Dp
Answer: C

Select the best translation into predicate logic.George borrows Hector's lawnmower. (g: George; h: Hector; l: Hector's lawnmower; Bxyx: x borrows y from z)
A. Blgh
B. Bhlg
C. Bglh
D. Bghl
Answer: C

Select the best English interpretation of the given arguments in predicate logic.
Dm
(∀x)(Wx ⊃ ~Dx)
(∀x)Wx ∨ Ag     / (∃x)Ax
A. Marina is a dancer. Some weaklings are not dancers. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.
B. Marina is a dancer. No weakling is a dancer. Everything is either a weakling or plays volleyball. So something plays volleyball.
C. Marina is a dancer. Some weaklings are not dancers. Everything is either a weakling or plays volleyball. So something plays volleyball.
D. Marina is a dancer. No weakling is a dancer. Either everything is a weakling or Georgia plays volleyball. So something plays volleyball.
Answer: D

Construct a complete truth table for the following pairs of propositions. Then, using the truth tables, determine whether the statements are logically equivalent or contradictory. If neither, determine whether they are consistent or inconsistent. Justify your answers.
E ⊃ (F · E) and ~E · F
A. Logically equivalent
B. Contradictory
C. Neither logically equivalent nor contradictory, but consistent
D. Inconsistent
Answer: C

Which of the given formulas of PL is the best symbolization of the following sentence?
Turtles live long lives and are happy creatures, unless they are injured.
A. (L • H) ≡ I
B. (L • H) ∨ I
C. L • (H ∨ I)
D. L • (H ⊃ R)
Answer: B

Identify the antecedent of the following conditional proposition: If the Bees win their first game, then neither the Aardvarks nor the Chipmunks win their first games.
A. The Aardvarks do not win their first game.
B. The Bees win their first game.
C. The Chipmunks do not win their first game.
D. Neither the Aardvarks nor the Chipmunks win their first games.
Answer:
>>>
```
</details>

### Accuracy

| Model                  | Task             | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                     | :-               | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-13b` | `humanities`     | 43.82  | 43.24  | 41.12  | 39.77  | 38.61  |
|                        | `social_science` | 54.01  | 53.41  | 54.30  | 54.01  | 47.18  |
|                        | `stem`           | 38.63  | 38.63  | 37.38  | 39.25  | 35.51  |
|                        | `other`          | 52.39  | 53.24  | 54.08  | 50.99  | 49.58  |
|                        | (AVG)            | 47.21  | 47.13  | 46.72  | 46.01  | 42.72  |
| `huggyllama/llama-30b` | `humanities`     | 53.47  | 54.05  | 53.47  | 52.32  | 52.12  |
|                        | `social_science` | 67.06  | 66.77  | 67.06  | 67.36  | 63.50  |
|                        | `stem`           | 38.32  | 37.69  | 37.69  | 38.63  | 38.01  |
|                        | `other`          | 60.28  | 60.56  | 61.41  | 61.13  | 56.90  |
|                        | (AVG)            | 54.78  | 54.77  | 54.91  | 54.86  | 52.63  |

### Negative Log-Likelihood

| Model                  | Task             | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                     | :-               | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-13b` | `humanities`     | 1.222  | 1.222  | 1.233  | 1.240  | 1.294  |
|                        | `social_science` | 1.056  | 1.057  | 1.065  | 1.083  | 1.166  |
|                        | `stem`           | 1.273  | 1.272  | 1.284  | 1.282  | 1.333  |
|                        | `other`          | 1.044  | 1.045  | 1.049  | 1.061  | 1.108  |
|                        | (AVG)            | 1.149  | 1.149  | 1.158  | 1.167  | 1.225  |
| `huggyllama/llama-30b` | `humanities`     | 1.037  | 1.036  | 1.046  | 1.055  | 1.090  |
|                        | `social_science` | .7998  | .8020  | .8015  | .8177  | .8843  |
|                        | `stem`           | 1.237  | 1.236  | 1.240  | 1.255  | 1.277  |
|                        | `other`          | .8821  | .8809  | .8830  | .9214  | .9481  |
|                        | (AVG)            | .9890  | .9887  | .9926  | 1.012  | 1.050  |

### Expected Calibration Error

| Model                  | Task             | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                     | :-               | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-13b` | `humanities`     | 5.167  | 6.455  | 6.025  | 6.872  | 6.221  |
|                        | `social_science` | 5.441  | 4.853  | 4.160  | 7.089  | 5.164  |
|                        | `stem`           | 5.679  | 4.919  | 5.202  | 7.966  | 5.494  |
|                        | `other`          | 9.269  | 8.485  | 4.709  | 7.246  | 5.796  |
|                        | (AVG)            | 6.389  | 6.178  | 5.024  | 7.293  | 5.669  |
| `huggyllama/llama-30b` | `humanities`     | 3.703  | 3.476  | 4.699  | 5.996  | 3.953  |
|                        | `social_science` | 3.201  | 4.643  | 6.589  | 4.058  | 5.506  |
|                        | `stem`           | 9.963  | 10.53  | 10.80  | 9.745  | 8.483  |
|                        | `other`          | 5.076  | 5.132  | 5.501  | 5.483  | 5.866  |
|                        | (AVG)            | 5.486 | 5.945  | 6.897  | 6.321  | 5.952  |
