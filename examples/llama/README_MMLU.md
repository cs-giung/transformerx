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
|                        | `commonsense_qa` | 
|                        | `piqa`           | 64.80  | 65.23  | 64.58  | 65.02  | 61.43  |

### Negative Log-Likelihood

| Model                  | Task             | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                     | :-               | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-13b` | `arc_e`          | .9329  | .9319  | .9391  | .9798  | 1.127  |
|                        | `arc_c`          | 1.238  | 1.237  | 1.240  | 1.249  | 1.385  |
|                        | `commonsense_qa` | 
|                        | `piqa`           | .6223  | .6226  | .6215  | .6266  | .6434  |

### Expected Calibration Error

| Model                  | Task             | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                     | :-               | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-13b` | `arc_e`          | 21.88  | 21.76  | 21.49  | 21.00  | 16.83  |
|                        | `arc_c`          | 8.796  | 8.751  | 7.797  | 7.011  | 7.332  |
|                        | `commonsense_qa` | 
|                        | `piqa`           | 3.981  | 4.663  | 4.032  | 6.574  | 2.515  |

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
| `huggyllama/llama-13b` | `humanities`     | 43.82  |
|                        | `social_science` | 54.01  |
|                        | `stem`           | 38.63  |
|                        | `other`          | 52.39  |

### Negative Log-Likelihood

| Model                  | Task             | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                     | :-               | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-13b` | `humanities`     | 1.222  |
|                        | `social_science` | 1.056  |
|                        | `stem`           | 1.273  |
|                        | `other`          | 1.044  |

### Expected Calibration Error

| Model                  | Task             | `FP16` | `Q8_0` | `Q6_0` | `Q5_0` | `Q4_0` |
| :-                     | :-               | :-:    | :-:    | :-:    | :-:    | :-:    |
| `huggyllama/llama-13b` | `humanities`     | 5.167  |
|                        | `social_science` | 5.441  |
|                        | `stem`           | 5.679  |
|                        | `other`          | 9.269  |
