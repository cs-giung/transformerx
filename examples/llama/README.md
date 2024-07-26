# Llama Examples

> [!NOTE]
> Experiments are mainly carried out in TPU VM environments, with support for Cloud TPUs provided by Google's TPU Research Cloud (TRC). Although there may be minor differences in the results based on the specific execution environment, these variations are expected to be minimal.

## Perplexity

The `perplexity` example computes how well a language model predicts the next word in a text, with lower scores indicating better performance. Perplexity scores cannot be directly compared between models.

## Reasoning

The `reasoning` example assesses a language model's ability to use common sense or world knowledge to make inferences. Reasoning scores reflect the knowledge gained during pretraining by evaluating the models exclusively in zero-shot and few-shot scenarios.
