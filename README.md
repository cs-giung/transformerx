# Transformerx

> [!IMPORTANT]
> This project is currently under development and is not yet stable. Keep in mind that any features of the project are subject to change.

_Transformerx_ is inspired by [Hugging Face Transformers](https://github.com/huggingface/transformers) and aims to implement state-of-the-art deep neural network architectures in [JAX](https://github.com/google/jax) with minimal dependencies.
It prioritizes simplicity and hackability by favoring code replication over complexity or increased abstraction.
Currently, _Transformerx_ is being developed in a Python 3.12 environment, and using alternative Python versions may lead to unexpected behaviors.

[`examples/notebooks/*.ipynb`](examples/notebooks/) provide a walk-through on how to use the supported models.

## Getting started

Basic dependencies for development in TPU environments:
```bash
pip install -U pip setuptools wheel google-cloud-tpu
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu tensorflow-datasets
pip install datasets einops einshard jax-smi pylint qax tabulate transformers
```
