# Transformerx

This project takes inspiration from [🤗 Transformers](https://github.com/huggingface/transformers) and aims to implement modern deep neural network architectures in [JAX](https://github.com/google/jax) with minimal dependencies.

## Getting Started

This project is being developed in a Python 3.12 environment, and using other Python versions could result in unexpected behaviors.

### Create virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -U pip setuptools wheel google-cloud-tpu
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu tensorflow-datasets
pip install datasets einops einshard jax-smi pylint qax tabulate transformers
```
