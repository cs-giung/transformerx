# transformerx

## Getting started

This project requires Python 3.12.

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
