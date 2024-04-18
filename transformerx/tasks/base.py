"""Base classes for implementing tasks."""
import abc
import datasets
import numpy as np


class Task(abc.ABC):
    DATASET_PATH: str = None
    DATASET_NAME: str = None

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        self.download(data_dir, cache_dir, download_mode)
        self._train_docs = None
        self._valid_docs = None
        self._kshot_docs = None

    def download(self, data_dir, cache_dir, download_mode):
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH, name=self.DATASET_NAME, data_dir=data_dir,
            cache_dir=cache_dir, download_mode=download_mode,
            trust_remote_code=True)


class MultipleChoiceTask(Task):

    @classmethod
    def evaluate(cls, docs, log_probs, n_bins=15):
        """
        Args:
            docs:
            log_probs:
            n_bins: the number of bins for computing ECE (default: 15).

        Returns:
            a dictionary of evaluation results.
        """
        assert len(docs) == len(log_probs)

        metrics = {
            'acc': 0.0,
            'nll': 0.0,
            'ece': 0.0,
            'ece_bins': [[] for _ in range(n_bins)]}

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        for doc, log_prob in zip(docs, log_probs):

            metrics['acc'] += np.equal(
                np.argmax(log_prob), doc['gold']) / len(docs)
            metrics['nll'] += np.negative(
                log_prob[doc['gold']]) / len(docs)

            max_prob = np.max(np.exp(log_prob))
            for i in range(n_bins):
                if bin_boundaries[i] < max_prob <= bin_boundaries[i+1]:
                    break
            metrics['ece_bins'][i].append(
                [np.equal(np.argmax(log_prob), doc['gold']), max_prob])

        for i in range(n_bins):
            if len(metrics['ece_bins'][i]) == 0:
                continue
            b = np.array(metrics['ece_bins'][i]).mean(0)
            metrics['ece'] += np.abs(b[1] - b[0]) \
                * len(metrics['ece_bins'][i]) / len(docs)

        return metrics
