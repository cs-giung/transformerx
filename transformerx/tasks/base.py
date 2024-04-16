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

    def download(self, data_dir, cache_dir, download_mode):
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH, name=self.DATASET_NAME, data_dir=data_dir,
            cache_dir=cache_dir, download_mode=download_mode,
            trust_remote_code=True)


class MultipleChoiceTask(Task):

    def doc_to_target(self, doc):
        return ' ' + doc['choices'][doc['gold']]

    @classmethod
    def evaluate(cls, docs, scores, n_bins=15):
        """
        Args:
            docs:
            scores:
            n_bins: the number of bins for computing ECE (default: 15).

        Returns:
            a dictionary of evaluation results.
        """
        assert len(docs) == len(scores)

        metrics = {
            'acc': 0.0, 'acc_tnorm': 0.0, 'acc_bnorm': 0.0,
            'nll': 0.0, 'nll_tnorm': 0.0, 'nll_bnorm': 0.0,
            'ece': 0.0, 'ece_tnorm': 0.0, 'ece_bnorm': 0.0,
            'ece_bins': [[] for _ in range(n_bins)],
            'ece_tnorm_bins': [[] for _ in range(n_bins)],
            'ece_bnorm_bins': [[] for _ in range(n_bins)]}

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        for doc, score in zip(docs, scores):

            # compute unnormalized metrics
            log_prob = np.array([e.sum() for e in score])
            log_prob = np.exp(log_prob - log_prob.max())
            log_prob = np.log(log_prob / log_prob.sum())

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

            # compute token-length normalized metrics
            log_prob = np.array([e.mean() for e in score])
            log_prob = np.exp(log_prob - log_prob.max())
            log_prob = np.log(log_prob / log_prob.sum())

            metrics['acc_tnorm'] += np.equal(
                np.argmax(log_prob), doc['gold']) / len(docs)
            metrics['nll_tnorm'] += np.negative(
                log_prob[doc['gold']]) / len(docs)

            max_prob = np.max(np.exp(log_prob))
            for i in range(n_bins):
                if bin_boundaries[i] < max_prob <= bin_boundaries[i+1]:
                    break
            metrics['ece_tnorm_bins'][i].append(
                [np.equal(np.argmax(log_prob), doc['gold']), max_prob])

            # compute byte-length normalized metrics
            log_prob = np.array([e.sum() for e in score])
            log_prob = log_prob / np.array([len(e) for e in doc['choices']])
            log_prob = np.exp(log_prob - log_prob.max())
            log_prob = np.log(log_prob / log_prob.sum())

            metrics['acc_bnorm'] += np.equal(
                np.argmax(log_prob), doc['gold']) / len(docs)
            metrics['nll_bnorm'] += np.negative(
                log_prob[doc['gold']]) / len(docs)

            max_prob = np.max(np.exp(log_prob))
            for i in range(n_bins):
                if bin_boundaries[i] < max_prob <= bin_boundaries[i+1]:
                    break
            metrics['ece_bnorm_bins'][i].append(
                [np.equal(np.argmax(log_prob), doc['gold']), max_prob])

        for i in range(n_bins):
            if len(metrics['ece_bins'][i]) == 0:
                continue
            b = np.array(metrics['ece_bins'][i]).mean(0)
            metrics['ece'] += np.abs(b[1] - b[0]) \
                * len(metrics['ece_bins'][i]) / len(docs)

        for i in range(n_bins):
            if len(metrics['ece_tnorm_bins'][i]) == 0:
                continue
            b = np.array(metrics['ece_tnorm_bins'][i]).mean(0)
            metrics['ece_tnorm'] += np.abs(b[1] - b[0]) \
                * len(metrics['ece_tnorm_bins'][i]) / len(docs)

        for i in range(n_bins):
            if len(metrics['ece_bnorm_bins'][i]) == 0:
                continue
            b = np.array(metrics['ece_bnorm_bins'][i]).mean(0)
            metrics['ece_bnorm'] += np.abs(b[1] - b[0]) \
                * len(metrics['ece_bnorm_bins'][i]) / len(docs)

        return metrics
