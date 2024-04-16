import abc
import datasets
import numpy as np
from tqdm import tqdm


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
    def evaluate(cls, log_probs, labels, n_bins=15):
        """
        Args:
            log_probs
            labels
            n_bins

        Returns:
        """
        acc = 0.0
        nll = 0.0
        bins = [[] for _ in range(n_bins)]
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        for log_prob, label in tqdm(
                zip(log_probs, labels), total=len(log_probs)):
            acc += np.equal(np.argmax(log_prob), label) / len(log_probs)
            nll += np.negative(log_prob[label]) / len(log_probs)
            max_prob = np.max(np.exp(log_prob))
            for i in range(n_bins):
                if bin_boundaries[i] < max_prob <= bin_boundaries[i+1]:
                    break
            bins[i].append([np.equal(np.argmax(log_prob), label), max_prob])

        ece = 0.0
        for i in range(n_bins):
            if len(bins[i]) == 0:
                continue
            b = np.array(bins[i]).mean(0)
            ece += np.abs(b[1] - b[0]) * len(bins[i]) / len(log_probs)

        return {'acc': acc, 'nll': nll, 'ece': ece}
