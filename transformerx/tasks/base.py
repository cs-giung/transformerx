import abc
import datasets


class Task(abc.ABC):
    DATASET_PATH: str = None
    DATASET_NAME: str = None

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None

    def download(self, data_dir, cache_dir, download_mode):
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH, name=self.DATASET_NAME, data_dir=data_dir,
            cache_dir=cache_dir, download_mode=download_mode,
            trust_remote_code=True)


class MultipleChoiceTask(Task):

    def doc_to_target(self, doc):
        return ' ' + doc['choices'][doc['gold']]
