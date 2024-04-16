"""
PIQA: Reasoning about Physical Commonsense in Natural Language
https://arxiv.org/abs/1911.11641
"""
from transformerx.tasks.base import MultipleChoiceTask


class PIQA(MultipleChoiceTask):
    DATASET_PATH = 'piqa'
    DATASET_NAME = None

    def train_docs(self):
        if self._train_docs is None:
            self._train_docs = list(
                map(self._process_doc, self.dataset['train']))
        return self._train_docs

    def valid_docs(self):
        if self._valid_docs is None:
            self._valid_docs = list(
                map(self._process_doc, self.dataset['validation']))
        return self._valid_docs

    def _process_doc(self, doc):
        out = {
            'query': 'Question: ' + doc['goal'] + '\nAnswer:',
            'choices': [doc['sol1'], doc['sol2']],
            'gold': doc['label']}
        return out
