"""
PIQA: Reasoning about Physical Commonsense in Natural Language
https://arxiv.org/abs/1911.11641
"""
from transformerx.tasks.base import MultipleChoiceTask


class PIQA(MultipleChoiceTask):
    DATASET_PATH = 'piqa'
    DATASET_NAME = None

    def train_docs(self):
        return list(map(self._process_doc, self.dataset['train']))

    def valid_docs(self):
        return list(map(self._process_doc, self.dataset['validation']))

    def _process_doc(self, doc):
        out = {
            'query': 'Question: ' + doc['goal'] + '\nAnswer:',
            'choices': [doc['sol1'], doc['sol2']],
            'gold': doc['label']}
        return out
