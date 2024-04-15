"""
HellaSwag: Can a Machine Really Finish Your Sentence?
https://arxiv.org/pdf/1905.07830.pdf
"""
import re
from .base import MultipleChoiceTask


class HellaSwag(MultipleChoiceTask):
    DATASET_PATH = 'hellaswag'
    DATASET_NAME = None

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(
                map(self._process_doc, self.dataset['train']))
        return self._training_docs

    def validation_docs(self):
        return list(map(self._process_doc, self.dataset['validation']))

    def _process_doc(self, doc):
        ctx = doc['ctx_a'] + ' ' + doc['ctx_b'].capitalize()
        out = {
            'query': self.preprocess(doc['activity_label'] + ': ' + ctx),
            'choices': [self.preprocess(ending) for ending in doc['endings']],
            'gold': int(doc['label'])}
        return out

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        text = text.replace(' [title]', '. ')
        text = re.sub('\\[.*?\\]', '', text)
        text = text.replace('  ', ' ')
        return text
