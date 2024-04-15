"""
Pointer Sentinel Mixture Models
https://arxiv.org/pdf/1609.07843.pdf
"""
from .base import PerplexityTask


class WikiText2(PerplexityTask):
    DATASET_PATH = 'EleutherAI/wikitext_document_level'
    DATASET_NAME = 'wikitext-2-raw-v1'

    def train_docs(self):
        return list(map(self._process_doc, self.dataset['train']))

    def valid_docs(self):
        return list(map(self._process_doc, self.dataset['validation']))

    def _process_doc(self, doc):
        return doc['page']
