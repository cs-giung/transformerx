"""
HellaSwag: Can a Machine Really Finish Your Sentence?
https://arxiv.org/abs/1905.07830
"""
import re
from transformerx.tasks.base import MultipleChoiceTask


class HellaSwag(MultipleChoiceTask):
    DATASET_PATH = 'hellaswag'
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

    def kshot_docs(self):
        raise NotImplementedError

    def create_qa_prompt_choices(self, doc):
        prompt = doc['query']
        for i, choice in enumerate(doc['choices']):
            prompt += '\n' + chr(65 + i) + '. ' + choice
        prompt += '\n'
        prompt += 'Answer:'
        return prompt

    def create_qa_prompt_choices_fewshot(self, example_docs, doc):
        prompt = (
            "The following are multiple choice questions (with answers) "
            "about common sense reasoning.\n\n")
        for example in example_docs:
            prompt += self.create_qa_prompt_choices(example)
            prompt += ' ' + chr(65 + example['gold']) + '\n\n'
        prompt += self.create_qa_prompt_choices(doc)
        return prompt

    def _process_doc(self, doc):
        def _preprocess(text):
            text = text.strip()
            text = text.replace(' [title]', '. ')
            text = re.sub('\\[.*?\\]', '', text)
            text = text.replace('  ', ' ')
            text = text.strip()
            return text
        return {
            'query': _preprocess(doc['ctx']),
            'choices': [_preprocess(e) for e in doc['endings']],
            'gold': int(doc['label'])}
