"""
CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge
https://arxiv.org/abs/1811.00937
"""
from transformerx.tasks.base import MultipleChoiceTask


class CommonsenseQA(MultipleChoiceTask):
    DATASET_PATH = 'commonsense_qa'
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

    def create_qa_prompt_choices_fewshot_for_train(self, example_docs, doc):
        return self.create_qa_prompt_choices_fewshot(example_docs, doc)

    def _process_doc(self, doc):
        return {
            'query': doc['question'],
            'choices': doc['choices']['text'],
            'gold': int(ord(doc['answerKey']) - 65)}
