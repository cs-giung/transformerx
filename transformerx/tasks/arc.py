"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf
"""
from transformerx.tasks.base import MultipleChoiceTask


class ARCEasy(MultipleChoiceTask):
    DATASET_PATH = 'ai2_arc'
    DATASET_NAME = 'ARC-Easy'

    def train_docs(self):
        return list(map(self._process_doc, self.dataset['train']))

    def valid_docs(self):
        return list(map(self._process_doc, self.dataset['validation']))

    def _process_doc(self, doc):
        num2abc = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'}
        doc['answerKey'] = num2abc.get(doc['answerKey'], doc['answerKey'])
        out = {
            'query': 'Question: ' + doc['question'] + '\nAnswer:',
            'choices': doc['choices']['text'],
            'gold': ['A', 'B', 'C', 'D', 'E'].index(doc['answerKey'])}
        return out


class ARCChallenge(ARCEasy):
    DATASET_PATH = 'ai2_arc'
    DATASET_NAME = 'ARC-Challenge'
