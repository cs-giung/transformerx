"""
Measuring Massive Multitask Language Understanding
https://arxiv.org/abs/2009.03300
"""
from transformerx.tasks.base import MultipleChoiceTask


CATEGORIES = {
    "humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "social_sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "stem": [
        "abstract_algebra",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "other": [
        "anatomy",
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ],
}


class HendrycksTest(MultipleChoiceTask):
    DATASET_PATH = 'hails/mmlu_no_train'
    DATASET_NAME = None

    def __init__(self, subject: str):
        self.DATASET_NAME = subject
        super().__init__()

    def train_docs(self):
        raise NotImplementedError

    def valid_docs(self):
        if self._valid_docs is None:
            self._valid_docs = list(
                map(self._process_doc, self.dataset['validation']))
        return self._valid_docs

    def kshot_docs(self):
        if self._kshot_docs is None:
            self._kshot_docs = list(
                map(self._process_doc, self.dataset['dev']))
        return self._kshot_docs

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
            "about {}.\n\n".format(' '.join(self.DATASET_NAME.split('_'))))
        for example in example_docs:
            prompt += self.create_qa_prompt_choices(example)
            prompt += ' ' + chr(65 + example['gold']) + '\n\n'
        prompt += self.create_qa_prompt_choices(doc)
        return prompt

    def create_qa_prompt_choices_fewshot_for_train(self, example_docs, doc):
        prompt = \
            "The following are multiple choice questions (with answers).\n\n"
        for example in example_docs:
            prompt += self.create_qa_prompt_choices(example)
            prompt += ' ' + chr(65 + example['gold']) + '\n\n'
        prompt += self.create_qa_prompt_choices(doc)
        return prompt

    def _process_doc(self, doc):
        return {
            'query': doc['question'].strip(),
            'choices': doc['choices'],
            'gold': doc['answer']}
