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
    """
    This implementation resolves specific concerns regarding prompting;
        (1) adding a space after "Answer:" rather than spaces before "A/B/C/D".
        (2) mitigating occurences of double spaces after "about".

    https://github.com/hendrycks/test/pull/13
    """
    DATASET_PATH = 'hails/mmlu_no_train'
    DATASET_NAME = None

    def __init__(self, subject: str):
        self.DATASET_NAME = subject
        super().__init__()

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

    def _process_doc(self, doc):
        keys = ['A', 'B', 'C', 'D']
        out = {
            'query': doc['question'] + '\n' + '\n'.join([
                f'{k}. {a}' for k, a in zip(keys, doc['choices'])
            ]) + '\nAnswer:\n',
            'choices': keys,
            'gold': doc['answer']}
        return out

    def create_fewshot_prompt(self, example_docs, doc):
        prompt = (
            "The following are multiple choice questions (with answers) "
            "about {}.\n\n".format(' '.join(self.DATASET_NAME.split('_'))))
        for example in example_docs:
            prompt += example['query']
            prompt += '{}\n\n'.format(example['choices'][example['gold']])
        prompt += doc['query']
        return prompt
