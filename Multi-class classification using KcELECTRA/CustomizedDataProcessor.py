import pandas as pd
import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

class CustomizedDataProcessor(object):
    
    relabel_dict = {'가격':0, '디자인':1, '사이즈': 2}
    
    def __init__(self):
        print('relabel_dict:', self.relabel_dict)
    
    def get_unique_labels(self):
        return list(self.relabel_dict.values())

    @classmethod
    def _read_file(cls, data_filepath):
        """Reads a csv file."""
        df = pd.read_csv(data_filepath)
        df['label'] = df['label'].apply(lambda x: cls.relabel_dict[x])
        return df

    def _create_examples(self, df):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, row in df.iterrows():
            text = row['text']
            label = row['label']
            examples.append(InputExample(text=text, label=label))
        print(examples[0])
        return examples

    def get_examples(self, data_filepath):
        """
        Args:
            mode: train, dev, test
        """
        return self._create_examples(self._read_file(data_filepath))