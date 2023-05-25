from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, AutoModel
from transformers import pipeline
import numpy as np

def load_tokenizer_and_model(model_name_or_path, num_classes=None, mode='classification'):
    if num_classes is not None: # train
        config = AutoConfig.from_pretrained(model_name_or_path, num_classes=num_classes)
    else: # test
        config = AutoConfig.from_pretrained(model_name_or_path)      
    
    print('Loading tokenizer & model for {}..\n'.format(model_name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if mode == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    elif mode == 'masking':
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
        
    return tokenizer, model