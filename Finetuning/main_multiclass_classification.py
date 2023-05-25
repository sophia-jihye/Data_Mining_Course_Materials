from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm 
tqdm.pandas()
import pandas as pd
import os, torch, copy, shutil, re, argparse
import numpy as np

from transformers_helper import load_tokenizer_and_model
from CustomDataset import CustomDataset, encode_for_inference
import finetuning_classification

root_dir = '/home/jihyeparkk/DATA/Data_Mining_Course_Materials' 
train_filepath = os.path.join(root_dir, '마스크_가격,디자인,사이즈_3000.csv')

model_name_or_dir = 'beomi/KcELECTRA-base-v2022'
model_name_alias_dict = {'beomi/KcELECTRA-base-v2022': 'KcELECTRA'}
model_save_dir = os.path.join(root_dir, model_name_alias_dict[model_name_or_dir])
if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    
def do_prepare_data(relabel_dict, filepath):
    df = pd.read_csv(filepath)[['text', 'label']]
    print('Loaded {}'.format(filepath))
    df['label'] = df['label'].apply(lambda x: relabel_dict[x])
    return df
    
def start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, save_dir):
    tokenizer, model = load_tokenizer_and_model(model_name_or_dir, num_classes=num_classes, mode='classification')
    
    print('Getting data..\n')
    train_dataset = CustomDataset(tokenizer, train_texts, train_labels)
    val_dataset = CustomDataset(tokenizer, val_texts, val_labels)
    
    finetuning_classification.train(model, train_dataset, val_dataset, save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == '__main__':        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relabel_dict = {'가격':0, '디자인':1, '사이즈':2}
    num_classes = len(relabel_dict)    
        
    source_df = do_prepare_data(relabel_dict, train_filepath)
    X = source_df['text'].values
    y = source_df['label'].values
    train_texts, val_texts, train_labels, val_labels = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=0)

    start_finetuning(model_name_or_dir, num_classes, train_texts, train_labels, val_texts, val_labels, model_save_dir)

    # To save memory, delete the finetuned model in `temp` directory once model training is finished
    try: shutil.rmtree(model_save_dir)
    except OSError as e: print("Error: %s - %s." % (e.filename, e.strerror))
