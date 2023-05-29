import argparse
import json
import random
import logging
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import force_console_behavior
master_bar,progress_bar = force_console_behavior()
from attrdict import AttrDict
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset

from CustomizedDataProcessor import CustomizedDataProcessor

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def simple_accuracy(labels, preds):
    return (labels == preds).mean()
        
def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

def convert_examples_to_features(processor, examples, tokenizer, max_length):
    label_list = processor.get_unique_labels()

    label_map = {label: i for i, label in enumerate(label_list)}

    # classification
    labels = [label_map[example.label] for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [example.text for example in examples], max_length=max_length, pad_to_max_length=True
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features

def load_examples(processor, tokenizer, max_seq_len, data_filepath):
 
    examples = processor.get_examples(data_filepath)
    features = convert_examples_to_features(processor, examples, tokenizer, max_length=max_seq_len)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    
    # classification
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  Total train batch size = {}".format(args.train_batch_size))
    print("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    print("  Total optimization steps = {}".format(t_total))
    print("  Logging steps = {}".format(args.logging_steps))
    print("  Save steps = {}".format(args.save_steps))

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", global_step)
                    else:
                        evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    print("Saving model checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        print("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch + 1))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, output_dir, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    labels = None

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    
    preds = np.argmax(preds, axis=1) # classification
    
    result = acc_score(labels, preds)
    results.update(result)

    for key in sorted(results.keys()):
        print("  {} = {}".format(key, str(results[key])))

    return results


def main(config_filepath):
    with open(config_filepath) as f:
        args = AttrDict(json.load(f))
    print("Training/evaluation parameters {}".format(args))

    set_seed(args)

    processor = CustomizedDataProcessor()
    unique_labels = processor.get_unique_labels()

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels = len(unique_labels),
        id2label={str(i): label for i, label in enumerate(unique_labels)},
        label2id={label: i for i, label in enumerate(unique_labels)},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # Load dataset
    train_dataset = load_examples(processor, tokenizer, args.max_seq_len, args.train_filepath) if args.train_filepath else None
    dev_dataset = load_examples(processor, tokenizer, args.max_seq_len, args.dev_filepath) if args.dev_filepath else None
    test_dataset = load_examples(processor, tokenizer, args.max_seq_len, args.test_filepath) if args.test_filepath else None

    if dev_dataset == None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use testset

    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset, test_dataset)
        print(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        print("Evaluate the following checkpoints: {}".format(checkpoints))
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
                
            result = evaluate(args, model, test_dataset, output_dir, global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config_filepath", type=str, default="config.json")
    cli_args = cli_parser.parse_args()

    main(cli_args.config_filepath)
