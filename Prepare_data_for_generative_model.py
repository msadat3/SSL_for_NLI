# This script encodes and prepares the data to train generative model to generate hypothesis given a premise for a particular class.
# Example command: python Prepare_data_for_generative_model.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/' --label 'entailment'

from Utils import *
import os.path as p
import os
import pandas
import numpy as np
from transformers import BartTokenizer
import argparse


parser = argparse.ArgumentParser(description='Data preparation for generative models.')
parser.add_argument("--base", type=str, help="Location of a directory containing the class-wise split data.")
parser.add_argument("--label", type=str, help="The class of the examples on which generative models will be trained.")
args = parser.parse_args()

label = args.label

base = args.base + label + '/'

traininingSet_location = base+'train.tsv'
test_set_location = base+'test.tsv'
validation_set_location = base + 'dev.tsv'

traininingSet = pandas.read_csv(traininingSet_location, sep='\t')
testingSet = pandas.read_csv(test_set_location, sep='\t')
validationSet = pandas.read_csv(validation_set_location, sep='\t')

prertained_model_name = 'facebook/bart-large'
tokenizer = BartTokenizer.from_pretrained(prertained_model_name, do_lower_case=False)

def Tokenize_Input(text, max_length = 1024):
    text = str(text)
    text_encoded = tokenizer.encode(text, truncation=True, padding=True, add_special_tokens=True, max_length=max_length)
    return text_encoded

def get_attention_masks(X):
    attention_masks = []
    for sent in X:
        att_mask = [int(token_id != tokenizer.pad_token_id) for token_id in sent]
        att_mask = np.asarray(att_mask)
        attention_masks.append(att_mask)
    attention_masks = np.asarray(attention_masks)
    return attention_masks

def pad_seq(seq, max_len, pad_idx):
    if len(seq)>max_len:
        end_idx = seq[-1]
        seq = seq[0:max_len-1]
        seq.append(end_idx)
    while len(seq) != max_len:
        seq.append(pad_idx)
    return seq

def prepare_all_data(output_location, premise_max_length, hypothesis_max_length):
    if p.exists(output_location) == False:
        os.mkdir(output_location)

    X_train_premise = traininingSet.apply(lambda x: Tokenize_Input(x['sentence1'], max_length = premise_max_length), axis=1)
    X_train_hypothesis = traininingSet.apply(lambda x: Tokenize_Input(x['sentence2'], max_length = hypothesis_max_length), axis=1)

    X_test_premise = testingSet.apply(lambda x: Tokenize_Input(x['sentence1'], max_length = premise_max_length), axis=1)
    X_test_hypothesis = testingSet.apply(lambda x: Tokenize_Input(x['sentence2'], max_length = hypothesis_max_length), axis=1)

    X_valid_premise = validationSet.apply(lambda x: Tokenize_Input(x['sentence1'], max_length = premise_max_length), axis=1)
    X_valid_hypothesis = validationSet.apply(lambda x: Tokenize_Input(x['sentence2'], max_length = hypothesis_max_length), axis=1)

    X_train_premise = pandas.Series(X_train_premise)
    X_train_hypothesis = pandas.Series(X_train_hypothesis)

    X_test_premise = pandas.Series(X_test_premise)
    X_test_hypothesis = pandas.Series(X_test_hypothesis)

    X_valid_premise = pandas.Series(X_valid_premise)
    X_valid_hypothesis = pandas.Series(X_valid_hypothesis)

    max_len_premise = 0
    max_len_hypothesis = 0

    for x in X_train_premise:
        if len(x) > max_len_premise:
            max_len_premise = len(x)
    for x in X_train_hypothesis:
        if len(x) > max_len_hypothesis:
            max_len_hypothesis = len(x)

    X_train_premise = X_train_premise.apply(pad_seq, max_len=max_len_premise, pad_idx=tokenizer.pad_token_id)
    X_train_hypothesis = X_train_hypothesis.apply(pad_seq, max_len=max_len_hypothesis, pad_idx=tokenizer.pad_token_id)

    X_test_premise= X_test_premise.apply(pad_seq, max_len=max_len_premise, pad_idx=tokenizer.pad_token_id)
    X_test_hypothesis = X_test_hypothesis.apply(pad_seq, max_len=max_len_hypothesis, pad_idx=tokenizer.pad_token_id)

    X_valid_premise = X_valid_premise.apply(pad_seq, max_len=max_len_premise, pad_idx=tokenizer.pad_token_id)
    X_valid_hypothesis = X_valid_hypothesis.apply(pad_seq, max_len=max_len_hypothesis, pad_idx=tokenizer.pad_token_id)

    X_train_premise = np.array(X_train_premise.values.tolist())
    X_train_hypothesis = np.array(X_train_hypothesis.values.tolist())

    X_test_premise = np.array(X_test_premise.values.tolist())
    X_test_hypothesis = np.array(X_test_hypothesis.values.tolist())

    X_valid_premise = np.array(X_valid_premise.values.tolist())
    X_valid_hypothesis = np.array(X_valid_hypothesis.values.tolist())

    att_mask_train_premise= get_attention_masks(X_train_premise)
    att_mask_train_hypothesis = get_attention_masks(X_train_hypothesis)

    att_mask_test_premise = get_attention_masks(X_test_premise)
    att_mask_test_hypothesis = get_attention_masks(X_test_hypothesis)

    att_mask_valid_premise = get_attention_masks(X_valid_premise)
    att_mask_valid_hypothesis = get_attention_masks(X_valid_hypothesis)

    save_data(X_train_premise, output_location + 'X_train_premise.pkl')
    save_data(X_train_hypothesis, output_location + 'X_train_hypothesis.pkl')

    save_data(X_test_premise, output_location + 'X_test_premise.pkl')
    save_data(X_test_hypothesis, output_location + 'X_test_hypothesis.pkl')

    save_data(X_valid_premise, output_location + 'X_valid_premise.pkl')
    save_data(X_valid_hypothesis, output_location + 'X_valid_hypothesis.pkl')



    save_data(att_mask_train_premise, output_location + 'att_mask_train_premise.pkl')
    save_data(att_mask_train_hypothesis, output_location + 'att_mask_train_hypothesis.pkl')

    save_data(att_mask_test_premise, output_location + 'att_mask_test_premise.pkl')
    save_data(att_mask_test_hypothesis, output_location + 'att_mask_test_hypothesis.pkl')

    save_data(att_mask_valid_premise, output_location + 'att_mask_valid_premise.pkl')
    save_data(att_mask_valid_hypothesis, output_location + 'att_mask_valid_hypothesis.pkl')

    print(X_train_premise.shape, att_mask_train_premise.shape)
    print(X_train_hypothesis.shape, att_mask_train_hypothesis.shape)

    print(X_test_premise.shape, att_mask_test_premise.shape)
    print(X_test_hypothesis.shape, att_mask_test_hypothesis.shape)

    print(X_valid_premise.shape, att_mask_valid_premise.shape)
    print(X_valid_hypothesis.shape, att_mask_valid_hypothesis.shape)


output_location = base + 'BART_large/'

prepare_all_data(output_location, 1024,1024)