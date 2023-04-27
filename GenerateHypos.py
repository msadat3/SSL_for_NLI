import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from Utils import *
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import os.path as p
import pandas
from transformers import BartForConditionalGeneration, BartTokenizer
from Utils import *
import argparse


def generate_hypos(src_texts, model, tokenizer, batch_size, decoding_strategy):
    # Prepare the text data into appropriate format for the model

    # Tokenize the texts
    start = 0
    end = min(start+batch_size,len(src_texts))
    translated_texts = []

    while True:
        src_texts_batch = src_texts[start:end]

        encoded = tokenizer.prepare_seq2seq_batch(src_texts_batch,
                                                  return_tensors='pt')
        #print(type(encoded))
        # Generate translation using model
        encoded = encoded.to(torch.device('cuda'))
        if decoding_strategy == 'top_k':
            translated = model.generate(**encoded, num_beams=1, max_length=80, min_length = 2, length_penalty = 2.0, no_repeat_ngram_size = 3, do_sample=True, top_k=topK, temperature=temperature)

        # Convert the generated tokens indices back into text
        translated_batch = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translated_texts+=translated_batch
        if end == len(src_texts):
            break
        start = end
        end = min(start+batch_size,len(src_texts))

    return translated_texts


parser.add_argument("--base", type=str, help="Location of the directory containing the prepared data for all three classes.")
parser.add_argument("--label", type=str, help="Class label to train the generative model for.")
parser.add_argument("--checkpoint_save_directory", type=str, help="Name of the directory you want to save the model for.")
parser.add_argument("--model_type", type=str, help="Type of the model you want to train and test: BART_large")
parser.add_argument("--input_file_name", type=str, help="Name of the input file. The input file should be in the base directory.")
parser.add_argument("--device", type=str, default='cpu')


args = parser.parse_args()


model_type = args.model_type
label = args.label
checkpoint_directory_name = args.checkpoint_save_directory

base = args.base + args.label +'/'
model_base = base + model_type +'/' +checkpoint_directory_name+'/'

input_file_loc = args.base+'unlabeled_premises.txt'

model_loc = model_base +'/last_checkpoint.pt'

beam_size = 1
topK=10
temperature = 2.0
decoding_strategy = 'top_k'

output_file_location = base + '/synthetic.tsv'

if model_type == "BART_large":
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

model.load_state_dict(torch.load(model_loc))
model.to(torch.device(device))
model.eval()

i = 0

with open(input_file_loc,'r') as file:
    input_lines = file.readlines()

input_lines = [line.strip() for line in input_lines]
output_lines = []



output_lines = generate_hypos(input_lines, model, tokenizer, 64, decoding_strategy)
output_df = pandas.DataFrame({'sentence1':input_lines,'sentence2':output_lines, 'label': [label]*len(input_lines)})

output_df.to_csv(output_file_location, sep='\t')

