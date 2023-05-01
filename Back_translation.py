#copied and edited from: https://amitness.com/back-translation/

#This script augments the original data by using backtranslation.
#Example commands:
#For doubling the data size: python Back_translation.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/' --input_file_name 'train.tsv' --output_file_name 'train_aug_BT.tsv' --combine_as_columns 'no' --device 'cuda'
#For adding noise by data augmentation: python Back_translation.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/' --input_file_name 'synthetic.tsv' --output_file_name 'syntheticAndAugmented.tsv' --combine_as_columns 'yes' --device 'cuda'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pandas
from transformers import MarianMTModel, MarianTokenizer
import numpy as np
import argparse

def translate(texts, model, tokenizer, language="fr", batch_size=32, beam_size=1):
	# Prepare the text data into appropriate format for the model
	template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
	src_texts = [template(text) for text in texts]

	# Tokenize the texts
	start = 0
	end = min(start+batch_size,len(src_texts))
	translated_texts = []

	while True:
		src_texts_batch = src_texts[start:end]

		encoded = tokenizer.prepare_seq2seq_batch(src_texts_batch,
												  return_tensors='pt')
		encoded = encoded.to(torch.device(args.device))
		translated = model.generate(**encoded,num_beams=beam_size)
		translated_texts += tokenizer.batch_decode(translated, skip_special_tokens=True)

		if end == len(src_texts):
			break
		start = end
		end = min(start+batch_size,len(src_texts))

	return translated_texts

def back_translate(texts, source_lang="en", target_lang="fr",forward_beam_size=1, backward_beam_size=1):

	fr_texts = translate(texts, target_model, target_tokenizer, 
						 language=target_lang, beam_size=forward_beam_size)

	back_translated_texts = translate(fr_texts, en_model, en_tokenizer, 
									  language=source_lang, beam_size=backward_beam_size)
	return back_translated_texts


def augment_data(df, combine_as_columns = False):

	df_premises = df['sentence1'].tolist()
	df_hypotheses = df['sentence2'].tolist()

	df_labels = df['label'].tolist()
	aug_premises = back_translate(df_premises, source_lang="en", target_lang="fr")
	aug_hypotheses = back_translate(df_hypotheses, source_lang="en", target_lang="fr")
	
	if combine_as_columns == 'no':
		augmented_set = pandas.DataFrame({'sentence1':aug_premises, 'sentence2':aug_hypotheses, 'label': df_labels})
		df = pandas.concat([df,augmented_set])
	else:
		df['sentence1_augmented'] = aug_premises
		df['sentence2_augmented'] = aug_hypotheses

	return df



parser = argparse.ArgumentParser(description='Back translate both premise and hypothesis.')
parser.add_argument("--base", type=str, help="Directory containing the original data (training/synthetic) in TSV format.")
parser.add_argument("--input_file_name", type=str, help="Name of the file containing the original data e.g., 'train.tsv'")
parser.add_argument("--output_file_name", type=str, help="Name of the file containing the augmented data e.g., 'train_augmented.tsv'")
parser.add_argument("--combine_as_columns", type=str, help="Whether to combine the augmented data in the same column in the original data or create new columns containing the augmented data.")
parser.add_argument("--device", type=str, default='cpu')

args = parser.parse_args()

target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
target_model = MarianMTModel.from_pretrained(target_model_name)
target_model = target_model.to(torch.device(args.device))

en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
en_model = MarianMTModel.from_pretrained(en_model_name)
en_model = en_model.to(torch.device(args.device))


base = args.base
originalSet = pandas.read_csv(base+args.input_file_name, sep='\t')
originalSet_augmented = augment_data(originalSet, combine_as_columns = args.combine_as_columns)
print(originalSet_augmented.shape)
originalSet_augmented.to_csv(base+args.output_file_name, sep='\t')

	
	
	

