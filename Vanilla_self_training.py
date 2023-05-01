#This script runs the vanilla self-training approaches: DBST, DBST+N

#Example command:
#VST: python Vanilla_self_training.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/Vanilla_ST/' --model_type 'BERT' --batch_size 32 --num_epochs 10 --device 'cuda' --random_sample_size 4500 --noisy 'no' --dataset 'MNLI'
#VST+N: python Vanilla_self_training.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/Vanilla_ST_noisy/' --model_type 'BERT' --batch_size 32 --num_epochs 10 --device 'cuda' --random_sample_size 4500 --noisy 'yes' --dataset 'MNLI'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pandas
from Train_and_test_helper import *
from Data_preparation_helper import create_data_for_pretrained_lms
import random
import numpy as np
import torch
from Utils import *
import argparse



parser = argparse.ArgumentParser(description='Vanilla self-training: VST and VST+N.')

parser.add_argument("--base", type=str, help="Location of a directory containing the train, test and dev files and a sub-directory named 'iteration_0' containing the synthetic set.")
parser.add_argument("--model_type", type=str, default='BERT', help="Type of the model you want to train and test: BERT, Sci_BERT, RoBERTa or XLNet")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the model for.")
parser.add_argument("--epoch_patience", type=int, default=2, help="Patience for early stopping at epoch level.")
parser.add_argument("--iteration_patience", type=int, default=10, help="Patience for early stopping at iteration level.")
parser.add_argument("--report_every", type=int, default=-1, help="Step interval to report loss. By default loss will be reported only at the end of an epoch.")
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--random_seed", type=int, default=1234)
parser.add_argument("--noisy", type=str, default='no', help="Whethter to inject noise by using data augmentation.")
parser.add_argument("--ensure_label_consistency", type=str, default='yes', help="Whethter to ensure label consistency as an additional filter.")
parser.add_argument("--random_sample_size", type=int, help="Random sample size for synthetic data.")
parser.add_argument("--dataset", type=str, help="Name of the dataset.")

args = parser.parse_args()


SEED = args.random_seed

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

base = args.base

augmentation = args.noisy
ensure_label_consistency = args.ensure_label_consistency
random_sample_size = args.random_sample_size
max_iteration = 100
best_dev_score = 0
best_iteration = 0
patience = args.iteration_patience
not_improving_iteration = 0
epoch_patience = args.epoch_patience
dataset = args.dataset
model_type = args.model_type
batch_size = args.batch_size
accumulation_steps = 64/batch_size
num_epochs = args.num_epochs
report_every = args.report_every
device = args.device


if dataset == 'RTE':
	confidence_threshold = 0.7
	num_classes = 2

	label_to_idx_dict = {
		'entailment' : 0,
		'not_entailment' : 1,
	}

	idx_to_label_dict = {
		0:'entailment',
		1:'not_entailment',
	}

else:
	confidence_threshold = 0.9
	num_classes = 3

	label_to_idx_dict = {
	    'entailment' : 0,
	    'neutral' : 1,
	    'contradiction' : 2,

	}

	idx_to_label_dict = {
		0:'entailment',
		1:'neutral',
		2:'contradiction',

	}
	


traininingSet = pandas.read_csv(base+'train.tsv', sep='\t')
testingSet = pandas.read_csv(base+'test.tsv', sep='\t')
validationSet = pandas.read_csv(base+'dev.tsv', sep='\t')

#print(traininingSet.columns)
traininingSet['label'] = traininingSet.apply(lambda x: get_numeric_label(x['label'], label_to_idx_dict), axis=1)
testingSet['label'] = testingSet.apply(lambda x: get_numeric_label(x['label'], label_to_idx_dict), axis=1)
validationSet['label'] = validationSet.apply(lambda x: get_numeric_label(x['label'], label_to_idx_dict), axis=1)


if model_type == 'Sci_BERT':
	tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_cased', do_lower_case=False)
elif model_type == 'Sci_BERT_basevocab':
	tokenizer = BertTokenizer.from_pretrained('/home/ubuntu/scibert_basevocab_cased', do_lower_case=False)
elif model_type == 'RoBERTa':
	tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=False)
elif model_type == 'xlnet':
	tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=False)
else:
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


create_data_for_pretrained_lms(base, label_to_idx_dict, traininingSet, tokenizer, 'train',model_type)
create_data_for_pretrained_lms(base, label_to_idx_dict, testingSet, tokenizer, 'test',model_type)
create_data_for_pretrained_lms(base, label_to_idx_dict, validationSet, tokenizer, 'valid',model_type)

X_test = load_data(base+'X_test.pkl')
X_valid = load_data(base+'X_valid.pkl')
att_mask_test = load_data(base+'att_mask_test.pkl')
att_mask_valid = load_data(base+'att_mask_valid.pkl')
y_test = load_data(base+'y_test.pkl')
y_valid = load_data(base+'y_valid.pkl')
y_test = np.asarray(y_test)
y_valid = np.asarray(y_valid)
X_test = np.asarray(X_test)
X_valid = np.asarray(X_valid)


for curr_iteration in range(0,max_iteration):

	print("Iteration:", curr_iteration)
	print("==================================================================")
	curr_NLI_training_base = base+'iteration_'+str(curr_iteration)+'/'


	if curr_iteration != 0:
		teacher_base = base+'iteration_'+str(curr_iteration-1)+'/'
		teacher_model_location = teacher_base+dataset+'_'+model_type+'/model/'
	else:
		teacher_base = base+'iteration_'+str(curr_iteration)+'/'
		teacher_model_location = teacher_base+dataset+'_'+model_type+'/model/'

	syntheticSet = pandas.read_csv(curr_NLI_training_base+'syntheticAndAugmented.tsv', sep='\t')

	pseudoLabeledSet = None

	if p.exists(curr_NLI_training_base+'pseudoLabeledSet.tsv') == True:
		pseudoLabeledSet = pandas.read_csv(curr_NLI_training_base+'pseudoLabeledSet.tsv', sep='\t')
		pseudoLabeledSet['label'] = pseudoLabeledSet.apply(lambda x: get_numeric_label(x['label'], label_to_idx_dict), axis=1)
		pseudoLabeledSet = pseudoLabeledSet[['sentence1','sentence2', 'label']]
		

	if augmentation == 'yes':
		syntheticSet = syntheticSet[['sentence1','sentence2','sentence1_augmented','sentence2_augmented', 'label']]
	else:
		syntheticSet = syntheticSet[['sentence1','sentence2', 'label']]

	syntheticSet['label'] = syntheticSet.apply(lambda x: get_numeric_label(x['label'], label_to_idx_dict), axis=1)

	sampled_syntheticSet, remaining_synthetic_set = sample_balanced_random_set(syntheticSet, random_sample_size)
	
	Prepared_data_output_location = curr_NLI_training_base + dataset+'_'+model_type+'/'

	
	create_data_for_pretrained_lms(Prepared_data_output_location, label_to_idx_dict, sampled_syntheticSet, tokenizer, 'synthetic',model_type)
	X_synthetic = load_data(Prepared_data_output_location+'X_synthetic.pkl')
	att_mask_synthetic = load_data(Prepared_data_output_location+'att_mask_synthetic.pkl')
	y_synthetic = load_data(Prepared_data_output_location+'y_synthetic.pkl')
	y_synthetic = np.asarray(y_synthetic)
	X_synthetic = np.asarray(X_synthetic)


	model_location = Prepared_data_output_location+'model/'
	if p.exists(model_location) == False:
		os.mkdir(model_location)

	if curr_iteration == 0:
		X_train = load_data(base+'X_train.pkl')
		att_mask_train = load_data(base+'att_mask_train.pkl')
		y_train = load_data(base+'y_train.pkl')
		y_train = np.asarray(y_train)
		X_train = np.asarray(X_train)
		train_model(teacher_model_location, model_type, X_train, att_mask_train, y_train, X_valid, att_mask_valid, y_valid, device, batch_size, accumulation_steps, num_epochs, num_classes, report_every, epoch_patience)

	
	predicted_labels, confidences, _ = test_model(teacher_model_location, model_type, X_synthetic, att_mask_synthetic, y_synthetic, device, batch_size, num_classes, curr_NLI_training_base+'syntetic_set_performance.csv')
	
	sampled_syntheticSet['predicted'] = predicted_labels
	sampled_syntheticSet['confidence'] = confidences

	remaining_synthetic_set = [remaining_synthetic_set]

	if ensure_label_consistency == 'yes':
		not_matched = sampled_syntheticSet.loc[~(sampled_syntheticSet['label'] == sampled_syntheticSet['predicted'])]
		sampled_syntheticSet = sampled_syntheticSet.loc[sampled_syntheticSet['label'] == sampled_syntheticSet['predicted']]
		remaining_synthetic_set.append(not_matched)


	sampled_syntheticSet_low_confidence = sampled_syntheticSet.loc[sampled_syntheticSet['confidence'] < confidence_threshold]
	remaining_synthetic_set.append(sampled_syntheticSet_low_confidence)
	sampled_syntheticSet = sampled_syntheticSet.loc[sampled_syntheticSet['confidence'] >= confidence_threshold]

	if max(sampled_syntheticSet['predicted'].value_counts().tolist()) != min(sampled_syntheticSet['predicted'].value_counts().tolist()):
		sampled_syntheticSet, unbal_remaining = balance_set(sampled_syntheticSet,'predicted')
		remaining_synthetic_set.append(unbal_remaining)
	
	sampled_syntheticSet = sampled_syntheticSet.drop('label', axis=1)
	sampled_syntheticSet = sampled_syntheticSet.rename(columns={'predicted': 'label'})

	if augmentation == 'yes':
		sampled_syntheticSet = sampled_syntheticSet.drop('sentence1', axis=1)
		sampled_syntheticSet = sampled_syntheticSet.drop('sentence2', axis=1)
		sampled_syntheticSet = sampled_syntheticSet.rename(columns={'sentence1_augmented': 'sentence1'})
		sampled_syntheticSet = sampled_syntheticSet.rename(columns={'sentence2_augmented': 'sentence2'})
	
	pseudoLabeledSet = pandas.concat([pseudoLabeledSet, sampled_syntheticSet])

	traininingSet_curr_iter = pandas.concat([traininingSet, pseudoLabeledSet])

	create_data_for_pretrained_lms(Prepared_data_output_location, label_to_idx_dict, traininingSet_curr_iter, tokenizer, 'train',model_type)

	X_train = load_data(Prepared_data_output_location+'X_train.pkl')
	att_mask_train = load_data(Prepared_data_output_location+'att_mask_train.pkl')
	y_train = load_data(Prepared_data_output_location+'y_train.pkl')

	
	y_train = np.asarray(y_train)
	X_train = np.asarray(X_train)
	

	train_model(model_location, model_type, X_train, att_mask_train, y_train, X_valid, att_mask_valid, y_valid, device,batch_size,accumulation_steps,num_epochs,num_classes,report_every, epoch_patience, load=False)


	print('test set performance')
	
	_, _, test_score = test_model(model_location, model_type, X_test, att_mask_test, y_test, device, batch_size,num_classes, curr_NLI_training_base+'test_set_performance.csv')
	
	#quit()
	print('dev set performance')
	_, _, dev_score = test_model(model_location, model_type, X_valid, att_mask_valid, y_valid, device, batch_size,num_classes, curr_NLI_training_base+'dev_set_performance.csv')
	

	if dev_score > best_dev_score:
		print("Dev score improved from "+str(best_dev_score)+" to "+str(dev_score))
		best_dev_score = dev_score
		best_iteration = curr_iteration
		not_improving_iteration = 0
		with open(base+'best_performing_model.txt','w') as file:
			file.write('best iteration: '+str(best_iteration)+' \n')
			file.write('best dev score: '+str(best_dev_score)+' \n')
			file.write('test score: '+str(test_score)+' \n')
	else:
		not_improving_iteration+=1
		if not_improving_iteration == patience:
			print("Performance is not improving for "+str(patience)+" iterations. Stopping experiment.")
			print("Final best iteration "+str(best_iteration))
			break


	remaining_synthetic_set = pandas.concat(remaining_synthetic_set)
	pseudoLabeledSet['label'] = pseudoLabeledSet.apply(lambda x: get_textual_label(x['label'], idx_to_label_dict), axis=1)
	remaining_synthetic_set['label'] = remaining_synthetic_set.apply(lambda x: get_textual_label(x['label'], idx_to_label_dict), axis=1)

	
	next_iteration = curr_iteration+1

	next_NLI_training_base = base+'iteration_'+str(next_iteration)+'/'

	if p.exists(next_NLI_training_base) == False:
		os.mkdir(next_NLI_training_base)

	pseudoLabeledSet.to_csv(next_NLI_training_base+'pseudoLabeledSet.tsv',sep='\t')
	remaining_synthetic_set.to_csv(next_NLI_training_base+'syntheticAndAugmented.tsv',sep='\t')





