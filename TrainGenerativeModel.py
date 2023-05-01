#This script trains the hypothesis generation models for different classes.
#Example command: python TrainGenerativeModel.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/' --label 'entailment' --model_type 'BART_large' --checkpoint_save_directory 'checkpoints' --device 'cuda'

import os
from Utils import *
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import math
import os.path as p
from transformers import BartForConditionalGeneration
from Utils import *
import argparse



def create_data_loaders(X, X_att_mask, y, y_att_mask, batch_size, data_split='train'):
    X = torch.tensor(X, dtype=torch.long)
    X_att_mask = torch.tensor(X_att_mask, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    y_att_mask = torch.tensor(y_att_mask, dtype=torch.long)

    #print(X.shape, y.shape)
    data = TensorDataset(X, X_att_mask, y, y_att_mask)
    if data_split != 'train':
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return data_loader



def train_model(train_data_loader, validation_data_loader, model_type, learning_rate, num_epochs, device):
    last_checkpoint_info = {}

    last_checkpoint_info['epoch'] = 0
    last_checkpoint_info['i'] = 0
    last_checkpoint_info['current_best_perplexity'] = 999999999
    last_checkpoint_info['not_improving_checkpoints'] = 0

    last_checkpoint_info_location = checkpoint_location + 'last_checkpoint_info.pkl'
    last_checkpoint_optimizer_location = checkpoint_location + 'last_checkpoint_optimizer.pt'
    best_checkpoint_info_location = checkpoint_location + 'best_checkpoint_info.pkl'
    last_checkpoint_location = checkpoint_location + 'last_checkpoint.pt'
    best_checkpoint_location = checkpoint_location + 'best_checkpoint.pt'
    epoch_start = 0
    
    if model_type == 'BART_large':
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    model.to(torch.device(device))

    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.98))

    if p.exists(last_checkpoint_info_location) == True:
        model.load_state_dict(torch.load(last_checkpoint_location))
        last_checkpoint_info = load_data(last_checkpoint_info_location)
        optimizer.load_state_dict(torch.load(last_checkpoint_optimizer_location))
        print("Loading previously saved checkpoint")
        print(last_checkpoint_info)
        epoch_start = last_checkpoint_info['epoch']
    

    prev_validation_perplexity = last_checkpoint_info['current_best_perplexity']
    not_improving_checkpoints = last_checkpoint_info['not_improving_checkpoints']

    for epoch in range(epoch_start, num_epochs):
        model.train()
        i = 0
        optimizer.zero_grad()

        for X, X_att_mask, y, y_att_mask in train_data_loader:

            input_ids = X.to(torch.device(device))
            attention_mask = X_att_mask.to(torch.device(device))
            y = y.to(torch.device(device))
            y_att_mask = y_att_mask.to(torch.device(device))
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=y[:, :-1].contiguous(),
                            decoder_attention_mask=y_att_mask[:, :-1].contiguous(), labels=y[:, 1:].contiguous(), return_dict=False)

            crossEntropyLoss = outputs[0]
            loss = crossEntropyLoss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_data_loader):
                optimizer.step()
                optimizer.zero_grad()
                if (i + 1) == len(train_data_loader):
                    model.eval()
                    with torch.no_grad():
                        validation_loss = 0
                        batch_count = 0

                        for val_X, val_X_att_mask, val_y, val_y_att_mask in validation_data_loader:
                            batch_count += 1
                            input_ids = val_X.to(torch.device(device))
                            attention_mask = val_X_att_mask.to(torch.device(device))
                            val_y = val_y.to(torch.device(device))
                            val_y_att_mask = val_y_att_mask.to(torch.device(device))

                            val_outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids = val_y[:, :-1].contiguous(),
                                                decoder_attention_mask=val_y_att_mask[:, :-1].contiguous(),
                                                labels=val_y[:, 1:].contiguous(), return_dict=False)

                            val_loss_batch = val_outputs[0]
                            validation_loss += val_loss_batch.item()
                        validation_perplexity = math.exp(validation_loss / batch_count)

                        print('perplexity', validation_perplexity)
                        last_checkpoint_info['epoch'] = epoch
                        last_checkpoint_info['i'] = i

                        if (i + 1) == len(train_data_loader):
                            last_checkpoint_info['epoch'] = epoch+1
                            last_checkpoint_info['i'] = 0

                        if validation_perplexity < prev_validation_perplexity:
                            print("Epoch", epoch," Validation perplexity improved from ", prev_validation_perplexity, " to ",
                                  validation_perplexity)
                            prev_validation_perplexity = validation_perplexity
                            not_improving_checkpoints = 0
                            last_checkpoint_info['not_improving_checkpoints'] = not_improving_checkpoints
                            last_checkpoint_info['current_best_perplexity'] = validation_perplexity
                            torch.save(model.state_dict(), last_checkpoint_location)
                            torch.save(optimizer.state_dict(), last_checkpoint_optimizer_location)
                            torch.save(model.state_dict(), best_checkpoint_location)
                            save_data(last_checkpoint_info, last_checkpoint_info_location)
                            best_checkpoint_info = last_checkpoint_info
                            save_data(best_checkpoint_info, best_checkpoint_info_location)
                        else:
                            print("Epoch", epoch," Validation perplexity did not improve.")
                            not_improving_checkpoints += 1
                            last_checkpoint_info['not_improving_checkpoints'] = not_improving_checkpoints
                            torch.save(model.state_dict(), last_checkpoint_location)
                            torch.save(optimizer.state_dict(), last_checkpoint_optimizer_location)
                            print(last_checkpoint_info)
                            save_data(last_checkpoint_info, last_checkpoint_info_location)

                        
                        model.train()

            i += 1




parser = argparse.ArgumentParser(description='Train hypothesis generation models.')

parser.add_argument("--base", type=str, help="Location of the directory containing the data for all three classes.")
parser.add_argument("--label", type=str, help="Class label to train the generative model for.")
parser.add_argument("--checkpoint_save_directory", type=str, help="Name of the directory you want to save the model for.")
parser.add_argument("--model_type", type=str, default='BART_large', help="Type of the model you want to train and test: BART_large")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train the model for.")
parser.add_argument("--report_every", type=int, default=1, help="Step interval to report loss. By default loss will be reported only at the end of an epoch.")
parser.add_argument("--device", type=str, default='cpu')


args = parser.parse_args()

model_type = args.model_type
label = args.label


# base = '/home/msadat3/NLI/MNLI/Class_wise_BART_3K_unfiltered/'+label+'/' + model_type + "/" 
# #base = '/home/msadat3/NLI/MNLI/Class_wise_BART_6K/' + model_type + "/" 

base = args.base + label + '/' + model_type + '/'

checkpoint_location = base + args.checkpoint_save_directory+'/'

if p.exists(checkpoint_location) == False:
    os.mkdir(checkpoint_location)



batch_size = args.batch_size
accumulation_steps = args.gradient_accumulation_steps
learning_rate = args.learning_rate
num_epochs = args.num_epochs
report_every = args.report_every
device = args.device



print('Creating trainloader')
train_premise = load_data(base + 'X_train_premise.pkl')
train_premise_att = load_data(base + 'att_mask_train_premise.pkl')
train_hypothesis = load_data(base + 'X_train_hypothesis.pkl')
train_hypothesis_att = load_data(base + 'att_mask_train_hypothesis.pkl')
trainDataloader = create_data_loaders(train_premise, train_premise_att, train_hypothesis, train_hypothesis_att, batch_size, data_split='train')
torch.save(trainDataloader, base+"/trainDataloader")
                                    

print('Creating validationloader')
valid_premise = load_data(base + 'X_valid_premise.pkl')
valid_premise_att = load_data(base + 'att_mask_valid_premise.pkl')
valid_hypothesis = load_data(base + 'X_valid_hypothesis.pkl')
valid_hypothesis_att = load_data(base + 'att_mask_valid_hypothesis.pkl')
validDataloader = create_data_loaders(valid_premise, valid_premise_att, valid_hypothesis, valid_hypothesis_att, batch_size, data_split='valid')
torch.save(validDataloader, base+"/validDataloader")
    

print('Training model')

train_model(trainDataloader, validDataloader, model_type, learning_rate, num_epochs, device)

