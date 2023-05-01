# Learning to Infer from Unlabeled Data: A Semi-supervised Learning Approach for Robust Natural Language Inference

This repository contains the code for EMNLP 2022 Findings paper "[Learning to Infer from Unlabeled Data: A Semi-supervised Learning Approach for Robust Natural Language Inference](https://aclanthology.org/2022.findings-emnlp.351/)."

If you use our code and/or proposed approaches in your research, please cite our paper:

```
@inproceedings{sadat-caragea-2022-learning,
    title = "Learning to Infer from Unlabeled Data: A Semi-supervised Learning Approach for Robust Natural Language Inference",
    author = "Sadat, Mobashir and Caragea, Cornelia",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.351",
    pages = "4763--4776",
}
```

## Running Experiments

As described in our paper, our proposed Semi-supervised Learning (SSL) framwork consists of two phases: hypothesis generation and self-training. The instructions for each of these phases are described below.

### Hypothesis Generation

#### Step 1:
Create a base directory containing the train, test, dev splits of your low resource NLI dataset in tsv format. In addition, ensure this directory contains a txt file containing the unlabeled premises in a line-by-line format. 

#### Step 2:
Use the script named "Divide_class_wise_data.py" to split the data into sub-directories based on the classes.

Example command:
```
python Divide_class_wise_data.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/'
```

#### Step 3:
Use the script named "Prepare_data_for_generative_model.py" to encode the premise-hypothesis pairs of each class for training the hypothesis generation models.

Example command:
```
python Prepare_data_for_generative_model.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/' --label 'entailment'
```

#### Step 4:
Use the script named "TrainGenerativeModel.py" to train the hypothesis generation models for each class.

Example command:
```
python TrainGenerativeModel.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/' --label 'entailment' --model_type 'BART_large' --checkpoint_save_directory 'checkpoints' --device 'cuda'
```

#### Step 5:
Use the script named "GenerateHypos.py" to generate the hypotheses using the generative models trained in the previous step for the respective classes.

Example command:
```
python GenerateHypos.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/' --label 'entailment' --checkpoint_save_directory 'checkpoints' --model_type 'BART_large' --input_file_name  'unlabeled_premises.txt' --device 'cuda'
```
This step will create a file named 'synthetic.tsv' in the sub-directories of each NLI class.

#### Step 6:
Use the script named "Combine_class_wise_synthetic_data.py" to combine the synthetic data generated for each class to combine into one file. 

Example command:
```
python Combine_class_wise_synthetic_data.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/'
```
This command will combine the class-wise 'synthetic.tsv' files and save the data in the base directory in a file with the same name i.e., 'synthetic.tsv'

#### Step 7:
Finally, use the script named "Back_translation.py" to augment the premise-hypothesis pairs using backtranslation.

Example command:
```
python Back_translation.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/' --input_file_name 'synthetic.tsv' --output_file_name 'syntheticAndAugmented.tsv' --combine_as_columns 'yes' --device 'cuda'
```
This command will create a file named 'syntheticAndAugmented.tsv' that contains both the original and augmented versions of the premises and hypotheses in separate columns.

### Self-training

#### Step 1:
For running any of the self-training approaches (VST, VST+N, DBST, DBST+N), create a base directory containing the train, test and dev files in TSV format. Next, create a sub-directory named 'iteration_0' containing the file named 'syntheticAndAugmented.tsv'.

#### Step 2:
Run commands as follows to experiment with different self-training approaches.

=> VST
```
python Vanilla_self_training.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/Vanilla_ST/' --model_type 'BERT' --batch_size 32 --num_epochs 10 --device 'cuda' --random_sample_size 4500 --noisy 'no' --dataset 'MNLI'
```

=> VST+N
```
python Vanilla_self_training.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/Vanilla_ST_noisy/' --model_type 'BERT' --batch_size 32 --num_epochs 10 --device 'cuda' --random_sample_size 4500 --noisy 'yes' --dataset 'MNLI'
```

=> DBST
```
python Debiased_self_training.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/Debiased_ST/' --model_type 'BERT' --batch_size 32 --num_epochs 10 --device 'cuda' --random_sample_size 4500 --noisy 'no' --dataset 'MNLI'
```

=> VST+N
```
python Debiased_self_training.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/Debiased_ST_noisy/' --model_type 'BERT' --batch_size 32 --num_epochs 10 --device 'cuda' --random_sample_size 4500 --noisy 'yes' --dataset 'MNLI'
```

#### Step 3:
When the self-training iterations stop, check the file named 'best_performing_model.txt' in the base directory for the best performing iteration information. 


## Contact
Feel free to reach out to us at msadat3@uic.edu, cornelia@uic.edu, sadat.mobashir@gmail.com with any questions.

