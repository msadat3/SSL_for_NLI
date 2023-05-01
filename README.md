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

#### Step 6:
Finally, use the script named "Combine_class_wise_synthetic_data.py" to combine the synthetic data generated for each class to combine into one file. 

Example command:
```
python Combine_class_wise_synthetic_data.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/'
```




