#This script divides the data according to their classes for training class-wise hypothesis generation models.
#Example command: python Divide_class_wise_data.py --base '/home/msadat3/NLI/MNLI/MNLI_6K/'

import pandas
import os
import os.path as p
import argparse

parser = argparse.ArgumentParser(description='Divide class-wise data for training generative models.')
parser.add_argument("--base", type=str, help="Location of a directory containing the train, test and dev files in TSV format")
args = parser.parse_args()

def divide_class_wise(df, base_location, output_file_name):
	classes = list(set(df['label'].tolist()))

	df_grouped = df.groupby('label')

	for c in classes:
		df_class = df_grouped.get_group(c)

		if p.exists(base_location+c+'/') == False:
			os.mkdir(base_location+c+'/')
		df_class.to_csv(base_location+c+'/'+output_file_name,sep='\t')



base = args.base


train_df = pandas.read_csv(base+'train.tsv', sep='\t')
test_df = pandas.read_csv(base+'test.tsv', sep='\t')
dev_df = pandas.read_csv(base+'dev.tsv', sep='\t')

divide_class_wise(train_df, base, 'train.tsv')
divide_class_wise(test_df, base, 'test.tsv')
divide_class_wise(dev_df, base, 'dev.tsv')


