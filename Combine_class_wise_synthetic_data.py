import pandas
import os
import os.path as p
import argparse

parser = argparse.ArgumentParser(description='Combine class-wise synthetic data.')
parser.add_argument("--base", type=str, help="Location of a directory containing the train, test and dev files in TSV format")
args = parser.parse_args()

base = args.base

#getting the class name
dev_df = pandas.read_csv(base+'dev.tsv', sep='\t')
labels = set(dev_df['label'].tolist())

#combining
synthetic = []
for label in labels:
	df = pandas.read_csv(base+label+'/synthetic.tsv', sep='\t')
	synthetic.append(df)

synthetic = pandas.concat([synthetic])
synthetic.to_csv(base+'synthetic.tsv', sep='\t')



