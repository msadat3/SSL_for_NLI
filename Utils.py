import pickle
from ast import literal_eval
import pandas
import random

def load_data(location):
    with open(location, 'rb') as file:
        data = pickle.load(file)
        return data

def save_data(data, location):
    with open(location, 'wb') as file:
        pickle.dump(data,file)
def literal(df,col_names):
    for col in col_names:
        df[col] = df[col].apply(literal_eval)
    return df

def literal_all_cols(df):
    for col in df.columns:
        try:
            df[col] = df[col].apply(literal_eval)
        except:
            pass
    return df

def get_numeric_label(label, label_to_idx_dict):
    if label in label_to_idx_dict.keys():
        return label_to_idx_dict[label]
    else:
        return -1

def get_textual_label(label, idx_to_label_dict):
    if label in idx_to_label_dict.keys():
        return idx_to_label_dict[label]
    else:
        return -1


def sample_balanced_random_set(df, random_sample_size):
    class_sample_size = random_sample_size // len(list(set(df['label'].tolist())))

    df_grouped = df.groupby('label')

    selected_set = []
    remaining_set = []
    for group in df_grouped.groups.keys():
        label_df = df_grouped.get_group(group)

        random_indices = random.sample(range(label_df.shape[0]),class_sample_size)
        other_indices = [i for i in range(label_df.shape[0]) if i not in random_indices]

        random_selected = label_df.iloc[random_indices]
        remaining = label_df.iloc[other_indices]

        selected_set.append(random_selected)
        remaining_set.append(remaining)

    selected_set = pandas.concat(selected_set)
    remaining_set = pandas.concat(remaining_set)

    return selected_set, remaining_set


def balance_set(unbalanced_df, column):
    unbalanced_df_grouped = unbalanced_df.groupby(column)

    class_sample_size = min([unbalanced_df_grouped.get_group(label).shape[0] for label in unbalanced_df_grouped.groups.keys()])
    selected_set = []
    remaining_set = []
    for group in unbalanced_df_grouped.groups.keys():
        label_df = unbalanced_df_grouped.get_group(group)

        random_indices = random.sample(range(label_df.shape[0]),class_sample_size)
        other_indices = [i for i in range(label_df.shape[0]) if i not in random_indices]

        random_selected = label_df.iloc[random_indices]
        selected_set.append(random_selected)

        if len(other_indices)>0:
            remaining = label_df.iloc[other_indices]
            remaining_set.append(remaining)

    selected_set = pandas.concat(selected_set)
    remaining_set = pandas.concat(remaining_set)

    return selected_set, remaining_set