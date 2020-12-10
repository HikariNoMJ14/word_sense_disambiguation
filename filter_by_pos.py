import os
import pandas as pd


pos_mapping = {
    'N': 'NOUN',
    'A': 'ADJ',
    'V': 'VERB',
    'R': 'ADV'
}


def filter_by_pos(file_path, pos):

    df = pd.read_csv(file_path, delimiter='\t')

    df = df[(df.target_pos == pos) | (df.target_pos == pos_mapping[pos])]

    folder, file_name = os.path.split(file_path)
    name = file_name[:-4]
    extension = file_name[-4:]

    df.to_csv(f"{folder}/{name}-{pos}{extension}", sep='\t', index=False)


if __name__ == "__main__":
    filter_by_pos('./data/multi/evaluation/semeval-2015/data/semeval-2015-task-13-es.tsv', 'N')
