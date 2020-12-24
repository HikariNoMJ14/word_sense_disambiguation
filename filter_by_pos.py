import os
import pandas as pd

from utils import POS_MAPPING


def filter_by_pos(file_path, pos):

    df = pd.read_csv(file_path, delimiter='\t')

    df = df[(df.target_pos == pos) | (df.target_pos == POS_MAPPING[pos])]

    folder, file_name = os.path.split(file_path)
    name = file_name[:-4]
    extension = file_name[-4:]

    df.to_csv(f"{folder}/{name}_{pos.lower()}{extension}", sep='\t', index=False)


if __name__ == "__main__":
    filter_by_pos('data/mono/training/semcor/semcor.tsv', 'N')
