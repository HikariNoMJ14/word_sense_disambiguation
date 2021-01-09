import nltk
import os
import numpy as np
import pandas as pd
from wordnet_api import get_hyponyms, get_hypernyms
import nlpaug.augmenter.word as naw

nltk_stopwords = nltk.corpus.stopwords.words('english').append(r'".*"')


def add_hyper_hypo_glosses(synset_id, nym, n_hyper=3, n_hypo=3):
    glosses = []

    if nym == 'hyper':
        nyms = get_hypernyms(synset_id)[:n_hyper]
    elif nym == 'hypo':
        nyms = get_hyponyms(synset_id)[:n_hypo]
    else:
        raise Exception('Wrong nym')

    for n in nyms:
        glosses.append(n.definition())

    return glosses


def create_back_aug(language):
    return naw.BackTranslationAug(
        from_model_name=f'transformer.wmt19.{language}-en',
        to_model_name=f'transformer.wmt19.{language}-en',
        device='cuda'
    )


def create_context_aug(params):

    return naw.ContextualWordEmbsAug(
        model_path='distilbert-base-uncased',
        aug_p=params['aug_p'],
        top_k=params['top_k'],
        top_p=params['top_p'],
        aug_max=None,
        device='cuda',
        temperature=params['temperature'],
        stopwords=nltk_stopwords,
        stopwords_regex=r'".*"'
    )


def back_translate(aug, data):
    results = []

    translated = aug.augment(data)

    for i, t in enumerate(translated):
        if t.lower().replace(' ', '') != data[i].lower().replace(' ', ''):
            results.append(t)
            # print(f'Original:    {data[i]}')
            # print(f'Translated:  {t}')
            # print('-----------------------')

    return results


def add_context_words(aug, data, n=1):
    results = aug.augment(data, n=n)

    return results


def merge_dfs(file_1, file_2, file_out):
    # import csv
    #
    # df_1 = pd.read_csv(file_1, delimiter='\t')
    # df_2 = pd.read_csv(file_2, delimiter='\t')
    #
    # df = pd.concat([df_1, df_2], axis=0)
    # df.to_csv(
    #     file_out,
    #     index=False,
    #     sep='\t',
    #     quoting=csv.QUOTE_NONE,
    #     quotechar='')
    first_line = True
    with open(file_out, 'w') as fo:
        for line in open(file_1, 'r'):
            fo.write(line)
        for line in open(file_2, 'r'):
            if first_line:
                first_line = False
            else:
                fo.write(line)



def augment_context(filename, output_filename):
    df = pd.read_csv(filename, delimiter='\t')

    context_params = {
        'n': 1,
        'aug_p': 0.15,
        'top_p': 3,
        'top_k': 3,
        'temperature': 0.9
    }

    n_sent = 1
    chunksize = 250

    if os.path.exists(output_filename):
        os.remove(output_filename)

    with open(output_filename, 'w') as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsynset_id\n')
        for k, g in df.groupby(np.arange(len(df))//chunksize):
            aug = create_context_aug(context_params)
            print(f'group {k}, {len(g)}')

            g['sentence_aug'] = aug.augment(list(g.sentence.values), n=n_sent)
            g['sentence_aug'] = g['sentence_aug'].str.replace(r'" (.*) - (.*) "', r'" \1-\2 "')

            for i, row in g.iterrows():
                f.write(row['target_id'] + '\t' +
                        str(row['label']) + '\t' +
                        row['sentence_aug'] + '\t' +
                        row['gloss'] + '\t' +
                        row['synset_id'] + '\n')


def back_translate_gloss(filename, output_filename):
    aug = create_back_aug('de')

    df = pd.read_csv(filename, delimiter='\t')
    chunksize = 20000

    if os.path.exists(output_filename):
        os.remove(output_filename)

    with open(output_filename, 'w') as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsynset_id\n')
        for k, g in df.groupby(np.arange(len(df))//chunksize):
            print(f'group {k}, {len(g)}')

            g['sentence_aug'] = aug.augment(list(g.sentence.values))

            for i, row in g.iterrows():
                f.write(row['target_id'] + '\t' +
                        str(row['label']) + '\t' +
                        row['sentence_aug'] + '\t' +
                        row['gloss'] + '\t' +
                        row['synset_id'] + '\n')


if __name__ == "__main__":
    # filename = './data/mono/training/semcor/semcor_n_final.tsv'
    # output_filename = f'./data/mono/training/semcor/semcor_n_final_only_context1.tsv'
    #
    # augment_context(filename, output_filename)

    # filename = './data/mono/training/semcor/semcor_n_final_context1.tsv'
    # output_filename = f'./data/mono/training/semcor/semcor_n_final_context1_only_back.tsv'
    #
    # back_translate_gloss(filename, output_filename)

    file_1 = './data/mono/training/semcor/semcor_n_final_context1.tsv'
    file_2 = f'./data/mono/training/semcor/semcor_n_final_context1_only_back.tsv'
    file_out = f'./data/mono/training/semcor/semcor_n_final_context1_back_all.tsv'

    merge_dfs(file_1, file_2, file_out)


