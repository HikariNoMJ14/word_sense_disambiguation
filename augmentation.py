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
        from_model_name=f'transformer.wmt19.en-{language}',
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


def back_translate(aug, data, ignore_identical=False):
    translated = aug.augment(data)

    if ignore_identical:
        results = []

        for i, t in enumerate(translated):
            if t.lower().replace(' ', '') != data[i].lower().replace(' ', ''):
                results.append(t)
                # print(f'Original:    {data[i]}')
                # print(f'Translated:  {t}')
                # print('-----------------------')

        return results
    else:
        return translated




def add_context_words(aug, data, n=1):
    results = aug.augment(data, n=n)

    return results


def merge_dfs(file_1, file_2, file_out):
    first_line = True
    with open(file_out, 'w') as fo:
        for line in open(file_1, 'r'):
            fo.write(line)
        for line in open(file_2, 'r'):
            if first_line:
                first_line = False
            else:
                fo.write(line)


def augment_context(filename, output_filename, context_params):
    df = pd.read_csv(filename, delimiter='\t')

    n_sent = 1
    chunksize = 250

    if os.path.exists(output_filename):
        os.remove(output_filename)

    with open(output_filename, 'w') as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsynset_id\n')
        for k, g in df.groupby(np.arange(len(df)) // chunksize):
            aug = create_context_aug(context_params)
            print(f'group {k}, {len(g)}')

            g['sentence_aug'] = aug.augment(list(g.sentence.values), n=n_sent)
            g['sentence_aug'] = g['sentence_aug'].str.replace(r'" (.*) - (.*) "', r'" \1-\2 "')

            for i, row in g.iterrows():
                if i % 2 == 1:
                    sentence = row['sentence']
                else:
                    sentence = row['sentence_aug']

                f.write(row['target_id'] + '\t' +
                        str(row['label']) + '\t' +
                        sentence + '\t' +
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
        for k, g in df.groupby(np.arange(len(df)) // chunksize):
            print(f'group {k}, {len(g)}')

            rgx = r'(.*) : '

            g['target_word'] = g.gloss.str.extract(rgx)

            g['gloss_aug'] = aug.augment(list(g.gloss.str.replace(rgx, '').values))

            for i, row in g.iterrows():
                f.write(row['target_id'] + '\t' +
                        str(row['label']) + '\t' +
                        row['sentence'] + '\t' +
                        row['target_word'] + " : " + row['gloss_aug'] + '\t' +
                        row['synset_id'] + '\n')


def add_hypernym_to_gloss(file_in, file_out):
    df_in = pd.read_csv(file_in, delimiter='\t')

    with open(file_out, 'w') as fo:
        fo.write('target_id\tlabel\tsentence\tgloss\tsynset_id\n')

        for i, row in df_in.iterrows():
            hyper_gloss = add_hyper_hypo_glosses(row['synset_id'], 'hyper', n_hyper=1)
            hyper_gloss = hyper_gloss[0] if len(hyper_gloss) > 0 else ''

            if i % 1000 == 0:
                print(hyper_gloss)

            fo.write(row['target_id'] + '\t' \
                     + str(row['label']) + '\t' \
                     + row['sentence'] + '\t' \
                     + row['gloss'] + ' ^ ' + hyper_gloss + '\t' \
                     + row['synset_id'] + '\n')


if __name__ == "__main__":

    # -----------------------------------------------------------

    params = {
        'n': 1,
        'aug_p': 0.15,
        'top_p': 3,
        'top_k': 3,
        'temperature': 0.9
    }

    filename = './data/mono/training/semcor/semcor_n_bbase_de_final.tsv'
    output_filename = f'./data/mono/training/semcor/semcor_n_final_base_cbbase_de.tsv'

    augment_context(filename, output_filename, params)

    # filename = './data/mono/training/semcor/semcor_n_final.tsv'
    # output_filename = f'./data/mono/training/semcor/semcor_n_final_bbase.tsv'
    #
    # back_translate_gloss(filename, output_filename)

    # -----------------------------------------------------------

    # file_1 = './data/mono/training/semcor/semcor_n_bbase_ru_final.tsv'
    # file_2 = f'./data/mono/training/semcor/semcor_n_final_only_context1.tsv'
    # file_out = f'./data/mono/training/semcor/semcor_n_final_base_bbase_cbase_ru.tsv'
    #
    # merge_dfs(file_1, file_2, file_out)

    # -----------------------------------------------------------

    # bbase = pd.read_csv('./data/mono/training/semcor/semcor_n_final_context1_only_back.tsv',
    #                     delimiter='\t')
    # bbase = bbase.iloc[:401936, :]
    #
    # base_cbase = pd.read_csv('./data/mono/training/semcor/semcor_n_final_context1.tsv',
    #                  delimiter='\t')
    #
    # df = pd.concat([bbase, base_cbase], axis=0)
    #
    # print(df.shape)

    # -----------------------------------------------------------

    # file_1 = './data/mono/training/semcor/semcor_n_final.tsv'
    # file_2 = './data/mono/training/semcor/semcor_n_final_bbase.tsv'
    # file_3 = './data/mono/training/semcor/semcor_n_final_only_context1.tsv'
    # file_out = './data/mono/training/semcor/semcor_n_final_base_bbase_cbase.tsv'
    #
    # first_line = True
    # i = 0
    # with open(file_out, 'w') as fo:
    #     for line in open(file_1, 'r'):
    #         fo.write(line)
    #     for line in open(file_2, 'r'):
    #         if first_line:
    #             first_line = False
    #         else:
    #             fo.write(line)
    #     first_line = True
    #     for line in open(file_3, 'r'):
    #         if first_line:
    #             first_line = False
    #         else:
    #             fo.write(line)

    # -----------------------------------------------------------

    # file_in = './data/mono/evaluation/ALL/ALL_n_final.tsv'
    # file_out = './data/mono/evaluation/ALL/ALL_n_final_hyper_concatenate.tsv'
    #
    # add_hypernym_to_gloss(file_in, file_out)
