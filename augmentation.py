import nltk
import numpy as np
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


if __name__ == "__main__":
    import pandas as pd
    # import re
    #
    # filename = './data/mono/training/semcor/semcor_n_final.tsv'
    #
    # df = pd.read_csv(filename, delimiter='\t')
    #
    # context_params = {
    #     'n': 1,
    #     'aug_p': 0.15,
    #     'top_p': 3,
    #     'top_k': 3,
    #     'temperature': 0.9
    # }
    #
    n_sent = 1
    # chunksize = 250
    # ctx_sentences = []
    #
    # import os
    #
    # output_filename = f'./data/mono/training/semcor/semcor_n_context_{n_sent}.tsv'
    #
    # if os.path.exists(output_filename):
    #     os.remove(output_filename)
    #
    # with open(output_filename, 'w') as f:
    #     f.write('target_id\tlabel\tsentence\tgloss\tsynset_id\n')
    #     for k, g in df.groupby(np.arange(len(df))//chunksize):
    #         aug = create_context_aug(context_params)
    #         print(f'group {k}, {len(g)}')
    #
    #         g['sentence_aug'] = aug.augment(list(g.sentence.values), n=n_sent)
    #         g['sentence_aug'] = g['sentence_aug'].str.replace(r'" (.*) - (.*) "', r'" \1-\2 "')
    #
    #         match_g = r'" (.*) "'
    #
    #         g['target'] = g['sentence'].str.extract(match_g)
    #         g['target_aug'] = g['sentence_aug'].str.extract(match_g)
    #
    #         # if g[g['target'].str.lower() != g['target_aug'].str.lower()].shape[0] > 0:
    #         # print(g[(g['target'].str.lower() != g['target_aug'].str.lower())])
    #
    #         for i, row in g.iterrows():
    #             f.write(row['target_id'] + '\t' +
    #                     str(row['label']) + '\t' +
    #                     row['sentence'] + '\t' +
    #                     row['gloss'] + '\t' +
    #                     row['synset_id'] + '\n')

    # for i, row in df.iterrows():
    #     try:
    #         target_word = re.search(r'".*"', row.sentence)[0]
    #         ctx_sentences = aug.augment(row.sentence, n=5)
    #
    #         ctx_sentence = None
    #         for c in ctx_sentences:
    #             if target_word in c:
    #                 ctx_sentence = c
    #
    #         if ctx_sentence:
    #             ctx_row = row.copy()
    #             ctx_row.sentence = ctx_sentence
    #             df.append(ctx_row)
    #         else:
    #             print('not found!')
    #     except:
    #         pass

    filename = './data/mono/training/semcor/semcor_n_final.tsv'
    context_filename = f'./data/mono/training/semcor/semcor_n_context_{n_sent}.tsv'

    df = pd.read_csv(filename, delimiter='\t')
    ctx_df = pd.read_csv(context_filename, delimiter='\t')

    print(df.shape)
    print(ctx_df.shape)

    df = pd.concat([df, ctx_df], axis=0)
    df.to_csv('./data/mono/training/semcor/semcor_n_final_context1.tsv', index=False)

    # match_g = r'" (.*) "'
    # match_g_2 = r'(.*) :'
    #
    # df['target'] = df['sentence'].str.extract(match_g)
    # ctx_df['target'] = ctx_df['sentence'].str.extract(match_g)
    #
    # # df['target_2'] = df['gloss'].str.extract(match_g_2)
    # # ctx_df['target_2'] = ctx_df['gloss'].str.extract(match_g_2)
    #
    # print(df[df['target'] != df['target_2']].count())
    # print(ctx_df[ctx_df['target'] != ctx_df['target_2']].count())