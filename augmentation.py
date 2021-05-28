import nltk
import os
import numpy as np
import pandas as pd
from wordnet_api import get_hyponyms, get_hypernyms
import nlpaug.augmenter.word as naw
from transformers import MarianMTModel, MarianTokenizer

nltk_stopwords = nltk.corpus.stopwords.words('english')


class MarianBT:

    def __init__(self, language):
        self.language = language

        target_model_name = f'Helsinki-NLP/opus-mt-en-{self.language}'
        self.target_tokenizer = MarianTokenizer.from_pretrained(target_model_name)
        self.target_model = MarianMTModel.from_pretrained(target_model_name).to('cuda')

        en_model_name = f'Helsinki-NLP/opus-mt-{self.language}-en'
        self.en_tokenizer = MarianTokenizer.from_pretrained(en_model_name)
        self.en_model = MarianMTModel.from_pretrained(en_model_name).to('cuda')

    def augment(self, texts):
        # Translate from source to target language
        fr_texts = self.translate(texts, self.target_model, self.target_tokenizer, language=self.language, device='cuda')

        # Translate from target language back to source language
        back_translated_texts = self.translate(fr_texts, self.en_model, self.en_tokenizer, language='en', device='cuda')

        return back_translated_texts

    def translate(self, texts, model, tokenizer, language="fr", device='cuda'):
        # Prepare the text data into appropriate format for the model
        template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
        src_texts = [template(text) for text in texts]

        # Tokenize the texts
        encoded = tokenizer.prepare_seq2seq_batch(src_texts, return_tensors='pt').to(device)

        # Generate translation using model
        translated = model.generate(**encoded).to(device)

        # Convert the generated tokens indices back into text
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)

        return translated_texts


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


def create_context_wn_aug(params):
    return naw.synonym.SynonymAug(
        aug_p=params['aug_p'],
        stopwords=nltk_stopwords
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
                sentence = row['sentence_aug']

                f.write(row['target_id'] + '\t' +
                        str(row['label']) + '\t' +
                        sentence + '\t' +
                        row['gloss'] + '\t' +
                        row['synset_id'] + '\n')


def augment_context_wn(filename, output_filename, context_params):
    df = pd.read_csv(filename, delimiter='\t')

    n_sent = 1

    if os.path.exists(output_filename):
        os.remove(output_filename)

    with open(output_filename, 'w') as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsynset_id\n')

        aug = create_context_wn_aug(context_params)

        df['sentence_aug'] = aug.augment(list(df.sentence.values), n=n_sent)
        df['sentence_aug'] = df['sentence_aug'].str.replace(r'" (.*) - (.*) "', r'" \1-\2 "')

        for i, row in df.iterrows():
            sentence = row['sentence_aug']

            f.write(row['target_id'] + '\t' +
                    str(row['label']) + '\t' +
                    sentence + '\t' +
                    row['gloss'] + '\t' +
                    row['synset_id'] + '\n')


def back_translate_gloss(filename, output_filename, language='de', method="marian"):
    df = pd.read_csv(filename, delimiter='\t')
    chunksize = 50

    if method == "marian":
        aug = MarianBT(language)
    else:
        aug = create_back_aug(language)

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

            # if i % 1000 == 0:
            #     print(hyper_gloss)

            fo.write(row['target_id'] + '\t' \
                     + str(row['label']) + '\t' \
                     + row['sentence'] + '\t' \
                     + row['gloss'] + ' [SEP] ' + hyper_gloss + '\t' \
                     + row['synset_id'] + '\n')


def back_translate_context(language='de'):
    df = pd.read_csv('./data/mono/training/semcor/semcor.tsv', delimiter='\t')

    unique_sentence = df['sentence'].unique()
    all_sentences = {}

    aug = create_back_aug(language)

    translated_sentences = aug.augment(list(unique_sentence))

    len(translated_sentences)

    for i, translated_sentence in enumerate(translated_sentences):
        sentence = unique_sentence[i]

        print(sentence)
        print("*********")
        print(translated_sentence)
        print("------------------")

        all_sentences[sentence] = translated_sentence

    for i, row in df.iterrows():
        tis = row.loc['target_index_start']
        tie = row.loc['target_index_end']

        try:
            translated_sentence = np.array(all_sentences[row['sentence']].split(' '))
        except:
            break

        target_word = row['sentence'].split(' ')[tis:tie]
        # print(target_word)

        if " ".join(target_word) in translated_sentence:
            target_index_start = np.where(translated_sentence == target_word[0])[0]
            target_index_end = np.where(translated_sentence == target_word[-1])[0]

            if len(target_index_start) > 1 or len(target_index_end) > 1:
                print(target_index_start, target_index_end)

            new_row = row.copy()
            new_row['sentence'] = " ".join(translated_sentence)
            new_row['target_index_start'] = target_index_start[0]
            new_row['target_index_end'] = target_index_end[0] + 1

            df = df.append(new_row)

    with open('./data/mono/training/semcor/semcor_context_bt.tsv', 'w') as fo:
        fo.write('sentence\ttarget_index_start\ttarget_index_end\ttarget_id\ttarget_lemma\ttarget_pos\ttarget_pos\n')
        for i, row in df.iterrows():
            fo.write(row['sentence'] + '\t' \
                     + str(row['target_index_start']) + '\t' \
                     + str(row['target_index_end']) + '\t' \
                     + row['target_id'] + '\t' \
                     + row['target_lemma'] + '\t' \
                     + row['target_pos'] + '\t' \
                     + row['sense_key'] + '\n')


if __name__ == "__main__":
    # -----------------------------------------------------------

    # params = {
    #     'n': 1,
    #     'aug_p': 0.15,
    #     'top_p': 3,
    #     'top_k': 3,
    #     'temperature': 0.9
    # }
    #
    # filename = './data/mono/training/semcor/semcor_final_base.tsv'
    # output_filename = f'./data/mono/training/semcor/semcor_final_cwbase.tsv'
    #
    # augment_context_wn(filename, output_filename, params)

    # language = 'ru'
    # filename = 'data/mono/training/semcor/semcor_final_base.tsv'
    # output_filename = f'./data/mono/training/semcor/semcor_final_bbase_{language}.tsv'
    #
    # back_translate_gloss(filename, output_filename, language=language, method='marian')

    # -----------------------------------------------------------

    # file_1 = './data/mono/training/semcor/semcor_final_base.tsv'
    # file_2 = f'./data/mono/training/semcor/semcor_final_cbase.tsv'
    # file_out = f'./data/mono/training/semcor/semcor_final_base_cbase.tsv'
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

    # file_in = './data/mono/training/semcor/semcor_final_base.tsv'
    # file_out = './data/mono/training/semcor/semcor_final_base_hyper_concatenate.tsv'
    #
    # add_hypernym_to_gloss(file_in, file_out)

    back_translate_context()
