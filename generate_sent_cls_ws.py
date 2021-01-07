import pandas as pd
import babelnet_api
import wordnet_api
import numpy as np
from augmentation import add_hyper_hypo_glosses, back_translate, create_back_aug, create_context_aug, add_context_words
from utils import POS_MAPPING


def generate_auxiliary(train_file_name, train_file_final_name, mode, language, augment=[]):
    train_data = pd.read_csv(train_file_name, sep="\t", na_filter=False).values

    all_synset_ids = {}
    glosses = {}

    n_hyper = 3
    n_hypo = 3
    back_translate_lang = 'de'
    context_params = {
        'n': 1,
        'aug_p': 0.15,
        'top_p': 3,
        'top_k': 3,
        'temperature': 0.9
    }

    if 'bbase' in augment or 'bhyper' in augment or 'bhypo' in augment:
        baug = create_back_aug(back_translate_lang)

    if 'cbase' in augment or 'chyper' in augment or 'chypo' in augment:
        caug = create_context_aug(context_params)

    with open(train_file_final_name, "w", encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsynset_id\n')
        num = 0
        tot_hyper = 0
        tot_hypo = 0
        already_seen = 0

        for i in range(len(train_data)):
            assert train_data[i][-2] == "N" or train_data[i][-2] == POS_MAPPING['N']
            orig_sentence = train_data[i][0].split(' ')
            start_id = int(train_data[i][1])
            end_id = int(train_data[i][2])
            sentence = []
            for w in range(len(orig_sentence)):
                if w == start_id or w == end_id:
                    sentence.append('"')
                sentence.append(orig_sentence[w])
            if end_id == len(orig_sentence):
                sentence.append('"')
            sentence = ' '.join(sentence)

            lemma = train_data[i][4]

            # print(f'LEMMA: {lemma}')

            if mode == 'mono':
                try:
                    correct_synset_id = wordnet_api.get_synset_id_from_sense_key(train_data[i][6])
                except Exception as e:
                    # print(e)
                    if mode == 'mono':
                        print(f'WARNING: Couldn\'t find synset id for {train_data[i][6]}, '
                              f'getting sense by lemma instead')
                        tmp_synset_ids = wordnet_api.get_synset_ids(lemma, 'n')
                        if len(tmp_synset_ids) == 1:
                            correct_synset_id = tmp_synset_ids[0]
                        else:
                            print(f'Multiple synsets found')
                            for syn in tmp_synset_ids:
                                print(syn)
                            print('-----------------')
                        continue

            elif mode == 'multi':
                correct_synset_id = train_data[i][6]  # TODO double-check

            if not lemma in all_synset_ids:
                if mode == 'mono':
                    all_synset_ids[lemma] = wordnet_api.get_synset_ids(lemma, 'n')
                elif mode == 'multi':
                    all_synset_ids[lemma] = babelnet_api.get_synset_ids(lemma, language)

            synset_ids = all_synset_ids[lemma]

            for j in range(len(synset_ids)):
                synset_id = synset_ids[j]

                # print(f'SYN ID: {synset_id}')

                if not synset_id in glosses:
                    # print(f'First time seeing synset {synset_id}, current line {num}')
                    if mode == 'mono':
                        glosses[synset_id] = wordnet_api.get_glosses(synset_id)
                    elif mode == 'multi':
                        glosses[synset_id] = babelnet_api.get_glosses(synset_id, language)

                    if 'bbase' in augment:
                        bbase = back_translate(baug, glosses[synset_id])
                        glosses[synset_id].extend(bbase)

                    if 'cbase' in augment:
                        cbase = add_context_words(caug, glosses[synset_id])
                        # print(glosses[synset_id] + cbase)
                        glosses[synset_id].extend(cbase)

                    if 'hyper' in augment:
                        hyper_glosses = add_hyper_hypo_glosses(synset_id, 'hyper', n_hyper=n_hyper)

                        tot_hyper += len(hyper_glosses)
                        # print(f'Hyper: {len(hyper_glosses)}')

                        glosses[synset_id].extend(hyper_glosses)

                        if 'bhyper' in augment:
                            bhyper = back_translate(baug, hyper_glosses)
                            # print(bhyper)
                            glosses[synset_id].extend(bhyper)

                    if 'hypo' in augment:
                        hypo_glosses = add_hyper_hypo_glosses(synset_id, 'hypo', n_hypo=n_hypo)

                        tot_hypo += len(hypo_glosses)
                        # print(f'Hypo: {len(hypo_glosses)}')

                        glosses[synset_id].extend(hypo_glosses)

                        if 'bhypo' in augment:
                            bhypo = back_translate(baug, hypo_glosses)
                            # print(bhypo)
                            glosses[synset_id].extend(bhypo)

                else:
                    already_seen += 1
                    # print(f'Synset already seen {synset_id}, current line {num}')

                if len(glosses) == 0:
                    print(f'WARNING: Glosses missing for synset {synset_id} , {language}')

                for gloss in glosses[synset_id]:
                    label = str(int(synset_id == correct_synset_id))

                    f.write(train_data[i][3] + '\t' +
                            label + '\t' +
                            sentence + '\t' +
                            lemma + " : " + gloss + '\t' +
                            synset_id + '\n')
                    num += 1

            correct_synset_id = None

        print(f"Total number of lemmas: {len(all_synset_ids)}")
        print(f"Total number of synsets: {len(glosses)}")
        print(f"Number of times synsets seen again {already_seen}")

        return num


if __name__ == "__main__":
    modes = ['mono']
    language = 'IT'
    datasets = {
        'training': ['semcor'],
        # 'evaluation': ['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']
        # 'evaluation': ['ALL']
    }
    augment = [
        # [],
        # ['hyper'],
        # ['hypo'],
        # ['hyper', 'hypo']
        # ['hyper', 'hypo', 'bbase', 'bhyper']
        ['cbase']
    ]

    for mode in modes:
        for type, dataset in datasets.items():
            for ds in dataset:
                file_name = f'./data/{mode}/{type}/{ds}/{ds}_n'

                for augment_mode in augment:
                    output_final_name = file_name
                    for aug in augment_mode:
                        output_final_name = output_final_name + f'_{aug}'
                    output_final_name = output_final_name + '_final'

                    n_rows = generate_auxiliary(
                        f'{file_name}.tsv',
                        f'{output_final_name}.tsv',
                        mode=mode,
                        language=language,
                        augment=augment_mode
                    )

                    print(f'{output_final_name}')
                    print(f'Written {n_rows} rows')
