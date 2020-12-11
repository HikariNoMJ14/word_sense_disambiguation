import pandas as pd
import babelnet_api
import wordnet_api
from augmentation import add_hyper_hypo_glosses
from utils import POS_MAPPING

all_synset_ids = {}
glosses = {}


def generate_auxiliary(train_file_name, train_file_final_name, mode, language, augment=[]):
    train_data = pd.read_csv(train_file_name, sep="\t", na_filter=False).values

    with open(train_file_final_name, "w", encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsynset_id\n')
        num = 0
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
                        print(
                            f'WARNING: Couldn\'t find synset id for {train_data[i][6]}, getting sense by lemma instead')
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
                    if mode == 'mono':
                        glosses[synset_id] = wordnet_api.get_glosses(synset_id)
                    elif mode == 'multi':
                        glosses[synset_id] = babelnet_api.get_glosses(synset_id, language)

                    if 'hyper' in augment:
                        glosses[synset_id].extend(add_hyper_hypo_glosses(synset_id, 'hyper'))

                    if 'hypo' in augment:
                        glosses[synset_id].extend(add_hyper_hypo_glosses(synset_id, 'hypo'))

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


if __name__ == "__main__":
    modes = ['mono']
    language = 'IT'
    datasets = {
        'training': ['semcor'],
        'evaluation': ['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']
    }
    augment = [[], ['hyper'], ['hypo'], ['hyper', 'hypo']]

    for mode in modes:
        for type, dataset in datasets.items():
            for ds in dataset:
                file_name = f'./data/{mode}/{type}/{ds}/{ds}_n'

                for augment_mode in augment:
                    output_final_name = file_name
                    for aug in augment_mode:
                        output_final_name = output_final_name + f'_{aug}'
                    output_final_name = output_final_name + '_final'

                    generate_auxiliary(
                        f'{file_name}.tsv',
                        f'{output_final_name}.tsv',
                        mode=mode,
                        language=language,
                        augment=augment
                    )
