import pandas as pd
from babelnet_api import get_synset_ids, get_glosses

all_bn_synset_ids = {}
glosses = {}


def generate_auxiliary(train_file_name, train_file_final_name, language):
    train_data = pd.read_csv(train_file_name, sep="\t", na_filter=False).values

    with open(train_file_final_name, "w", encoding="utf-8") as f:
        f.write('target_id\tlabel\tsentence\tgloss\tsynset_id\n')
        num = 0
        for i in range(len(train_data)):
            assert train_data[i][-2] == "N"
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
            correct_synset_id = train_data[i][6]
            if not lemma in all_bn_synset_ids:
                all_bn_synset_ids[lemma] = get_synset_ids(lemma, language)
            synset_ids = all_bn_synset_ids[lemma]

            for j in range(len(synset_ids)):
                synset_id = synset_ids[j]
                if not synset_id in glosses:
                    glosses[synset_id] = get_glosses(synset_id, language)

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


if __name__ == "__main__":
    dataset = 'semeval-2013'
    language = 'IT'
    file_name = "multilingual-all-words.it"

    train_file_name = f'./{dataset}/data/{file_name}.tsv'
    train_file_final_name = f'./{dataset}/data/{file_name}_sent_cls_ws.tsv'

    generate_auxiliary(train_file_name, train_file_final_name, language)
