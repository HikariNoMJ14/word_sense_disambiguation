from nltk.corpus import wordnet as wn
import xml.etree.ElementTree as ET
import re

position_dict = {
    'case-sensitive': 0,
    'case-insensitive': 1,
    'only-pos': 2
}


def extract_str(string, position):
    return " ".join([x.split('/')[position] for x in string.split(' ')])


def generate(file_name):
    tree = ET.ElementTree(file=file_name)
    root = tree.getroot()

    sentences = []
    poss = []
    targets = []
    targets_index_start = []
    targets_index_end = []
    lemmas = []

    for doc in root:
        for sent in doc:
            sentence = []
            pos = []
            target = []
            target_index_start = []
            target_index_end = []
            lemma = []
            for token in sent:
                assert token.tag == 'wf' or token.tag == 'instance'
                if token.tag == 'wf':
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append('X')
                        lemma.append(token.attrib['lemma'])
                if token.tag == 'instance':
                    target_start = len(sentence)
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append(token.attrib['id'])
                        lemma.append(token.attrib['lemma'])
                    target_end = len(sentence)
                    assert ' '.join(sentence[target_start:target_end]) == token.text
                    target_index_start.append(target_start)
                    target_index_end.append(target_end)
            sentences.append(sentence)
            poss.append(pos)
            targets.append(target)
            targets_index_start.append(target_index_start)
            targets_index_end.append(target_index_end)
            lemmas.append(lemma)

    gold_keys = []
    with open(file_name[:-len('.data.xml')] + '.gold.key.txt', "r", encoding="utf-8") as m:
        key = m.readline().strip().split()
        while key:
            gold_keys.append(key[1])
            key = m.readline().strip().split()

    output_file = file_name[:-len('.data.xml')] + '.csv'
    with open(output_file, "w", encoding="utf-8") as g:
        g.write('sentence\ttarget_index_start\ttarget_index_end\ttarget_id\ttarget_lemma\ttarget_pos\tsense_key\n')
        num = 0
        for i in range(len(sentences)):
            for j in range(len(targets_index_start[i])):
                sentence = ' '.join(sentences[i])
                target_start = targets_index_start[i][j]
                target_end = targets_index_end[i][j]
                target_id = targets[i][target_start]
                target_lemma = lemmas[i][target_start]
                target_pos = poss[i][target_start]
                sense_key = gold_keys[num]
                num += 1
                g.write('\t'.join(
                    (sentence, str(target_start), str(target_end), target_id, target_lemma, target_pos, sense_key)))
                g.write('\n')


def generate_2(file_name, method='case-sensitive'):
    tree = ET.ElementTree(file=file_name)
    root = tree.getroot()

    position = position_dict[method]

    sentences = []
    poss = []
    targets_id = []
    targets_index_start = []
    targets_index_end = []
    lemmas = []
    scores = []
    sense_keys = []

    for lex in root:
        lex_item = lex.get('item').split('.')
        lemma = lex_item[0]
        pos = lex_item[1]
        for inst in lex:
            id = inst.get('id')
            score = float(inst.get('score'))
            for c in inst:
                if c.tag == 'answer':
                    sense_key = c.get('sensekey')
                else:
                    text = (c.text + b''.join(map(ET.tostring, c)).decode('utf-8')).strip()
                    target_index_start = len(re.search('(.*)<head>', text).group(1).split(' '))
                    # there doesn't seem to be any instance of a target with multiple words
                    # target_index_end = target_index_start + len(target.split(' ')) - 1
                    target_index_end = target_index_start + 1
                    sentence = extract_str(
                        text.replace('<head>', '').replace('</head>', ''),
                        position
                    )

            sentences.append(sentence)
            poss.append(pos)
            targets_id.append(id)
            targets_index_start.append(target_index_start)
            targets_index_end.append(target_index_end)
            scores.append(score)
            lemmas.append(lemma)
            sense_keys.append(sense_key)

    output_file = file_name[:-len('.data.xml')] + '.tsv'
    with open(output_file, "w", encoding="utf-8") as g:
        g.write(
            'sentence\ttarget_index_start\ttarget_index_end\ttarget_id\ttarget_lemma\ttarget_pos\tsense_key\tscore\n')
        for i in range(len(sentences)):
            sentence = sentences[i]
            target_start = targets_index_start[i]
            target_end = targets_index_end[i]
            target_id = targets_id[i]
            target_lemma = lemmas[i]
            target_pos = poss[i]
            sense_key = sense_keys[i]
            score = scores[i]
            g.write('\t'.join(
                (sentence, str(target_start), str(target_end), target_id, target_lemma, target_pos,
                 sense_key, str(score))))
            g.write('\n')


def change_encoding(file_name):
    with open(file_name) as f:
        lines = f.readlines()

    lines[0] = '<?xml version="1.0" encoding="UTF-8"?>\n'

    with open(file_name, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    languages = ['IT']

    for lang in languages:
        file_name = f'./train-o-matic/{lang}/evaluation-framework-ims-training.utf8.xml'
        # change_encoding(file_name)
        generate_2(file_name, method='case-sensitive')
