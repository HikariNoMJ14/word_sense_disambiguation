import xml.etree.ElementTree as ET


def generate(dataset, language, file_name):

    data_file_path = f'./{dataset}/data/{file_name}.xml'

    tree = ET.ElementTree(file=data_file_path)
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
                if token.tag == 'wf' and 'lemma' not in token.attrib:
                    for i in token.text.split(' '):
                        sentence.append(i)
                        pos.append(token.attrib['pos'])
                        target.append('X')
                        lemma.append('nan')
                if token.tag == 'wf' and 'lemma' in token.attrib:
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

    keys_file_path = f'./{dataset}/keys/gold_keys/{language.upper()}/{file_name}.key'
    
    gold_keys = {}
    with open(keys_file_path, "r", encoding="utf-8") as m:
        key = m.readline().strip().split()
        while key:
            gold_keys[key[1]] = key[2]
            key = m.readline().strip().split()

    output_file = data_file_path[:-len('.xml')] + '.tsv'
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
                if target_id in gold_keys:
                    sense_key = gold_keys[target_id]
                    num += 1
                    g.write('\t'.join((sentence, str(target_start), str(target_end), target_id, target_lemma, target_pos, sense_key)))
                    g.write('\n')


if __name__ == "__main__":
    dataset = 'semeval-2015'
    language = 'IT'
    file_name = "semeval-2015-task-13-it"

    generate(dataset, language, file_name)
