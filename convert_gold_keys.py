from nltk.corpus import wordnet as wn

dataset = 'ALL'

if __name__ == "__main__":

    with open(f'./data/mono/evaluation/{dataset}/{dataset}_n.gold.key.txt', 'w') as out:
        for line in open(f'./data/mono/evaluation/{dataset}/{dataset}.gold.key.txt'):
            line = line.replace('\n', '')
            id = line.split(' ')[0]
            sense_key = line.split(' ')[1]

            syn = wn.synset_from_sense_key(sense_key)

            if syn.pos() != 'n':
                continue

            syn_id = syn.name()

            out.write(f"{id} {syn_id}\n")