from nltk.corpus import wordnet as wn


def get_synset_id_from_sense_key(sense_key):
    return wn.synset_from_sense_key(sense_key).name()


def get_synset_ids(lemma, pos):
    ids = []

    for syn in wn.synsets(lemma, pos):
        ids.append(syn.name())

    return ids


def get_glosses(id):
    syn = wn.synset(id)

    return [syn.definition()]


def get_hypernyms(id):
    syn = wn.synset(id)

    return syn.hypernyms()


def get_hyponyms(id):
    syn = wn.synset(id)

    return syn.hyponyms()
