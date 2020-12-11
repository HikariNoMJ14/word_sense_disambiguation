from wordnet_api import get_hyponyms, get_hypernyms


def add_hyper_hypo_glosses(synset_id, nym):
    glosses = []

    if nym == 'hyper':
        nyms = get_hypernyms(synset_id)
    elif nym == 'hypo':
        nyms = get_hyponyms(synset_id)
    else:
        raise Exception('Wrong nym')

    for n in nyms:
        glosses.append(n.definition())

    return glosses


if __name__ == "__main__":
    filename = './data/mono/training/semcor/semcor_n.tsv'

