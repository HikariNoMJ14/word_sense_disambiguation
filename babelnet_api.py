from py_babelnet.calls import BabelnetAPI

api = BabelnetAPI('40c0cadc-be6d-47f4-a188-11e899eed2f3')


def get_synset_ids(lemma, lang):
    res = api.get_synset_ids(lemma=lemma, searchLang=lang)

    synset_ids = []
    for synset in res:
        synset_ids.append(synset['id'])

    return synset_ids


def get_glosses(synset_id, lang):
    res = api.get_synset(id=synset_id, targetLang=lang)

    glosses = []
    for gloss in res['glosses']:
        glosses.append(gloss['gloss'])

    return glosses

if __name__ == "__main__":
    get_glosses('bn:00066578n', 'IT')
