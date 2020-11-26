from py_babelnet.calls import BabelnetAPI


def get_api_key():
    with open('./babelnet.key') as f:
        api_key = f.readline()

    return api_key


def get_api():
    return BabelnetAPI(get_api())


def get_synset_ids(lemma, lang):
    api = get_api()
    res = api.get_synset_ids(lemma=lemma, searchLang=lang)

    synset_ids = []
    for synset in res:
        synset_ids.append(synset['id'])

    return synset_ids


def get_glosses(synset_id, lang):
    api = get_api()
    res = api.get_synset(id=synset_id, targetLang=lang)

    glosses = []
    for gloss in res['glosses']:
        glosses.append(gloss['gloss'])

    return glosses

if __name__ == "__main__":
    get_glosses('bn:00066578n', 'IT')
