from wordnet_api import get_hyponyms, get_hypernyms
import nlpaug.augmenter.word as naw


def add_hyper_hypo_glosses(synset_id, nym, n_hyper=3, n_hypo=3):
    glosses = []

    if nym == 'hyper':
        nyms = get_hypernyms(synset_id)[:n_hyper]
    elif nym == 'hypo':
        nyms = get_hyponyms(synset_id)[:n_hypo]
    else:
        raise Exception('Wrong nym')

    for n in nyms:
        glosses.append(n.definition())

    return glosses


def back_translate(data, language):
    aug = naw.BackTranslationAug(
        from_model_name=f'transformer.wmt19.{language}-en',
        to_model_name=f'transformer.wmt19.{language}-en',
        device='gpu'
    )

    results = []

    for d in data:
        translated = aug.augment(d)
        if translated.lower() != d.lower():
            results.append(translated)
            print(f'Original: {d}, Translated: {translated}')

    return results


if __name__ == "__main__":
    filename = './data/mono/training/semcor/semcor_n.tsv'

