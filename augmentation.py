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


def create_aug(language):
    return naw.BackTranslationAug(
        from_model_name=f'transformer.wmt19.{language}-en',
        to_model_name=f'transformer.wmt19.{language}-en',
        device='cuda'
    )


def back_translate(aug, data):
    results = []

    translated = aug.augment(data)

    for i, t in enumerate(translated):
        if t.lower().replace(' ', '') != data[i].lower().replace(' ', ''):
            results.append(t)
            # print(f'Original:    {data[i]}')
            # print(f'Translated:  {t}')
            # print('-----------------------')

    return results


if __name__ == "__main__":
    import pandas as pd

    filename = 'Training_Corpora/SemCor/semcor_n.tsv'

    df = pd.read_csv(filename, delimiter='\t')

    aug = create_aug('de')

    back_translate(aug, list(df['sentence'].sample(100)))



