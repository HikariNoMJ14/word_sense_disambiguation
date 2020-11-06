from bs4 import BeautifulSoup
import pandas as pd

y = BeautifulSoup(open('./data/ita+xml/ita/wn-ita-lmf.xml'), features='xml')
lex = y.LexicalResource.Lexicon
language = lex['language']
version = lex['version']
synset_prefix = f"{language}-{version}-"


ita_wn = pd.DataFrame(columns=['id', 'synset', 'language', 'pos', 'written_form'])

for entry in lex.findAll('LexicalEntry'):
    lemma = entry.Lemma
    written_form = lemma['writtenForm']
    pos = lemma['partOfSpeech']
    entry_id = entry['id']

    for sense in entry.findAll('Sense'):
        ita_wn = ita_wn.append({
            'id': entry_id,
            'synset': sense['id'].replace(f"{entry_id}_", ""),
            'language': language,
            'pos': pos,
            'written_form': written_form
        }, ignore_index=True)


ita_wn.to_csv('data/final/ita_wn.csv')





