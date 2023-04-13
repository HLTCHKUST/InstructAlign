import datasets

subset_langs = ['eng_Latn', 'ind_Latn', 'sun_Latn', 'jav_Latn', 'bug_Latn', 'ace_Latn', 'bjn_Latn', 'ban_Latn', 'min_Latn']

lang_map = {
    'eng_Latn': 'English', 'ind_Latn': 'Indonesian', 'sun_Latn': 'Sundanese', 
    'jav_Latn': 'Javanese', 'bug_Latn': 'Buginese', 'ace_Latn': 'Acehnese', 
    'bjn_Latn': 'Banjarese', 'ban_Latn': 'Balinese', 'min_Latn': 'Minangkabau'
}

def load_flores_datasets(pivot_langs=['eng_Latn']):
    def inject_lang(row, lang1, lang2):
        row['lang1'] = lang_map[lang1]
        row['lang2'] = lang_map[lang2]
        return row

    dsets = {}
    for lang1 in pivot_langs:
        for lang2 in ['ind_Latn', 'sun_Latn', 'jav_Latn', 'bug_Latn', 'ace_Latn', 'bjn_Latn', 'ban_Latn', 'min_Latn']:
            if lang1 != lang2:
                subset = f'{lang1}-{lang2}'
                dset = datasets.load_dataset('facebook/flores', subset)
                dset = dset.rename_columns({f'sentence_{lang1}': 'sentence1', f'sentence_{lang2}': 'sentence2'})
                dset = dset.map(inject_lang, fn_kwargs={'lang1': lang1, 'lang2': lang2}, load_from_cache_file=False)
                dsets[subset] = dset

                subset = f'{lang2}-{lang1}'
                dset = datasets.load_dataset('facebook/flores', subset)
                dset = dset.rename_columns({f'sentence_{lang2}': 'sentence1', f'sentence_{lang1}': 'sentence2'})
                dset = dset.map(inject_lang, fn_kwargs={'lang1': lang2, 'lang2': lang1}, load_from_cache_file=False)
                dsets[subset] = dset
                
    dset_subsets = []
    for key in dsets.keys():
        for split in ['dev', 'devtest']:
            dset_subsets.append(dsets[key][split])
    combined_dset = datasets.concatenate_datasets(dset_subsets)

    return combined_dset.train_test_split(test_size=1000, seed=0)