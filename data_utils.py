import datasets
from datasets import Dataset, DatasetDict
from nusacrowd import NusantaraConfigHelper
import glob

""" NusaCrowd Datasets """
TEXT_CLASSIFICATION_TASKS = [
    # # Monolongual Senti, Emot, NLI 
    # 'emot_nusantara_text',
    # 'imdb_jv_nusantara_text',
    'indolem_sentiment_nusantara_text',
    # 'smsa_nusantara_text',    
    # 'indonli_nusantara_pairs',
    # 'su_emot_nusantara_text',
    
    # NusaX Sentiment
    'nusax_senti_ace_nusantara_text',
    'nusax_senti_ban_nusantara_text',
    'nusax_senti_bjn_nusantara_text',
    'nusax_senti_bug_nusantara_text',
    'nusax_senti_eng_nusantara_text',
    'nusax_senti_ind_nusantara_text',
    'nusax_senti_jav_nusantara_text',
    'nusax_senti_mad_nusantara_text',
    'nusax_senti_min_nusantara_text',
    'nusax_senti_nij_nusantara_text',
    'nusax_senti_sun_nusantara_text',
    'nusax_senti_bbc_nusantara_text',
]

def load_nlu_tasks():
    conhelps = NusantaraConfigHelper()
    nlu_datasets = {
        helper.config.name: helper.load_dataset() for helper in conhelps.filtered(lambda x: x.config.name in TEXT_CLASSIFICATION_TASKS)
    }
    return nlu_datasets

""" NusaMenulis Datasets """
NUSA_MENULIS_TASKS = [
    # # Nusa Kalimat Emot
    # ('nusa_kalimat','emot','abs'),
    # ('nusa_kalimat','emot','bew'),
    # ('nusa_kalimat','emot','bhp'),
    # ('nusa_kalimat','emot','btk'),
    # ('nusa_kalimat','emot','jav'),
    # ('nusa_kalimat','emot','mad'),
    # ('nusa_kalimat','emot','mak'),
    # ('nusa_kalimat','emot','min'),
    # ('nusa_kalimat','emot','mui'),
    # ('nusa_kalimat','emot','rej'),
    # ('nusa_kalimat','emot','sun'),
   
    # Nusa Kalimat Senti
    ('nusa_kalimat','senti','abs'),
    ('nusa_kalimat','senti','bew'),
    ('nusa_kalimat','senti','bhp'),
    ('nusa_kalimat','senti','btk'),
    ('nusa_kalimat','senti','jav'),
    ('nusa_kalimat','senti','mad'),
    ('nusa_kalimat','senti','mak'),
    ('nusa_kalimat','senti','min'),
    ('nusa_kalimat','senti','mui'),
    ('nusa_kalimat','senti','rej'),
    ('nusa_kalimat','senti','sun'),

    # Nusa Alinea Emot
    ('nusa_alinea','emot','bew'),
    ('nusa_alinea','emot','btk'),
    ('nusa_alinea','emot','bug'),
    ('nusa_alinea','emot','jav'),
    ('nusa_alinea','emot','mad'),
    ('nusa_alinea','emot','mak'),
    ('nusa_alinea','emot','min'),
    ('nusa_alinea','emot','mui'),
    ('nusa_alinea','emot','rej'),
    ('nusa_alinea','emot','sun'),

#     # Nusa Alinea Paragraph
#     ('nusa_alinea','paragraph','bew'),
#     ('nusa_alinea','paragraph','btk'),
#     ('nusa_alinea','paragraph','bug'),
#     ('nusa_alinea','paragraph','jav'),
#     ('nusa_alinea','paragraph','mad'),
#     ('nusa_alinea','paragraph','mak'),
#     ('nusa_alinea','paragraph','min'),
#     ('nusa_alinea','paragraph','mui'),
#     ('nusa_alinea','paragraph','rej'),
#     ('nusa_alinea','paragraph','sun'),

    # Nusa Alinea Topic
    ('nusa_alinea','topic','bew'),
    ('nusa_alinea','topic','btk'),
    ('nusa_alinea','topic','bug'),
    ('nusa_alinea','topic','jav'),
    ('nusa_alinea','topic','mad'),
    ('nusa_alinea','topic','mak'),
    ('nusa_alinea','topic','min'),
    ('nusa_alinea','topic','mui'),
    ('nusa_alinea','topic','rej'),
    ('nusa_alinea','topic','sun'),
]

def load_single_dataset(dataset, task, lang, base_path='./nusamenulis'):
    data_files = {}
    for path in glob.glob(f'{base_path}/{dataset}-{task}-{lang}-*.csv'):
        split = path.split('-')[-1][:-4]
        data_files[split] = path
        #add path arguments to enable sampled data collection
    return datasets.load_dataset('csv', data_files=data_files)

def load_nusa_menulis_dataset():
    nusa_menulis_dsets = {}
    for (dset, task, lang) in NUSA_MENULIS_TASKS:
        nusa_menulis_dsets[f'{dset}_{task}_{lang}'] = load_single_dataset(dset, task, lang, base_path='./nusamenulis')
    return nusa_menulis_dsets

""" XNLI Dataset """
def load_xnli_dataset():
    xnli_dataset = datasets.load_dataset('xtreme', 'XNLI')
    df = xnli_dataset['test'].to_pandas()
    
    xnli_dsets = {}
    for lang, lang_df in df.groupby('language'):
        lang_df = lang_df[['sentence1', 'sentence2', 'gold_label']]
        lang_df.columns = ['text_1', 'text_2', 'label']
        xnli_dsets[f'xnli_{lang}'] = DatasetDict({'test': Dataset.from_pandas(lang_df.reset_index(drop=True))})
    return xnli_dsets
    
""" FLORES-200 Dataset """
subset_langs = ['eng_Latn', 'ind_Latn', 'sun_Latn', 'jav_Latn', 'bug_Latn', 'ace_Latn', 'bjn_Latn', 'ban_Latn', 'min_Latn']
lang_map = {
    'eng_Latn': 'English', 'ind_Latn': 'Indonesian', 'sun_Latn': 'Sundanese', 
    'jav_Latn': 'Javanese', 'bug_Latn': 'Buginese', 'ace_Latn': 'Acehnese', 
    'bjn_Latn': 'Banjarese', 'ban_Latn': 'Balinese', 'min_Latn': 'Minangkabau'
}

def load_rehearsal_dataset(n_samples=1000, random_seed=42):
    en_dset = datasets.load_dataset('bigscience/xP3', 'en', split='train', streaming=True)
    # id_dset = datasets.load_dataset('bigscience/xP3', 'id', split='train', streaming=True)

    sample_en_dset = en_dset.shuffle(random_seed).take(n_samples)
    # sample_id_dset = id_dset.shuffle(random_seed).take(n_samples)
    
    # return datasets.concatenate_datasets([sample_en_dset, sample_id_dset])
    return sample_en_dset

def load_flores_datasets(pivot_langs=['eng_Latn'], augmentation='multilingual', num_train_ratio=1.0):
    def inject_lang(row, lang1, lang2):
        row['lang1'] = lang_map[lang1]
        row['lang2'] = lang_map[lang2]
        return row

    dsets = {}
    if augmentation == 'monolingual':
        for lang1 in pivot_langs:
            # Load a single dataset from the pivot language as `lang1` and random `lang2`
            lang2 = 'bug_Latn'  # This random `lang2` is not used for training
            subset = f'{lang1}-{lang2}'
            dset = datasets.load_dataset('facebook/flores', subset)
            dset = dset.rename_columns({f'sentence_{lang1}': 'sentence1', f'sentence_{lang2}': 'sentence2'})
            dset = dset.map(inject_lang, fn_kwargs={'lang1': lang1, 'lang2': lang2}, load_from_cache_file=True)
            dsets[subset] = dset
        
    for lang1 in pivot_langs:
        for lang2 in ['ind_Latn', 'sun_Latn', 'jav_Latn', 'bug_Latn', 'ace_Latn', 'bjn_Latn', 'ban_Latn', 'min_Latn']:
            if lang1 != lang2:
                if augmentation != 'monolingual':
                    # If not monolingual take both directions
                    subset = f'{lang1}-{lang2}'
                    dset = datasets.load_dataset('facebook/flores', subset)
                    dset = dset.rename_columns({f'sentence_{lang1}': 'sentence1', f'sentence_{lang2}': 'sentence2'})
                    dset = dset.map(inject_lang, fn_kwargs={'lang1': lang1, 'lang2': lang2}, load_from_cache_file=True)
                    dsets[subset] = dset

                subset = f'{lang2}-{lang1}'
                dset = datasets.load_dataset('facebook/flores', subset)
                dset = dset.rename_columns({f'sentence_{lang2}': 'sentence1', f'sentence_{lang1}': 'sentence2'})
                dset = dset.map(inject_lang, fn_kwargs={'lang1': lang2, 'lang2': lang1}, load_from_cache_file=True)
                dsets[subset] = dset
                
    dset_subsets = []
    for key in dsets.keys():
        for split in ['dev', 'devtest']:
            if 0 < num_train_ratio < 1:
                dset_subsets.append(dsets[key][split].train_test_split(test_size=num_train_ratio, seed=0)['test'])
            else:
                dset_subsets.append(dsets[key][split])
                
    combined_dset = datasets.concatenate_datasets(dset_subsets)

    return combined_dset.train_test_split(test_size=1000, seed=0)