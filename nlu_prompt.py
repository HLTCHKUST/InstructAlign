SENTIMENT_ID = '[INPUT]\nApakah sentimen dari teks tersebut? [OPTIONS]? [LABELS_CHOICE]'
EMOT_ID = '[INPUT]\nApakah emosi dari teks diatas? [OPTIONS]? [LABELS_CHOICE]'
NLI_ID = '[INPUT_A]\nBerdasarkan kutipan sebelumnya, apakah benar bahwa "[INPUT_B]"? [OPTIONS]? [LABELS_CHOICE]'
DATA_TO_ID_PROMPT = {
    'emot_nusantara_text': EMOT_ID,
    'imdb_jv_nusantara_text': SENTIMENT_ID,
    'indolem_sentiment_nusantara_text':SENTIMENT_ID,
    'smsa_nusantara_text':SENTIMENT_ID,
    'indonli_nusantara_pairs': NLI_ID,
    'su_emot_nusantara_text': EMOT_ID,
    
    'nusax_senti_ace_nusantara_text':SENTIMENT_ID,
    'nusax_senti_ban_nusantara_text': SENTIMENT_ID,
    'nusax_senti_bjn_nusantara_text': SENTIMENT_ID,
    'nusax_senti_bug_nusantara_text': SENTIMENT_ID,
    'nusax_senti_eng_nusantara_text': SENTIMENT_ID,
    'nusax_senti_ind_nusantara_text': SENTIMENT_ID,
    'nusax_senti_jav_nusantara_text': SENTIMENT_ID,
    'nusax_senti_mad_nusantara_text': SENTIMENT_ID,
    'nusax_senti_min_nusantara_text': SENTIMENT_ID,
    'nusax_senti_nij_nusantara_text': SENTIMENT_ID,
    'nusax_senti_sun_nusantara_text': SENTIMENT_ID,
    'nusax_senti_bbc_nusantara_text': SENTIMENT_ID,
    
    'nusa_kalimat_emot_abs': EMOT_ID,
    'nusa_kalimat_emot_bew': EMOT_ID,
    'nusa_kalimat_emot_bhp': EMOT_ID,
    'nusa_kalimat_emot_btk': EMOT_ID,
    'nusa_kalimat_emot_jav': EMOT_ID,
    'nusa_kalimat_emot_mad': EMOT_ID,
    'nusa_kalimat_emot_mak': EMOT_ID,
    'nusa_kalimat_emot_min': EMOT_ID,
    'nusa_kalimat_emot_mui': EMOT_ID,
    'nusa_kalimat_emot_rej': EMOT_ID,
    'nusa_kalimat_emot_sun': EMOT_ID,

    'nusa_kalimat_senti_abs': SENTIMENT_ID,
    'nusa_kalimat_senti_bew': SENTIMENT_ID,
    'nusa_kalimat_senti_bhp': SENTIMENT_ID,
    'nusa_kalimat_senti_btk': SENTIMENT_ID,
    'nusa_kalimat_senti_jav': SENTIMENT_ID,
    'nusa_kalimat_senti_mad': SENTIMENT_ID,
    'nusa_kalimat_senti_mak': SENTIMENT_ID,
    'nusa_kalimat_senti_min': SENTIMENT_ID,
    'nusa_kalimat_senti_mui': SENTIMENT_ID,
    'nusa_kalimat_senti_rej': SENTIMENT_ID,
    'nusa_kalimat_senti_sun': SENTIMENT_ID,
}

SENTIMENT_ID2 = "Apakah sentimen dari teks berikut?\nTeks: [INPUT]\nJawab dengan [OPTIONS]: [LABELS_CHOICE]"
EMOT_ID2 = 'Apakah emosi dari teks ini?\nTeks: [INPUT]\nJawab dengan [OPTIONS]: [LABELS_CHOICE]'
NLI_ID2 = '[INPUT_A]\nBerdasarkan kutipan sebelumnya, apakah benar bahwa "[INPUT_B]"? [OPTIONS]? [LABELS_CHOICE]'
DATA_TO_ID2_PROMPT = {
    'emot_nusantara_text': EMOT_ID2,
    'imdb_jv_nusantara_text': SENTIMENT_ID2,
    'indolem_sentiment_nusantara_text':SENTIMENT_ID2,
    'smsa_nusantara_text': SENTIMENT_ID2,
    'indonli_nusantara_pairs': NLI_ID2,
    'su_emot_nusantara_text': EMOT_ID2,
    
    'nusax_senti_ace_nusantara_text':SENTIMENT_ID2,
    'nusax_senti_ban_nusantara_text': SENTIMENT_ID2,
    'nusax_senti_bjn_nusantara_text': SENTIMENT_ID2,
    'nusax_senti_bug_nusantara_text': SENTIMENT_ID2,
    'nusax_senti_eng_nusantara_text': SENTIMENT_ID2,
    'nusax_senti_ind_nusantara_text': SENTIMENT_ID2,
    'nusax_senti_jav_nusantara_text': SENTIMENT_ID2,
    'nusax_senti_mad_nusantara_text': SENTIMENT_ID2,
    'nusax_senti_min_nusantara_text': SENTIMENT_ID2,
    'nusax_senti_nij_nusantara_text': SENTIMENT_ID2,
    'nusax_senti_sun_nusantara_text': SENTIMENT_ID2,
    'nusax_senti_bbc_nusantara_text': SENTIMENT_ID2,
    
    'nusa_kalimat_emot_abs': EMOT_ID2,
    'nusa_kalimat_emot_bew': EMOT_ID2,
    'nusa_kalimat_emot_bhp': EMOT_ID2,
    'nusa_kalimat_emot_btk': EMOT_ID2,
    'nusa_kalimat_emot_jav': EMOT_ID2,
    'nusa_kalimat_emot_mad': EMOT_ID2,
    'nusa_kalimat_emot_mak': EMOT_ID2,
    'nusa_kalimat_emot_min': EMOT_ID2,
    'nusa_kalimat_emot_mui': EMOT_ID2,
    'nusa_kalimat_emot_rej': EMOT_ID2,
    'nusa_kalimat_emot_sun': EMOT_ID2,

    'nusa_kalimat_senti_abs': SENTIMENT_ID2,
    'nusa_kalimat_senti_bew': SENTIMENT_ID2,
    'nusa_kalimat_senti_bhp': SENTIMENT_ID2,
    'nusa_kalimat_senti_btk': SENTIMENT_ID2,
    'nusa_kalimat_senti_jav': SENTIMENT_ID2,
    'nusa_kalimat_senti_mad': SENTIMENT_ID2,
    'nusa_kalimat_senti_mak': SENTIMENT_ID2,
    'nusa_kalimat_senti_min': SENTIMENT_ID2,
    'nusa_kalimat_senti_mui': SENTIMENT_ID2,
    'nusa_kalimat_senti_rej': SENTIMENT_ID2,
    'nusa_kalimat_senti_sun': SENTIMENT_ID2,
}

SENTIMENT_ID3 =  'Teks: [INPUT]\n\nTolong prediksikan sentimen dari teks diatas. Jawab dengan [OPTIONS]: [LABELS_CHOICE]'
EMOT_ID3 = 'Teks: [INPUT]\n\nTolong prediksikan emosi dari teks diatas. Jawab dengan [OPTIONS]: [LABELS_CHOICE]'
NLI_ID3 = 'Diberikan [INPUT_A]. Apakah kalimat tersebut sesuai dengan [INPUT_B]? [OPTIONS]? [LABELS_CHOICE]'
DATA_TO_ID3_PROMPT = {
    'emot_nusantara_text': EMOT_ID3,
    'imdb_jv_nusantara_text': SENTIMENT_ID3,
    'indolem_sentiment_nusantara_text':SENTIMENT_ID3,
    'smsa_nusantara_text': SENTIMENT_ID3,
    'indonli_nusantara_pairs': NLI_ID3,
    'su_emot_nusantara_text': EMOT_ID3,
    
    'nusax_senti_ace_nusantara_text':SENTIMENT_ID3,
    'nusax_senti_ban_nusantara_text': SENTIMENT_ID3,
    'nusax_senti_bjn_nusantara_text': SENTIMENT_ID3,
    'nusax_senti_bug_nusantara_text': SENTIMENT_ID3,
    'nusax_senti_eng_nusantara_text': SENTIMENT_ID3,
    'nusax_senti_ind_nusantara_text': SENTIMENT_ID3,
    'nusax_senti_jav_nusantara_text': SENTIMENT_ID3,
    'nusax_senti_mad_nusantara_text': SENTIMENT_ID3,
    'nusax_senti_min_nusantara_text': SENTIMENT_ID3,
    'nusax_senti_nij_nusantara_text': SENTIMENT_ID3,
    'nusax_senti_sun_nusantara_text': SENTIMENT_ID3,
    'nusax_senti_bbc_nusantara_text': SENTIMENT_ID3,
    
    'nusa_kalimat_emot_abs': EMOT_ID3,
    'nusa_kalimat_emot_bew': EMOT_ID3,
    'nusa_kalimat_emot_bhp': EMOT_ID3,
    'nusa_kalimat_emot_btk': EMOT_ID3,
    'nusa_kalimat_emot_jav': EMOT_ID3,
    'nusa_kalimat_emot_mad': EMOT_ID3,
    'nusa_kalimat_emot_mak': EMOT_ID3,
    'nusa_kalimat_emot_min': EMOT_ID3,
    'nusa_kalimat_emot_mui': EMOT_ID3,
    'nusa_kalimat_emot_rej': EMOT_ID3,
    'nusa_kalimat_emot_sun': EMOT_ID3,

    'nusa_kalimat_senti_abs': SENTIMENT_ID3,
    'nusa_kalimat_senti_bew': SENTIMENT_ID3,
    'nusa_kalimat_senti_bhp': SENTIMENT_ID3,
    'nusa_kalimat_senti_btk': SENTIMENT_ID3,
    'nusa_kalimat_senti_jav': SENTIMENT_ID3,
    'nusa_kalimat_senti_mad': SENTIMENT_ID3,
    'nusa_kalimat_senti_mak': SENTIMENT_ID3,
    'nusa_kalimat_senti_min': SENTIMENT_ID3,
    'nusa_kalimat_senti_mui': SENTIMENT_ID3,
    'nusa_kalimat_senti_rej': SENTIMENT_ID3,
    'nusa_kalimat_senti_sun': SENTIMENT_ID3,
}

SENTIMENT_EN = "[INPUT]\nWhat would be the sentiment of the text above? [OPTIONS]? [LABELS_CHOICE]"
EMOT_EN = '[INPUT]\nWhat would be the emotion of the text above? [OPTIONS]? [LABELS_CHOICE]'
NLI_EN = '[INPUT_A]\nBased on the previous passage, is it true that "[INPUT_B]"? Yes, no, or maybe? [OPTIONS]? [LABELS_CHOICE]'
DATA_TO_EN_PROMPT = {
    'emot_nusantara_text': EMOT_EN,
    'imdb_jv_nusantara_text': SENTIMENT_EN,
    'indolem_sentiment_nusantara_text':SENTIMENT_EN,
    'smsa_nusantara_text': SENTIMENT_EN,
    'indonli_nusantara_pairs': NLI_EN,
    'su_emot_nusantara_text': EMOT_EN,
    
    'nusax_senti_ace_nusantara_text':SENTIMENT_EN,
    'nusax_senti_ban_nusantara_text': SENTIMENT_EN,
    'nusax_senti_bjn_nusantara_text': SENTIMENT_EN,
    'nusax_senti_bug_nusantara_text': SENTIMENT_EN,
    'nusax_senti_eng_nusantara_text': SENTIMENT_EN,
    'nusax_senti_ind_nusantara_text': SENTIMENT_EN,
    'nusax_senti_jav_nusantara_text': SENTIMENT_EN,
    'nusax_senti_mad_nusantara_text': SENTIMENT_EN,
    'nusax_senti_min_nusantara_text': SENTIMENT_EN,
    'nusax_senti_nij_nusantara_text': SENTIMENT_EN,
    'nusax_senti_sun_nusantara_text': SENTIMENT_EN,
    'nusax_senti_bbc_nusantara_text': SENTIMENT_EN,
    
    'nusa_kalimat_emot_abs': EMOT_EN,
    'nusa_kalimat_emot_bew': EMOT_EN,
    'nusa_kalimat_emot_bhp': EMOT_EN,
    'nusa_kalimat_emot_btk': EMOT_EN,
    'nusa_kalimat_emot_jav': EMOT_EN,
    'nusa_kalimat_emot_mad': EMOT_EN,
    'nusa_kalimat_emot_mak': EMOT_EN,
    'nusa_kalimat_emot_min': EMOT_EN,
    'nusa_kalimat_emot_mui': EMOT_EN,
    'nusa_kalimat_emot_rej': EMOT_EN,
    'nusa_kalimat_emot_sun': EMOT_EN,

    'nusa_kalimat_senti_abs': SENTIMENT_EN,
    'nusa_kalimat_senti_bew': SENTIMENT_EN,
    'nusa_kalimat_senti_bhp': SENTIMENT_EN,
    'nusa_kalimat_senti_btk': SENTIMENT_EN,
    'nusa_kalimat_senti_jav': SENTIMENT_EN,
    'nusa_kalimat_senti_mad': SENTIMENT_EN,
    'nusa_kalimat_senti_mak': SENTIMENT_EN,
    'nusa_kalimat_senti_min': SENTIMENT_EN,
    'nusa_kalimat_senti_mui': SENTIMENT_EN,
    'nusa_kalimat_senti_rej': SENTIMENT_EN,
    'nusa_kalimat_senti_sun': SENTIMENT_EN,

	'xnli_ar': NLI_EN, 'xnli_bg': NLI_EN, 'xnli_de': NLI_EN, 'xnli_el': NLI_EN,
	'xnli_en': NLI_EN, 'xnli_es': NLI_EN, 'xnli_fr': NLI_EN, 'xnli_hi': NLI_EN,
	'xnli_ru': NLI_EN, 'xnli_sw': NLI_EN, 'xnli_th': NLI_EN, 'xnli_tr': NLI_EN,
	'xnli_ur': NLI_EN, 'xnli_vi': NLI_EN, 'xnli_zh': NLI_EN
}

SENTIMENT_EN2 =  'What is the sentiment of this text?\nText: [INPUT]\nAnswer with [OPTIONS]: [LABELS_CHOICE]'
EMOT_EN2 = 'What is the emotion of this text?\nText: [INPUT]\nAnswer with [OPTIONS]: [LABELS_CHOICE]'
NLI_EN2 = '[INPUT_A]\n\nQuestion: Does this imply that "[INPUT_B]"? Yes, no, or maybe? [LABELS_CHOICE]'
DATA_TO_EN2_PROMPT = {
    'emot_nusantara_text': EMOT_EN2,
    'imdb_jv_nusantara_text': SENTIMENT_EN2,
    'indolem_sentiment_nusantara_text':SENTIMENT_EN2,
    'smsa_nusantara_text': SENTIMENT_EN2,
    'indonli_nusantara_pairs': NLI_EN2,
    'su_emot_nusantara_text': EMOT_EN2,
    
    'nusax_senti_ace_nusantara_text':SENTIMENT_EN2,
    'nusax_senti_ban_nusantara_text': SENTIMENT_EN2,
    'nusax_senti_bjn_nusantara_text': SENTIMENT_EN2,
    'nusax_senti_bug_nusantara_text': SENTIMENT_EN2,
    'nusax_senti_eng_nusantara_text': SENTIMENT_EN2,
    'nusax_senti_ind_nusantara_text': SENTIMENT_EN2,
    'nusax_senti_jav_nusantara_text': SENTIMENT_EN2,
    'nusax_senti_mad_nusantara_text': SENTIMENT_EN2,
    'nusax_senti_min_nusantara_text': SENTIMENT_EN2,
    'nusax_senti_nij_nusantara_text': SENTIMENT_EN2,
    'nusax_senti_sun_nusantara_text': SENTIMENT_EN2,
    'nusax_senti_bbc_nusantara_text': SENTIMENT_EN2,
    
    'nusa_kalimat_emot_abs': EMOT_EN2,
    'nusa_kalimat_emot_bew': EMOT_EN2,
    'nusa_kalimat_emot_bhp': EMOT_EN2,
    'nusa_kalimat_emot_btk': EMOT_EN2,
    'nusa_kalimat_emot_jav': EMOT_EN2,
    'nusa_kalimat_emot_mad': EMOT_EN2,
    'nusa_kalimat_emot_mak': EMOT_EN2,
    'nusa_kalimat_emot_min': EMOT_EN2,
    'nusa_kalimat_emot_mui': EMOT_EN2,
    'nusa_kalimat_emot_rej': EMOT_EN2,
    'nusa_kalimat_emot_sun': EMOT_EN2,

    'nusa_kalimat_senti_abs': SENTIMENT_EN2,
    'nusa_kalimat_senti_bew': SENTIMENT_EN2,
    'nusa_kalimat_senti_bhp': SENTIMENT_EN2,
    'nusa_kalimat_senti_btk': SENTIMENT_EN2,
    'nusa_kalimat_senti_jav': SENTIMENT_EN2,
    'nusa_kalimat_senti_mad': SENTIMENT_EN2,
    'nusa_kalimat_senti_mak': SENTIMENT_EN2,
    'nusa_kalimat_senti_min': SENTIMENT_EN2,
    'nusa_kalimat_senti_mui': SENTIMENT_EN2,
    'nusa_kalimat_senti_rej': SENTIMENT_EN2,
    'nusa_kalimat_senti_sun': SENTIMENT_EN2,

	'xnli_ar': NLI_EN2, 'xnli_bg': NLI_EN2, 'xnli_de': NLI_EN2, 'xnli_el': NLI_EN2,
	'xnli_en': NLI_EN2, 'xnli_es': NLI_EN2, 'xnli_fr': NLI_EN2, 'xnli_hi': NLI_EN2,
	'xnli_ru': NLI_EN2, 'xnli_sw': NLI_EN2, 'xnli_th': NLI_EN2, 'xnli_tr': NLI_EN2,
	'xnli_ur': NLI_EN2, 'xnli_vi': NLI_EN2, 'xnli_zh': NLI_EN2
}

SENTIMENT_EN3 =  'Text: [INPUT]\n\nPlease classify the sentiment of above text. Answer with [OPTIONS]: [LABELS_CHOICE]'
EMOT_EN3 = 'Text: [INPUT]\n\nPlease classify the emotion of above text. Answer with [OPTIONS]: [LABELS_CHOICE]'
NLI_EN3 = 'Given that [INPUT_A]. Does it follow that [INPUT_B]? yes, no, or maybe? [LABELS_CHOICE]'
DATA_TO_EN3_PROMPT = {
    'emot_nusantara_text': EMOT_EN3,
    'imdb_jv_nusantara_text': SENTIMENT_EN3,
    'indolem_sentiment_nusantara_text':SENTIMENT_EN3,
    'smsa_nusantara_text': SENTIMENT_EN3,
    'indonli_nusantara_pairs': NLI_EN3,
    'su_emot_nusantara_text': EMOT_EN3,
    
    'nusax_senti_ace_nusantara_text':SENTIMENT_EN3,
    'nusax_senti_ban_nusantara_text': SENTIMENT_EN3,
    'nusax_senti_bjn_nusantara_text': SENTIMENT_EN3,
    'nusax_senti_bug_nusantara_text': SENTIMENT_EN3,
    'nusax_senti_eng_nusantara_text': SENTIMENT_EN3,
    'nusax_senti_ind_nusantara_text': SENTIMENT_EN3,
    'nusax_senti_jav_nusantara_text': SENTIMENT_EN3,
    'nusax_senti_mad_nusantara_text': SENTIMENT_EN3,
    'nusax_senti_min_nusantara_text': SENTIMENT_EN3,
    'nusax_senti_nij_nusantara_text': SENTIMENT_EN3,
    'nusax_senti_sun_nusantara_text': SENTIMENT_EN3,
    'nusax_senti_bbc_nusantara_text': SENTIMENT_EN3,
    
    'nusa_kalimat_emot_abs': EMOT_EN3,
    'nusa_kalimat_emot_bew': EMOT_EN3,
    'nusa_kalimat_emot_bhp': EMOT_EN3,
    'nusa_kalimat_emot_btk': EMOT_EN3,
    'nusa_kalimat_emot_jav': EMOT_EN3,
    'nusa_kalimat_emot_mad': EMOT_EN3,
    'nusa_kalimat_emot_mak': EMOT_EN3,
    'nusa_kalimat_emot_min': EMOT_EN3,
    'nusa_kalimat_emot_mui': EMOT_EN3,
    'nusa_kalimat_emot_rej': EMOT_EN3,
    'nusa_kalimat_emot_sun': EMOT_EN3,

    'nusa_kalimat_senti_abs': SENTIMENT_EN3,
    'nusa_kalimat_senti_bew': SENTIMENT_EN3,
    'nusa_kalimat_senti_bhp': SENTIMENT_EN3,
    'nusa_kalimat_senti_btk': SENTIMENT_EN3,
    'nusa_kalimat_senti_jav': SENTIMENT_EN3,
    'nusa_kalimat_senti_mad': SENTIMENT_EN3,
    'nusa_kalimat_senti_mak': SENTIMENT_EN3,
    'nusa_kalimat_senti_min': SENTIMENT_EN3,
    'nusa_kalimat_senti_mui': SENTIMENT_EN3,
    'nusa_kalimat_senti_rej': SENTIMENT_EN3,
    'nusa_kalimat_senti_sun': SENTIMENT_EN3,

	'xnli_ar': NLI_EN3, 'xnli_bg': NLI_EN3, 'xnli_de': NLI_EN3, 'xnli_el': NLI_EN3,
	'xnli_en': NLI_EN3, 'xnli_es': NLI_EN3, 'xnli_fr': NLI_EN3, 'xnli_hi': NLI_EN3,
	'xnli_ru': NLI_EN3, 'xnli_sw': NLI_EN3, 'xnli_th': NLI_EN3, 'xnli_tr': NLI_EN3,
	'xnli_ur': NLI_EN3, 'xnli_vi': NLI_EN3, 'xnli_zh': NLI_EN3
}

def get_prompt(prompt_lang):
    if prompt_lang == 'EN':
        return DATA_TO_EN_PROMPT
    elif prompt_lang == 'EN2':
        return DATA_TO_EN2_PROMPT
    elif prompt_lang == 'EN3':
        return DATA_TO_EN3_PROMPT
    elif prompt_lang == 'ID':
        return DATA_TO_ID_PROMPT
    elif prompt_lang == 'ID2':
        return DATA_TO_ID2_PROMPT
    elif prompt_lang == 'ID3':
        return DATA_TO_ID3_PROMPT
    else:
        raise ValueError(f'get_prompt() - Unknown prompt_lang `{prompt_lang}` (options: EN / EN2 / EN3 / ID / ID2 / ID3)')