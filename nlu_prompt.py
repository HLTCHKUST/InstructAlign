SENTIMENT_ID = "[INPUT]\nApakah sentimen dari teks tersebut? [OPTIONS]? [LABELS_CHOICE]"
DATA_TO_ID_PROMPT = {
    'emot_nusantara_text': '[INPUT]\nApakah emosi dari teks diatas? [OPTIONS]? [LABELS_CHOICE]',
    'imdb_jv_nusantara_text': SENTIMENT_ID,
    'indolem_sentiment_nusantara_text':SENTIMENT_ID,
    'smsa_nusantara_text':SENTIMENT_ID,
    'indonli_nusantara_pairs': '[INPUT_A]\nBerdasarkan kutipan sebelumnya, apakah benar bahwa "[INPUT_B]"? [OPTIONS]? [LABELS_CHOICE]',
    'su_emot_nusantara_text': '[INPUT]\nApakah emosi dari teks diatas? [OPTIONS]? [LABELS_CHOICE]',    
    
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
}

SENTIMENT_ID2 = "Apakah sentimen dari teks berikut?\nTeks: [INPUT]\nJawab dengan [OPTIONS]: [LABELS_CHOICE]"
DATA_TO_ID2_PROMPT = {
    'emot_nusantara_text': 'Apakah emosi dari teks ini?\nTeks: [INPUT]\nJawab dengan [OPTIONS]: [LABELS_CHOICE]',
    'imdb_jv_nusantara_text': SENTIMENT_ID2,
    'indolem_sentiment_nusantara_text':SENTIMENT_ID2,
    'smsa_nusantara_text': SENTIMENT_ID2,
    'indonli_nusantara_pairs': '[INPUT_A]\n\nPertanyaan: Apakah kaliamt tersebut mengimplikasikan bahwa "[INPUT_B]"? [OPTIONS]? [LABELS_CHOICE]',
    'su_emot_nusantara_text': 'Apakah emosi dari teks ini?\n[INPUT]\nJawab dengan [OPTIONS]? [LABELS_CHOICE]',
    
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
}

SENTIMENT_ID3 =  'Teks: [INPUT]\n\nTolong prediksikan sentimen dari teks diatas. Jawab dengan [OPTIONS]: [LABELS_CHOICE]'
DATA_TO_ID3_PROMPT = {
    'emot_nusantara_text': 'Teks: [INPUT]\n\nTolong prediksikan emosi dari teks diatas. Jawab dengan [OPTIONS]: [LABELS_CHOICE]',
    'imdb_jv_nusantara_text': SENTIMENT_ID3,
    'indolem_sentiment_nusantara_text': SENTIMENT_ID3,
    'smsa_nusantara_text': SENTIMENT_ID3,
    'indonli_nusantara_pairs': 'Diberikan [INPUT_A]. Apakah kalimat tersebut sesuai dengan [INPUT_B]? [OPTIONS]? [LABELS_CHOICE]',
    'su_emot_nusantara_text': 'Teks: [INPUT]\n\nTolong prediksikan emosi dari teks diatas. Jawab dengan [OPTIONS]: [LABELS_CHOICE]',

    'nusax_senti_ace_nusantara_text': SENTIMENT_ID3,
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
}


SENTIMENT_EN = "[INPUT]\nWhat would be the sentiment of the text above? [OPTIONS]? [LABELS_CHOICE]"
DATA_TO_EN_PROMPT = {
    'emot_nusantara_text': '[INPUT]\nWhat would be the emotion of the text above? [OPTIONS]? [LABELS_CHOICE]',
    'imdb_jv_nusantara_text': SENTIMENT_EN,
    'indolem_sentiment_nusantara_text':SENTIMENT_EN,
    'smsa_nusantara_text': SENTIMENT_EN,
    'indonli_nusantara_pairs': '[INPUT_A]\nBased on the previous passage, is it true that "[INPUT_B]"? Yes, no, or maybe? [OPTIONS]? [LABELS_CHOICE]',
    'su_emot_nusantara_text': '[INPUT]\nWhat would be the emotion of the text above? [OPTIONS]? [LABELS_CHOICE]',

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
}

SENTIMENT_EN2 =  'What is the sentiment of this text?\nText: [INPUT]\nAnswer with [OPTIONS]: [LABELS_CHOICE]'
DATA_TO_EN2_PROMPT = {
    'emot_nusantara_text': 'What is the emotion of this text?\nText: [INPUT]\nAnswer with [OPTIONS]: [LABELS_CHOICE]',
    'imdb_jv_nusantara_text': SENTIMENT_EN2,
    'indolem_sentiment_nusantara_text': SENTIMENT_EN2,
    'smsa_nusantara_text': SENTIMENT_EN2,
    'indonli_nusantara_pairs': '[INPUT_A]\n\nQuestion: Does this imply that "[INPUT_B]"? Yes, no, or maybe? [LABELS_CHOICE]',
    'su_emot_nusantara_text': 'What is the emotion of this text?\nText: [INPUT]\nAnswer with [OPTIONS]: [LABELS_CHOICE]',

    'nusax_senti_ace_nusantara_text': SENTIMENT_EN2,
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
}

SENTIMENT_EN3 =  'Text: [INPUT]\n\nPlease classify the sentiment of above text. Answer with [OPTIONS]: [LABELS_CHOICE]'
DATA_TO_EN3_PROMPT = {
    'emot_nusantara_text': 'Text: [INPUT]\n\nPlease classify the emotion of above text. Answer with [OPTIONS]: [LABELS_CHOICE]',
    'imdb_jv_nusantara_text': SENTIMENT_EN3,
    'indolem_sentiment_nusantara_text': SENTIMENT_EN3,
    'smsa_nusantara_text': SENTIMENT_EN3,
    'indonli_nusantara_pairs': 'Given that [INPUT_A]. Does it follow that [INPUT_B]? yes, no, or maybe? [LABELS_CHOICE]',
    'su_emot_nusantara_text': 'Text: [INPUT]\n\nPlease classify the emotion of above text. Answer with [OPTIONS]: [LABELS_CHOICE]',

    'nusax_senti_ace_nusantara_text': SENTIMENT_EN3,
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