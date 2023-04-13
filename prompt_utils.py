import random

MONOLINGUAL_PROMPTS = [
    'Denoise the following noisy text in [SOURCE_LANG]: "[SOURCE_TEXT]",  to make a complete sentence. [TARGET_TEXT]',
    "Fix and complete the following [SOURCE_LANG] sentence: [SOURCE_TEXT]\n[TARGET_TEXT]",
    "Text in [SOURCE_LANG]: [SOURCE_TEXT]\nHow would you fix the sentence to make a correct sentence? [TARGET_TEXT]",
    'Denoise the following noisy text "[SOURCE_TEXT]" to make a complete sentence. [TARGET_TEXT]',
    "Fix and complete the following sentence: [SOURCE_TEXT]\n[TARGET_TEXT]",
    "Input text: [SOURCE_TEXT]\nHow would you fix the sentence to make a correct sentence? [TARGET_TEXT]",
]
    
TRANSLATION_PROMPTS = [
    "Translate the following text from [SOURCE_LANG] to [TARGET_LANG].\nText: [SOURCE_TEXT]\nTranslation: [TARGET_TEXT]",
    "[SOURCE_TEXT]\nTranslate the text above from [SOURCE_LANG] to [TARGET_LANG]. [TARGET_TEXT]",
    "Text in [SOURCE_LANG]: [SOURCE_TEXT]\nHow would you translate that in [TARGET_LANG]? [TARGET_TEXT]",
    "Translate the following text to [TARGET_LANG].\nText: [SOURCE_TEXT]\nTranslation: [TARGET_TEXT]",
    "[SOURCE_TEXT]\nTranslate the text above to [TARGET_LANG]. [TARGET_TEXT]",
    "Input text: [SOURCE_TEXT]\nHow would you translate that in [TARGET_LANG]? [TARGET_TEXT]",
]

BILINGUAL_PROMPTS = [
    '[SOURCE_TEXT]. Denoise the previous text in [SOURCE_LANG] to it equivalent to the following [CONTEXT_LANG] sentence: [CONTEXT]\n[TARGET_TEXT]',
    'Context in [CONTEXT_LANG]: [CONTEXT]\nFix the following [SOURCE_LANG] text "[SOURCE_TEXT]" to ensure that the meaning is equivalent with the context. [TARGET_TEXT]',
    "Context in [CONTEXT_LANG]: [CONTEXT]\nNoisy text in [SOURCE_LANG]: [SOURCE_TEXT]\nHow would you fix the [SOURCE_LANG] sentence to make the meaning the same as the context? [TARGET_TEXT]",
    '[SOURCE_TEXT]. Denoise the previous text in [SOURCE_LANG] to it equivalent to the following sentence: [CONTEXT]\n[TARGET_TEXT]',
    'Context: [CONTEXT]\nFix the following [SOURCE_LANG] text "[SOURCE_TEXT]" to ensure that the meaning is equivalent with the context. [TARGET_TEXT]',
    "Context: [CONTEXT]\nNoisy text in [SOURCE_LANG]: [SOURCE_TEXT]\nHow would you fix the [SOURCE_LANG] sentence to make the meaning the same as the [CONTEXT_LANG] sentence? [TARGET_TEXT]",
]

def prompt_monolingual(src_text, tgt_text, src_lang, is_encoder_decoder):
    prompt = random.choice(MONOLINGUAL_PROMPTS)
    prompt = prompt.replace('[SOURCE_TEXT]', src_text)
    prompt = prompt.replace('[SOURCE_LANG]', src_lang)    
    if is_encoder_decoder:
        prompt = prompt.replace('[TARGET_TEXT]', '')
        return (prompt, tgt_text)
    else:
        prompt = prompt.replace('[TARGET_TEXT]', tgt_text)
        return (prompt, prompt)
    
def prompt_translation(src_text, tgt_text, src_lang, tgt_lang, is_encoder_decoder):
    prompt = random.choice(TRANSLATION_PROMPTS)
    prompt = prompt.replace('[SOURCE_LANG]', src_lang)
    prompt = prompt.replace('[TARGET_LANG]', tgt_lang)
    prompt = prompt.replace('[SOURCE_TEXT]', src_text)
    if is_encoder_decoder:
        prompt = prompt.replace('[TARGET_TEXT]', '')
        return (prompt, tgt_text)
    else:
        prompt = prompt.replace('[TARGET_TEXT]', tgt_text)
        return (prompt, prompt)

def prompt_bilingual(src_text, con_text, tgt_text, src_lang, con_lang, is_encoder_decoder):
    prompt = random.choice(BILINGUAL_PROMPTS)
    prompt = prompt.replace('[SOURCE_LANG]', src_lang)
    prompt = prompt.replace('[CONTEXT_LANG]', con_lang)
    prompt = prompt.replace('[SOURCE_TEXT]', src_text)
    prompt = prompt.replace('[CONTEXT]', con_text)
    if is_encoder_decoder:
        prompt = prompt.replace('[TARGET_TEXT]', '')
        return (prompt, tgt_text)
    else:
        prompt = prompt.replace('[TARGET_TEXT]', tgt_text)
        return (prompt, prompt)
    
if __name__ == '__main__':
    prompt_monolingual('___ adalah gembala', 'aku adalah anak gembala', 'indonesian', False)
    prompt_monolingual('___ adalah gembala', 'aku adalah anak gembala', 'indonesian', True)

    prompt_translation('___ adalah gembala', 'aku adalah anak gembala', 'indonesian', 'english', False)
    prompt_translation('___ adalah gembala', 'aku adalah anak gembala', 'indonesian', 'english', True)

    prompt_bilingual('___ adalah gembala', 'CONTEXT adalah gembala', 'aku adalah anak gembala', 'indonesian', 'english', False)
    prompt_bilingual('___ adalah gembala', 'CONTEXT adalah gembala', 'aku adalah anak gembala', 'indonesian', 'english', True)