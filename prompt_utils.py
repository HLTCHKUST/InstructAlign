import random

MONOLINGUAL_PROMPTS = [
    'Denoise the following noisy [SOURCE_LANG] text: "[SOURCE_TEXT]",  to make a correct sentence. [TARGET_TEXT]',
    "Fix and complete the following [SOURCE_LANG] sentence: [SOURCE_TEXT]\n[TARGET_TEXT]",
    "Sentence in [SOURCE_LANG]: [SOURCE_TEXT]\nHow would you fix the sentence to make a correct sentence? [TARGET_TEXT]",
    'Denoise the following noisy text "[SOURCE_TEXT]" to make a correct [SOURCE_LANG] sentence. [TARGET_TEXT]',
    "Fix and complete the following sentence: [SOURCE_TEXT]\n[TARGET_TEXT]",
    "Input text: [SOURCE_TEXT]\nHow would you fix the sentence to make a correct [SOURCE_LANG] sentence? [TARGET_TEXT]",
]
    
TRANSLATION_PROMPTS = [
    "Translate the following text from [SOURCE_LANG] to [TARGET_LANG].\nText: [SOURCE_TEXT]\nTranslation: [TARGET_TEXT]",
    "[SOURCE_TEXT]\nTranslate the text above from [SOURCE_LANG] to [TARGET_LANG]. [TARGET_TEXT]",
    "Text in [SOURCE_LANG]: [SOURCE_TEXT]\nHow would you translate that in [TARGET_LANG]? [TARGET_TEXT]",
    "Translate the following text to [TARGET_LANG].\nText: [SOURCE_TEXT]\nTranslation: [TARGET_TEXT]",
    "[SOURCE_TEXT]\nTranslate the text above to [TARGET_LANG]. [TARGET_TEXT]",
    "Input text: [SOURCE_TEXT]\nHow would you translate that into [TARGET_LANG]? [TARGET_TEXT]",
]
    
XSS_PROMPTS = [
    "[SOURCE_LANG] sentence: [SOURCE_TEXT]\n[TARGET_LANG] sentence: [TARGET_TEXT]\nDo the two sentences have the same meaning? [LABEL]",
    "Sentence A: [SOURCE_TEXT]\nSentence B: [TARGET_TEXT]\nDo sentence A and sentence B have the same meaning? [LABEL]",
    "[SOURCE_LANG] sentence: [SOURCE_TEXT]\n[TARGET_LANG] sentence: [TARGET_TEXT]\nAre the two sentences equivalent? [LABEL]",
    "Sentence A: [SOURCE_TEXT]\nSentence B: [TARGET_TEXT]\nAre sentence A and sentence B equivalent? [LABEL]",
    'Is the [SOURCE_LANG] sentence "[SOURCE_TEXT]" equivalent to the [TARGET_LANG] sentence "[TARGET_TEXT]"? [LABEL]',
    'Is the sentence "[SOURCE_TEXT]" equivalent to the sentence "[TARGET_TEXT]"? [LABEL]',
]

BILINGUAL_PROMPTS = [
    '[SOURCE_TEXT]. Denoise the previous [SOURCE_LANG] text to its equivalent sentence in [CONTEXT_LANG]: [CONTEXT]\n[TARGET_TEXT]',
    'Context in [CONTEXT_LANG]: [CONTEXT]\nFix the following [SOURCE_LANG] text "[SOURCE_TEXT]" ensuring the meaning is equivalent with the context. [TARGET_TEXT]',
    "Context in [CONTEXT_LANG]: [CONTEXT]\nNoisy text in [SOURCE_LANG]: [SOURCE_TEXT]\nHow would you fix the [SOURCE_LANG] sentence to make the meaning the same as the context? [TARGET_TEXT]",
    '[SOURCE_TEXT]. Denoise the previous [SOURCE_LANG] sentence to it equivalent sentence: [CONTEXT]\n[TARGET_TEXT]',
    'Context: [CONTEXT]\nFix the following [SOURCE_LANG] text "[SOURCE_TEXT]" ensuring the meaning is equivalent with the context. [TARGET_TEXT]',
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
    
def prompt_xss(src_text, tgt_text, src_lang, tgt_lang, label, is_encoder_decoder):
    prompt = random.choice(XSS_PROMPTS)
    prompt = prompt.replace('[SOURCE_LANG]', src_lang)
    prompt = prompt.replace('[TARGET_LANG]', tgt_lang)
    prompt = prompt.replace('[SOURCE_TEXT]', src_text)
    prompt = prompt.replace('[TARGET_TEXT]', tgt_text)
    if is_encoder_decoder:
        prompt = prompt.replace('[LABEL]', '')
        return (prompt, label)
    else:
        prompt = prompt.replace('[LABEL]', label)
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