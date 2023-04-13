from nltk import word_tokenize
import random    

def random_infilling(text, replacement='___', prob=0.2):
    tokens = word_tokenize(text)
    text_len = len(tokens)
    mask = np.random.binomial(1, 1-prob, text_len) == 0
    for i, bitmask in enumerate(mask):
        if bitmask:
            tokens[i] = replacement
    return " ".join(tokens)

def random_deletion(text, prob=0.2):
    text = np.array(word_tokenize(text))
    text_len = len(text)
    mask = np.random.binomial(1, 1-prob, text_len) == 1
    return " ".join(text[mask])

def random_permutation(text):
    text_split = word_tokenize(text)
    random.shuffle(text_split)
    return " ".join(text_split)

def do_augment(text, aug_type):
    if aug_type == 'infilling':
        return random_infilling(text)
    elif aug_type == 'deletion':
        return random_deletion(text)
    elif aug_type == 'permutation':
        return random_permutation(text)