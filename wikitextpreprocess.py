import numpy as np
from collections import Counter

def make_vocab(train, valid, test):
    """
    Makes the word -> index and index -> word mappings
    """
    
    #dictionaries to hold word -> idx and idx -> word mappings
    word2idx = {}
    idx2word = {}

    #with a counter it's easy to do things like set a max vocab size or min frequency
    #before tokens are converted to <UNK> so this is used more as a convenience i'm used to
    token_counter = Counter()

    #split the strings into a list of words and update the counter
    token_counter.update(train.split())
    token_counter.update(valid.split())
    token_counter.update(test.split())
    
    i = 0

    #for each token in the counter, update the dictionary and increase i
    #i.e. first word will be given index 0, etc. etc.
    for token, count in token_counter.items():
        word2idx[token] = i
        idx2word[i] = token
        i+=1

    return word2idx, idx2word

def get_wikitext_words():

    #read in the data
    #this is one long giant string
    with open('wikitext-2/wiki.train.tokens', 'r') as r:
        train = r.read()

    with open('wikitext-2/wiki.valid.tokens', 'r') as r:
        valid = r.read()

    with open('wikitext-2/wiki.test.tokens', 'r') as r:
        test = r.read()

    #creates the word->idx and idx->word mappings
    word2idx, idx2word = make_vocab(train, valid, test)

    #convert from a string to an array of integers using the mapping
    train_idx = [word2idx[t] for t in train.split()]
    valid_idx = [word2idx[t] for t in valid.split()]
    test_idx = [word2idx[t] for t in test.split()]

    #chainer expects numpy arrays (at least the example does) so let's convert
    train_idx = np.array(train_idx)
    valid_idx = np.array(valid_idx)
    test_idx = np.array(test_idx)

    return train_idx, valid_idx, test_idx