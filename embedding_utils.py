import numpy as np
import regex as re
import gensim.downloader as api
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping

# Text preprocessing function
def preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I|re.A)  # Remove non-alphanumeric characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)  # Remove single characters from the start
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Substitute multiple spaces with single space
    text = re.sub(r'^b\s+', '', text)  # Remove prefixed 'b'
    text = text.lower()  # Convert to lowercase
    return text


# Download word2vec Google news model for 300-dimensional embeddings
# 3,000,000 vectors, ~100 billion words
def load_word2vec_embedding_from_gensim():
    word2vec_embeddings = api.load("word2vec-google-news-300")  # download the model and return as object ready for use
    print(
        "Loaded with vocabulary size {}".format(len(list(word2vec_embeddings.index_to_key))
        )
    )
    return word2vec_embeddings


# Download Glove wiki model for 300-dimensional embeddings
# 400,000 vectors, ~6 billion words
def load_glove_embedding_from_gensim():
    glove_embeddings = api.load("glove-wiki-gigaword-300")
    print(
        "Loaded with vocabulary size {}".format(len(list(glove_embeddings.index_to_key))
        )
    )
    return glove_embeddings


def gensim_to_keras_embedding(wv_embeddings, train_embeddings=False):
    """Get a Keras 'Embedding' layer with weights set from Word2Vec model's learned word embeddings.

    Parameters
    ----------
    train_embeddings : bool
        If False, the returned weights are frozen and stopped from being updated.
        If True, the weights can / will be further updated in Keras.

    Returns
    -------
    `keras.layers.Embedding`
        Embedding layer, to be used as input to deeper network layers.

    """
    keyed_vectors = wv_embeddings  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array    
    index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=train_embeddings,
    )
    return layer



# load word2vec format embeddings as a dict
def load_word2vec_embedding_from_file(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r')
    lines = file.readlines()[1:]
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
        return embedding
    

# load glove format embeddings as a dict
def load_glove_embedding_from_file(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r')
    
    # create a map of words to vectors
    embedding = dict()
    for line in file:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
        return embedding
    
    file.close()

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0 for 300 size embeddings
    weight_matrix = np.zeros((vocab_size, 300))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
        return weight_matrix
    


# # load embedding from file
# raw_embedding = load_word2vec_embedding('embedding_word2vec.txt')

# # get vectors in the right order
# embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)

# # create the embedding layer
# embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)