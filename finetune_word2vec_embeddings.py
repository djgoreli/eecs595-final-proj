import gensim
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
import gensim.downloader as api
import pandas as pd
from keras.preprocessing.text import Tokenizer
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
from embedding_utils import preprocess

# Get notes from csv
patient_notes = pd.read_csv("MIMIC_III_train.csv")
print(len(patient_notes))

# Preprocess text
patient_notes['TEXT'] = patient_notes['TEXT'].apply(preprocess)

# Tokenize
MAX_NB_WORDS = 50000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(patient_notes['TEXT'].values)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(patient_notes['TEXT'].values)

print('Found %s unique tokens.' % len(word_index))

# Generate list of sentences
# sentences_tokenized = patient_notes['TEXT'].astype('str').tolist()
# sentences_tokenized = [s.split() for s in sentences_tokenized]
# print(sentences_tokenized[0])


# Creating a reverse dictionary
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

# Creating texts 
sentences_tokenized = list(map(sequence_to_text, sequences))
print(sentences_tokenized[0])

# Load MIMIC-III embeddings for size 300 embeddings
mimic_iii_embeddings = Word2Vec(vector_size=300, min_count=1)
mimic_iii_embeddings.build_vocab(sentences_tokenized)
total_examples = mimic_iii_embeddings.corpus_count
print("Loaded MIMIC-III model with ", total_examples, " examples")

# Load word2vec embeddings
pretrained_path = "../pretrained-embeddings/GoogleNews-vectors-negative300.bin.gz"
pretrained_model = KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
print("Loaded word2vec model")

print("Model similarity between pulmonary and circulatory is ", pretrained_model.similarity('pulmonary','circulatory'))
print("Model similarity between artery and vein is ", pretrained_model.similarity('artery','vein'))

# Add the pre-trained model vocabulary
mimic_iii_embeddings.build_vocab(pretrained_model.index_to_key, update=True)
mimic_iii_embeddings.wv.vectors_lockf = np.ones(len(mimic_iii_embeddings.wv))
mimic_iii_embeddings.wv.intersect_word2vec_format(pretrained_path, binary=True, lockf=1.0)
print("Added pretrained word2vec model vocab with ", mimic_iii_embeddings.corpus_count, " examples")

# Fine tune the pre-trained embedding model
mimic_iii_embeddings.train(sentences_tokenized, total_examples=total_examples, epochs=mimic_iii_embeddings.epochs)
print("Fine-tuned model similarity between pulmonary and circulatory is ", mimic_iii_embeddings.wv.similarity('pulmonary','circulatory'))
print("Fine-tuned model similarity between artery and vein is ", mimic_iii_embeddings.wv.similarity('artery','vein'))

# Save the fine-tuned word2vec embeddings
finetuned_path = "../pretrained-embeddings/GoogleNews-vectors-finetuned.txt"
mimic_iii_embeddings.wv.save_word2vec_format(finetuned_path, binary=False)