import gensim
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
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

# Create new text file for Glove embeddings in word2vec format (if it doesn't already exist)
glove_word2vec_path = "../pretrained-embeddings/glove_300_word2vec.txt"
pretrained_path = "../pretrained-embeddings/glove.6B.300d.txt"
# glove2word2vec(pretrained_path, glove_word2vec_path)
# print("Created new text file for Glove fine-tuned embeddings in word2vec format")

# Load MIMIC-III embeddings for size 300 embeddings
mimic_iii_embeddings = Word2Vec(vector_size=300, min_count=1)
mimic_iii_embeddings.build_vocab(sentences_tokenized)
total_examples = mimic_iii_embeddings.corpus_count
print("Loaded MIMIC-III model with ", total_examples, " examples")

# Load Glove embeddings
pretrained_model = KeyedVectors.load_word2vec_format(glove_word2vec_path, binary=False)
print("Loaded Glove model")

print("Model similarity between pulmonary and circulatory is ", pretrained_model.similarity('pulmonary','circulatory'))
print("Model similarity between artery and vein is ", pretrained_model.similarity('artery','vein'))

# Add the pre-trained model vocabulary
mimic_iii_embeddings.build_vocab(pretrained_model.index_to_key, update=True)
mimic_iii_embeddings.wv.vectors_lockf = np.ones(len(mimic_iii_embeddings.wv))
mimic_iii_embeddings.wv.intersect_word2vec_format(glove_word2vec_path, binary=False, lockf=1.0)
print("Added pretrained glove model vocab with ", mimic_iii_embeddings.corpus_count, " examples")

# Fine tune the pre-trained embedding model
mimic_iii_embeddings.train(sentences_tokenized, total_examples=total_examples, epochs=mimic_iii_embeddings.epochs)
print("Fine-tuned model similarity between pulmonary and circulatory is ", mimic_iii_embeddings.wv.similarity('pulmonary','circulatory'))
print("Fine-tuned model similarity between artery and vein is ", mimic_iii_embeddings.wv.similarity('artery','vein'))

# Save the fine-tuned Glove embeddings in word2vec format
finetuned_path = "../pretrained-embeddings/glove_model_fine_tuned.txt"
mimic_iii_embeddings.wv.save_word2vec_format(finetuned_path, binary=False)