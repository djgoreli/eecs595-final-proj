import gensim
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
import gensim.downloader as api
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import regex as re
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
# from keras.callbacks import EarlyStopping

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


# Get notes from csv
patient_notes = pd.read_csv("MIMIC_III_train.csv")
print(len(patient_notes))

# Return "TEXT" column as a list
sentences = patient_notes.TEXT.astype('str').tolist()
print(sentences[0])


# Applying text preprocessing to the data
# patient_notes['TEXT'] = patient_notes['TEXT'].apply(preprocess)
# sentences = patient_notes['TEXT'].astype('str').tolist()

# Tokenize
# MAX_NB_WORDS = 50000
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)
tokenizer = RegexpTokenizer(r'\w+')
sentences_tokenized = [w.lower() for w in sentences]
sentences_tokenized = [tokenizer.tokenize(i) for i in sentences_tokenized]

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