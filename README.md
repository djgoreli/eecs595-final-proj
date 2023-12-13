# EECS 595 Final Project: Medical Diag'notes'is

## Overview
This repository focuses on the task of medical diagnosis classification using deep learning and natural language processing (NLP) techniques. The primary objective is to build a model capable of accurately categorizing medical diagnoses based on textual data. The core implementation is provided in the "classification_embeddings.ipynb" Jupyter Notebook.

## Word Embeddings

### 1. `finetune_glove_embeddings.py`

#### Overview
Run `finetune_glove_embeddings.py` to create a fine-tuned embeddings text file based on the GloVe Wiki Gigaword 300-dimensional dataset. This script performs the following steps:

1. Preprocesses and tokenizes the MIMIC-III training data, building a vocabulary for domain-specific embeddings.
2. Loads the GloVe embeddings from a local file.
3. Converts GloVe embeddings to word2vec format for easier processing.
4. Intersects the GloVe and domain-specific embeddings, combining the vast vocabulary of the Wiki Gigaword embeddings with the domain-specific, smaller MIMIC-III training embeddings.
5. Saves the result in a file named `glove_model_fine_tuned.txt`.

#### Usage
Run the script in a Python environment with the necessary dependencies installed.

```bash
python finetune_glove_embeddings.py
```

---

### 2. `finetune_word2vec_embeddings.py`

#### Overview
Run `finetune_word2vec_embeddings.py` to create a fine-tuned embeddings text file based on the word2vec Google News 300-dimensional dataset. This script performs the following steps:

1. Preprocesses and tokenizes the MIMIC-III training data, building a vocabulary for domain-specific embeddings.
2. Loads the word2vec embeddings from a local file (already in the correct format).
3. Intersects the word2vec and domain-specific embeddings, combining the vast vocabulary of the Google News embeddings with the domain-specific, smaller MIMIC-III training embeddings.
4. Saves the result in a file named `GoogleNews-vectors-finetuned.txt`.

#### Usage
Run the script in a Python environment with the necessary dependencies installed.

```bash
python finetune_word2vec_embeddings.py
```

---

### 3. `embedding_utils.py`

#### Overview
The `embedding_utils.py` file contains helper functions useful for preprocessing and converting embedding formats prior to classification. This file is not meant to be run independently.

#### Usage
Import the necessary functions from `embedding_utils.py` into other scripts for preprocessing and converting embeddings.

## Diagnosis Prediction Model

### 1. `classification_embeddings.ipynb`

#### Overview
The main Jupyter Notebook containing the implementation of the diagnosis classification model. This notebook covers data loading, preprocessing, model creation, training, evaluation, and additional tasks such as similarity prediction. The notebook may need to be modified to accept the correct embedding file.

#### Usage
Open and run the notebook "classification_embeddings.ipynb" in a Jupyter environment such as Google Collab. Follow the instructions and execute each cell in sequence to train the model and perform evaluations.
