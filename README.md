# EECS 595 Final Project: Medical Diag'notes'is

# Word Embeddings

Run _finetune_glove_embeddings.py_ in order to create a fine-tuned embeddings text file based on the GloVe Wiki Gigaword 300-dimensional dataset. This file first preprocesses and tokenizes the MIMIC-III training data,
building a vocabulary for domain-specific embeddings. It then loads in the GloVe embeddings from a local file, and converts them to word2vec format for easier processing. These two embeddings are intersected, combining
the vast vocabulary of the Wiki Gigaword embeddings with the domain-specific, smaller MIMIC-III training embeddings. The result is a file called _glove_model_fine_tuned.txt_ containing these embeddings. Note that this 
implementation relies on local file paths as the pretrained embeddings are too large to be shared over GitHub. 

Run _finetune_word2vec_embeddings.py_ in order to create a fine-tuned embeddings text file based on the word2vec Google News 300-dimensional dataset. This file first preprocesses and tokenizes the MIMIC-III training data,
building a vocabulary for domain-specific embeddings. It then loads in the word2vec embeddings from a local file, which are already in the correct format. These two embeddings are intersected, combining
the vast vocabulary of the Google News embeddings with the domain-specific, smaller MIMIC-III training embeddings. The result is a file called _GoogleNews-vectors-finetuned.txt_ containing these embeddings. Note that this 
implementation relies on local file paths as the pretrained embeddings are too large to be shared over GitHub. 

# Diagnosis Classification