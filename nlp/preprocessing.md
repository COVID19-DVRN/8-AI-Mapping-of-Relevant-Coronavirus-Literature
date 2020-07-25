# Data preprocessing

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites
### Libraries
* gensim (version 3.8.0)
* nltk (version 3.4.5)

### Programs

preprocessing.py - The program runs on preprocessed text to produce nlp items (Supplemental Information 1 in the paper). Stemming, removing stop words, and creating bigrams are done for the text. We removed all punctuation and numbers from the text and lemmatized the remaining words to reduce inflectional forms. We also leveraged existing NLP packages (Gensim) to identify word pairs, or "bigrams". 
