import gensim
# for preprocessing
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
# for bigrams
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.stem.porter import *
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet
from gensim import corpora, models
from pprint import pprint
# auxiliary
import numpy as np
import pandas as pd

stemmer = SnowballStemmer('english')

class Preprocessor:
	def __init__(self, text):
		"""
		Initialize NLP preprocessor.

		------PARAMETERS------
		text: a dataframe column of str text

		------FUNCTIONS------
		preprocess: preprocesses text
		create_nlp_items_from_preprocessed: processes preprocessed text to produce nlp items
		"""
		self.text_ = text

	def preprocess(self, lemmatize=True, stopwords=[], min_token_length=3):
		"""
		Takes a dataframe's text column and preprocesses it by:
			* Removing typical English stop words and removing stop words designated by user
			* Lemmatizing words
        
		------PARAMETERS------
		text: dataframe column
		stopwords: list of *lower case* stop words to remove
		min_token_length: minimum character length of a word (e.g, if it is 3, 'an' is removed)
    
		------ATTRIBUTES STORED------
		preprocessed_text_: a list of lists for of words preprocessed
		"""
		self.preprocessed_text_ = self.text_.map(
			lambda p: self._preprocess_row(p, lemmatize, stopwords=stopwords, min_token_length=min_token_length)
		).tolist()
		self.stopwords_ = stopwords
		self.min_token_length_ = min_token_length

	def _preprocess_row(self, text, lemmatize=True, stopwords=[], min_token_length=3):
		"""
		Takes a dataframe's text column and preprocesses it by:
			* Removing typical English stop words and removing stop words designated by user
			* Lemmatizing words
        
		------PARAMETERS------
		text: dataframe column
		stopwords: list of *lower case* stop words to remove
		min_token_length: minimum character length of a word (e.g, if it is 3, 'an' is removed)
    
		------OUTPUT------
		result: a list of words preprocessed. Use .map() functionality in pandas to compute
			dataframe column
		"""
		# Lemmatize and stem preprocessing steps on the data set.
		def lemmatize_stemming(text):
			return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
		result = []
		for token in gensim.utils.simple_preprocess(text):
			if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > min_token_length:
				if lemmatize:
					result.append(lemmatize_stemming(token))
				else:
					result.append(token)
		result = [item for item in result if item not in stopwords]

		return result

	def get_bigrams_from_preprocessed(self, min_count=0.1, threshold=10., scoring='default'):
		"""
        Computes bigrams after preprocessing. NOTE: overwrites preprocessed_text_ attribute.

        ------PARAMETERS------
        min_count: minimum count of bigrams to be included
        threshold: scoring threshold  for bigrams for inclusion
        scoring: gensim Phrases scoring function to evaluate bigrams for threshold
		"""
		x = Phrases(self.preprocessed_text_, min_count=min_count, threshold=threshold, scoring=scoring)
		x = Phraser(x)

		bigram_token = []
		for sent in self.preprocessed_text_:
			bigram_token.append(x[sent])
		
		self.preprocessed_text_ = bigram_token

	def create_nlp_items_from_preprocessed_df(self,
											  no_below=2,
											  no_above=1.0,
											  keep_n=None,
											  verbose=True):
		"""
		Creates key NLP items from a pandas dataframe column that's already been preprocessed ('preprocess' function).
		------PARAMETERS------
		no_below: int, only include words that appear in this many documents
		no_above: float, exclude all words that appear in this proportion of documents (1.0 = 100%)
		keep_n: maximum number of terms to keep
		verbose: show progress with output previews and etc.
    
		------ATTRIBUTES STORED------
		dictionary_: A dictionary
		bow_corpus_: The bag of words corpus corresponding with the dictionary
		tfidf_: A tfidf 'model' for gensim
		tfidf_corpus_: A tfidf in corpus form
		tfidf_sparse_: A sparse tfidf *matrix*, readily usable for further calculations.
		"""

		# Create dictionary
		dictionary = gensim.corpora.Dictionary(self.preprocessed_text_)
		if verbose:
			count = 0
			for k, v in dictionary.iteritems():
				print(k, v)
				count += 1
				if count > 10:
					break
			print('\n ' + str(len(dictionary)) + ' unique words. \n')
        
		# Filter dictionary's extremes
		if keep_n == None:
			keep_n = len(dictionary)
		dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    
		if verbose:
			print(str(len(dictionary)) + ' unique words after extremes filtered. \n')
			print('')
        
		# Create bow_corpus
		bow_corpus = [dictionary.doc2bow(doc) for doc in self.preprocessed_text_]
    
		if verbose:
			# Preview the corpus
			bow_doc_n = bow_corpus[np.random.choice(list(range(len(self.preprocessed_text_))))]
			for i in range(len(bow_doc_n)):
				print("Word {} (\"{}\") appears {} time.".format(bow_doc_n[i][0], 
														   dictionary[bow_doc_n[i][0]], 
						bow_doc_n[i][1]))
    
		# Create tfidf
		tfidf = models.TfidfModel(bow_corpus)
		corpus_tfidf = tfidf[bow_corpus]
		if verbose:
			print('\n Corpus TF-IDF preview:')
			for doc in corpus_tfidf:
				pprint(doc)
				break
          
		# Make sparse tfidf
		# .T to transpose since it's originally a word-doc matrix, not doc-word matrix (we want rows as docs)
		tfidf_sparse = gensim.matutils.corpus2csc(corpus_tfidf, printprogress=500).T
        
		# Store attributes
		self.dictionary_ = dictionary
		self.bow_corpus_ = bow_corpus
		self.tfidf_ = tfidf
		self.corpus_tfidf_ = corpus_tfidf
		self.tfidf_sparse_ = tfidf_sparse

		self.dictionary_no_above_ = no_above
		self.dictionary_no_below_ = no_below