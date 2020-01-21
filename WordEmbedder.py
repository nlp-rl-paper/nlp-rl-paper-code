from gensim.models import Word2Vec,KeyedVectors
import os
from Embedder import Embedder
import numpy as np

GOOGLENEWS_VECTOR_PATH = "GoogleNews-vectors-negative300.bin.gz"
#this class implemets the word embedding feature
#the main purpose is to convert the game state parse string to image
#sentance, which  is a string will be converted to a 2d numpy array with the following dim:
#[word1,word2,word3...,wordN] -> numpy.array([300,N]) where N == sentance_len
class WordEmbedder(Embedder):
    #constructor:
    #pre_trained_model_path: the path to a pretrained word embedding model
    #parser: a Parser object
    def __init__(self,parser,word_embedding_len,sentance_len,pre_trained_model_path=GOOGLENEWS_VECTOR_PATH):
        #Embedder.__init__(parser,word_embedding_len,sentance_len,pre_trained_model_path)
        super(WordEmbedder,self).__init__(parser,word_embedding_len,sentance_len)
        self.pretrained_model_path = pre_trained_model_path
        self.lang_model = KeyedVectors.load_word2vec_format(self.pretrained_model_path,binary=True)



