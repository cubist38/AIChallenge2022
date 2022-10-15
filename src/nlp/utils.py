import nltk
import numpy as np

nltk.download('punkt')

def tokenize(paragraph):
    words = nltk.word_tokenize(paragraph)
    return words

def calc_dot_score(sentence_embeddings, paragraph_embeddings):
    print(sentence_embeddings.shape)
    print(paragraph_embeddings.shape)
    scores = np.dot(sentence_embeddings, np.transpose(paragraph_embeddings))
    return scores