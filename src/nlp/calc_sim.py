# https://www.sbert.net/examples/applications/computing-embeddings/README.html

from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub

from src.nlp.utils import *


# Load the models
sbert = SentenceTransformer('all-MiniLM-L6-v2')
universal = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def calc_sim(sentence, paragraph, trim_length=128):
    '''
        Calculate the similarity between a sentence and a paragraph.
        
        The longer the paragraph, the more embeddings vector diluted. So we trim the paragraph to sub-paragraphs
        with length of `trim_length` words, and calculate the similarity between the sentence and each sub-paragraph.
        After that, we return the MAX similarity.

        We use two models to calculate the similarity: SBERT and Universal Sentence Encoder.
        The final result will be average of the two models.

        Args:
            - `sentence` (str): The sentence to compare.
            - `paragraph` (str): The paragraph to compare.
            - `trim_length` (int): The length of sub-paragraphs.
    '''
    # Get sub-paragraphs
    words = tokenize(paragraph)
    sub_paragraphs = [' '.join(words[i:i+trim_length]) for i in range(0, len(words), trim_length)]

    # SBert
    sentence_embedding = sbert.encode(sentence, normalize_embeddings=True)
    paragraph_embeddings = sbert.encode(sub_paragraphs, normalize_embeddings=True)
    sbert_score = calc_dot_score(sentence_embedding, paragraph_embeddings)

    # Universal Sentence Encoder
    sentence_embedding = universal([sentence])
    paragraph_embeddings = universal(sub_paragraphs)
    universal_score = calc_dot_score(sentence_embedding, paragraph_embeddings)
    
    # Average
    score = (sbert_score + universal_score) / 2
    return score