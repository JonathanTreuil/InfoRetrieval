import os
import string
import nltk
from nltk.stem import WordNetLemmatizer # Used instead of porterstemmer as it was too aggressively reducing words.
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def preprocess(documents):
    with open('stopwords.txt', 'r') as file:
        stopwords = file.read().splitlines()

    punctuation = set(string.punctuation)
    punctuation.remove('-')  # Keep hyphenated words

    lemmatizer = WordNetLemmatizer()

    def getWordnetPos(treebankTag):
        # Converts treebank tags to wordnet tags.
        if treebankTag.startswith('J'):
            return wordnet.ADJ
        elif treebankTag.startswith('V'):
            return wordnet.VERB
        elif treebankTag.startswith('N'):
            return wordnet.NOUN
        elif treebankTag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  # Default to noun if unknown

    def isNumeric(token):
        # Returns True if the token is numeric (including numbers with punctuation)
        return bool(re.search(r'\d', token))

    tokenDictionary = {}

    for docID, text in documents.items():
        tokens = word_tokenize(text.lower())  # Tokenize text
        tokens = [token for token in tokens if not isNumeric(token) and token not in punctuation and token not in {"'s", "``", "''"}]
        taggedTokens = pos_tag(tokens)  # Part-of-speech tagging
        
        # Lemmatize words with appropriate POS tag
        lemmatizedTokens = [lemmatizer.lemmatize(word, getWordnetPos(tag)) for word, tag in taggedTokens]
        
        # Remove stopwords after lemmatization
        finalTokens = [token for token in lemmatizedTokens if token not in stopwords]
        
        tokenDictionary[docID] = finalTokens
    
    return tokenDictionary