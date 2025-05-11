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
        if treebankTag.startswith('J'):
            return wordnet.ADJ
        elif treebankTag.startswith('V'):
            return wordnet.VERB
        elif treebankTag.startswith('N'):
            return wordnet.NOUN
        elif treebankTag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def isNumeric(token):
        return bool(re.search(r'\d', token))

    tokensList = []
    docIDs = []

    for docID, text in documents.items():
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if not isNumeric(token) and token not in punctuation and token not in {"'s", "``", "''"}]
        taggedTokens = pos_tag(tokens)
        
        lemmatizedTokens = [lemmatizer.lemmatize(word, getWordnetPos(tag)) for word, tag in taggedTokens]
        
        finalTokens = [token for token in lemmatizedTokens if token not in stopwords]
        
        docIDs.append(docID)
        tokensList.append(finalTokens)
    
    return tokensList, docIDs