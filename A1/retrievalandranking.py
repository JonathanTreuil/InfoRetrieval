import math
from collections import defaultdict
from math import log

# This file includes all the functions required to do the retrieval and ranking steps.

# Computes the idf scores for each of the terms in the inverted index.
def computeIDF(invertedIndex, numDocs):
    idfScores = {}
    for term, docs in invertedIndex.items():
        # The number of documents containing the term is the length of the docs list
        docFreq = len(docs)
        # Use the IDF formula as specified
        idfScores[term] = log(numDocs / docFreq, 10)
    
    return idfScores


def computeTFIDF(invertedIndex, idfScores, numDocs):
    tfidfScores = {}

    for term, docDict in invertedIndex.items():
        for docID, termFreq in docDict.items():
            # The Term Frequency is the raw count of how many times a word occurs in a document
            # Retrieve the corresponding IDF score for that term
            idf = idfScores.get(term, log(numDocs, 10))  # Default IDF if term is not found
            # Calculate TF-IDF score
            tfidf = termFreq * idf
            # Initialize a nested dictionary if that document hasnt been processed before
            if docID not in tfidfScores:
                tfidfScores[docID] = {}
            # Assign the TF-IDF score to the term for this document
            tfidfScores[docID][term] = tfidf

    return tfidfScores

def computeDocumentLengths(tfidfScores):
    docLengths = {}

    for docID, scores in tfidfScores.items():
        # Calculate the square root of the sum of the squares of TF-IDF scores for the document
        length = math.sqrt(sum(tfidf**2 for tfidf in scores.values()))
        docLengths[docID] = length

    return docLengths

# Function to convert the query into its vector representation using term frequency and inverse document frequency.
def buildQueryVector(queryTerms, idfValues, docsCount):
    termFreq = defaultdict(int)
    # Count the frequency of each term in the query
    for term in queryTerms:
        termFreq[term] += 1
    # Calculate the query vector with IDF weighting
    queryVec = {}
    for term, freq in termFreq.items():
        # Use the IDF value if the term is in the idfValues, otherwise assume 0
        termIDF = idfValues.get(term, math.log(docsCount))  # Default IDF if term not found should not be 0
        queryVec[term] = freq * termIDF
    
    return queryVec

# Function to calculate the cosine similarity between two vectors.
def cosineSim(vecA, vecB):
    # Calculate the dot product
    dotProd = sum(vecA.get(term, 0) * vecB.get(term, 0) for term in set(vecA.keys()).union(vecB.keys()))
    # Calculate the magnitude of the vectors
    magA = math.sqrt(sum(value ** 2 for value in vecA.values()))
    magB = math.sqrt(sum(value ** 2 for value in vecB.values()))
    # Avoid division by zero
    if magA == 0 or magB == 0:
        return 0
    # Calculate cosine similarity
    cosineSimVal = dotProd / (magA * magB)
    
    return cosineSimVal

# Function to retrieve and rank documents based on cosine similarity.
def retrieveAndRank(query, invertedIndex, IDF, lengthsOfDocs, numDocs):
    # Build the query vector
    queryVec = buildQueryVector(query, IDF, numDocs)
    
    # Initialize dictionary to store similarity scores
    docScores = defaultdict(float)
    
    # Calculate scores for each document containing terms from the query
    for term in queryVec:
        if term in invertedIndex:
            for docID, termFreq in invertedIndex[term].items():
                docVec = {term: termFreq * IDF.get(term, 0)}
                # Calculate cosine similarity for each document
                simScore = cosineSim(docVec, queryVec)
                # Add to total score for the document
                docScores[docID] += simScore
    
    # Normalize the scores
    for docID, score in docScores.items():
        length = lengthsOfDocs.get(docID, 1)
        docScores[docID] = score / length  # Avoid division by zero by defaulting length to 1
    
    return docScores