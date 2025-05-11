import os
import re
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

relevantDocuments = {}

def extractRelevantDocuments(file):
    relevantDocuments = {}
    subDocuments = file.split('\n')
    for line in subDocuments:
        parts = line.split()
        relevantDocuments[parts[2]] = 1
    return relevantDocuments


filePath = "ranked_results_all_queries.txt"
if os.path.isfile(filePath):
    with open(filePath, 'r') as file:
        relevantDocuments.update(extractRelevantDocuments(file.read()))


def extractSubDocuments(docContent):
    documents = {}
    
    subDocs = docContent.split('<DOC>')[1:]

    for subDoc in subDocs:
        flag = 0
        docNoStart = subDoc.find('<DOCNO>') + len('<DOCNO>')
        docNoEnd = subDoc.find('</DOCNO>', docNoStart)
        docName = subDoc[docNoStart:docNoEnd].strip()
        if docName in relevantDocuments:
            flag = 1

        combinedText = ""

        textStart = 0
        while subDoc.find('<TEXT>', textStart) != -1:
            textStart = subDoc.find('<TEXT>', textStart) + len('<TEXT>')
            textEnd = subDoc.find('</TEXT>', textStart)
            combinedText += subDoc[textStart:textEnd].strip() + " "
            textStart = textEnd

        if flag == 1: documents[docName] = combinedText

    return documents

def extractQueries(queryContent):
    queries = {}
    subQueries = queryContent.split('<top>')[1:]
    for subQuery in subQueries:
        numStart = subQuery.find('<num>') + len('<num>')
        numEnd = subQuery.find('\n', numStart)
        queryNum = int(subQuery[numStart:numEnd].strip())
        titleStart = subQuery.find('<title>') + len('<title>')
        titleEnd = subQuery.find('\n', titleStart)
        title = subQuery[titleStart:titleEnd].strip()
        queries[queryNum] = title
    return queries

def preprocess_text(text, stopwords):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)

def load_USE_model():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    return model

def embed_text(model, text):
    return model([text])[0].numpy()

documentsRaw = {}
for file in os.listdir('coll/'):
    filePath = os.path.join('coll', file)
    if os.path.isfile(filePath):
        with open(filePath, 'r') as file:
            documentsRaw.update(extractSubDocuments(file.read()))

queriesRaw = {}
filePath = "topics1-50.txt"
if os.path.isfile(filePath):
    with open(filePath, 'r') as file:
        queriesRaw.update(extractQueries(file.read()))

with open('stopwords.txt', 'r') as file:
    stopwords = file.read().splitlines()

documentsProcessed = {docName: preprocess_text(content, stopwords) for docName, content in documentsRaw.items()}
queriesProcessed = {queryNum: preprocess_text(query, stopwords) for queryNum, query in queriesRaw.items()}

model = load_USE_model()

document_embeddings = {docName: embed_text(model, content) for docName, content in documentsProcessed.items()}
query_embeddings = {queryNum: embed_text(model, query) for queryNum, query in queriesProcessed.items()}

ranked_results_with_scores = {}
for queryNum, query_embedding in query_embeddings.items():
    similarities = cosine_similarity([query_embedding], list(document_embeddings.values())).flatten()
    doc_scores = [(docName, score) for docName, score in zip(document_embeddings.keys(), similarities)]
    ranked_docs_with_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    ranked_results_with_scores[queryNum] = ranked_docs_with_scores

with open('use_results.txt', 'w') as results_file:
    for queryNum, ranked_docs_with_scores in ranked_results_with_scores.items():
        for rank, (docName, score) in enumerate(ranked_docs_with_scores, start=1):
            line = f"{queryNum} Q0 {docName} {rank} {score:.4f} FinalRun\n"
            results_file.write(line)
            if rank == 1000:
                break


