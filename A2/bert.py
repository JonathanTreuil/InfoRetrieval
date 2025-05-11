import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
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

documentsRaw = {}
for file in os.listdir('coll/'):
    filePath = 'coll/' + file
    if os.path.isfile(filePath):
        with open(filePath, 'r') as file:
            documentsRaw.update(extractSubDocuments(file.read()))

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

queriesRaw = {}
filePath = "topics1-50.txt"
if os.path.isfile(filePath):
    with open(filePath, 'r') as file:
        queriesRaw.update(extractQueries(file.read()))

with open('stopwords.txt', 'r') as file:
    stopwords = set(file.read().splitlines())

def preprocess(text, stopwords=None):
    tokens = text.lower().split()
    if stopwords:
        tokens = [token for token in tokens if token not in stopwords]
    return " ".join(tokens)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():  
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

doc_embeddings = {}
for docName, docContent in documentsRaw.items():
    processed_text = preprocess(docContent, stopwords=stopwords)
    doc_embeddings[docName] = get_bert_embedding(processed_text)

query_embeddings = {}
for queryNum, queryTitle in queriesRaw.items():
    processed_query = preprocess(queryTitle, stopwords=stopwords)
    query_embeddings[queryNum] = get_bert_embedding(processed_query)

output_file_name = "bert_results.txt"

with open(output_file_name, 'w') as output_file:
    for queryNum, queryEmbedding in query_embeddings.items():
        similarities = {}
        for docName, docEmbedding in doc_embeddings.items():
            similarity = cosine_similarity([queryEmbedding], [docEmbedding])[0][0]
            similarities[docName] = similarity

        ranked_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (docName, score) in enumerate(ranked_docs, start=1):
            line = f"{queryNum} Q0 {docName} {rank} {score:.4f} FinalRun\n"
            output_file.write(line)
            
            if rank == 1000:
                break


