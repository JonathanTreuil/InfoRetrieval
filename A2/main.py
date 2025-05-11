import os
import preprocessing
from rank_bm25 import BM25Okapi

documents = {}

def extractSubDocuments(docContent):
    documents = {}
    
    subDocs = docContent.split('<DOC>')[1:]

    for subDoc in subDocs:
        docNoStart = subDoc.find('<DOCNO>') + len('<DOCNO>')
        docNoEnd = subDoc.find('</DOCNO>', docNoStart)
        docName = subDoc[docNoStart:docNoEnd].strip()

        combinedText = ""

        textStart = 0
        while subDoc.find('<TEXT>', textStart) != -1:
            textStart = subDoc.find('<TEXT>', textStart) + len('<TEXT>')
            textEnd = subDoc.find('</TEXT>', textStart)
            combinedText += subDoc[textStart:textEnd].strip() + " "
            textStart = textEnd

        documents[docName] = combinedText

    return documents

queries = {}

def extractQueries(queryContent):
    queries = {}
    # Split the content by <top> tag to separate each query
    subQueries = queryContent.split('<top>')[1:]

    for subQuery in subQueries:
        # Extract query number from <num> tag
        numStart = subQuery.find('<num>') + len('<num>')
        numEnd = subQuery.find('\n', numStart)
        queryNum = subQuery[numStart:numEnd].strip()

        # Extract title from <title> tag
        titleStart = subQuery.find('<title>') + len('<title>')
        titleEnd = subQuery.find('\n', titleStart)
        title = subQuery[titleStart:titleEnd].strip()

        # Store in the dictionary using query number as key
        queries[queryNum] = title

    return queries

# Read each of the documents from the folder that contains all of them and store it to a dictionary.
for file in os.listdir('coll/'):
    filePath = 'coll/' + file
    if os.path.isfile(filePath):
        with open(filePath, 'r') as file:
            documents.update(extractSubDocuments(file.read()))

# Read each of the queries from the topics1-50.txt file and store them seperately inside a dictionary.
filePath = "topics1-50.txt"
if os.path.isfile(filePath):
    with open(filePath, 'r') as file:
        queries.update(extractQueries(file.read()))

queryTokens, queryIDs = preprocessing.preprocess(queries)
documentTokens, docIDs = preprocessing.preprocess(documents)

bm25 = BM25Okapi(documentTokens)

with open('ranked_results_all_queries.txt', 'w') as output_file:
    for queryID, tokens in zip(queryIDs, queryTokens):
        query_scores = bm25.get_scores(tokens)

        scored_docs = zip(docIDs, query_scores)
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        for rank, (docID, score) in enumerate(sorted_docs[:1000], 1):
            line = f"{queryID} Q0 {docID} {rank} {score:.4f} FinalRun\n"
            output_file.write(line)
