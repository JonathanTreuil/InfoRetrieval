import os
import preprocessing
import indexing
import retrievalandranking
import nltk

documents = {}

def extractSubDocuments(docContent):
    documents = {}

    subDocs = docContent.split('<DOC>')[1:]

    for subDoc in subDocs:
        # Extract document name from <DOCNO> tag
        docNoStart = subDoc.find('<DOCNO>') + len('<DOCNO>')
        docNoEnd = subDoc.find('</DOCNO>', docNoStart)
        docName = subDoc[docNoStart:docNoEnd].strip()

        # Extract text from <TEXT> tag
        textStart = subDoc.find('<TEXT>') + len('<TEXT>')
        textEnd = subDoc.find('</TEXT>', textStart)
        textContent = subDoc[textStart:textEnd].strip()

        # Store in the nested dictionary
        documents[docName] = textContent

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

# print(queries)

# Testing if the document reader seperates each subDoc correctly
# print(documents["AP880212-0001"][:10])
# print(documents["AP881120-0001"][:10])

queryDictionary = preprocessing.preprocess(queries)
# print(queryDictionary)

numDocs = len(documents)
# print(numDocs)

tokenDictionary = preprocessing.preprocess(documents)
# print(tokenDictionary["AP880212-0001"])
# print(tokenDictionary)

invertedIndex = indexing.createInvertedIndex(tokenDictionary)
print(invertedIndex)
print(len(invertedIndex))

idfScores = retrievalandranking.computeIDF(invertedIndex, numDocs)
# print(idfScores)
# print(len(idfScores))

tfidfScores = retrievalandranking.computeTFIDF(invertedIndex, idfScores, numDocs)
# print(tfidfScores["AP880212-0045"])
# print(len(tfidfScores))

documentLengths = retrievalandranking.computeDocumentLengths(tfidfScores)
# print(documentLengths)

# CHOOSE WHAT QUERY YOU WISH TO COMPARE TO THE DOCUMENTS HERE
# Currently running a for loop for all 50 queries
results = []

runName = "FinalRun"

for queryID, query in queryDictionary.items():
    # print(queryDictionary[queryID])
    documentScores = retrievalandranking.retrieveAndRank(queryDictionary[queryID], invertedIndex, idfScores, documentLengths, numDocs)
    rankedDocuments = sorted(documentScores.items(), key=lambda x: x[1], reverse=True)
    for rank, (docID, score) in enumerate(rankedDocuments[:1000], start=1):
        results.append(f"{queryID} Q0 {docID} {rank} {score:.4f} {runName}")
    # Save results to a file named "Results"
    with open('Results', 'w') as file:
        file.write('\n'.join(results))
