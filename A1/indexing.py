# Creates the inverted index based off the token dictionary provided by the preprocessing step.
def createInvertedIndex(tokenDictionary):
    wordFind = {}

    for docNo, allWords in tokenDictionary.items():
        for valueWord in set(allWords):
            wordCount = allWords.count(valueWord)
            wordFind.setdefault(valueWord, {}).setdefault(docNo, 0)
            wordFind[valueWord][docNo] += wordCount

    finalDictionary = {indivWord: occurrences for indivWord, occurrences in wordFind.items()}
    return finalDictionary
