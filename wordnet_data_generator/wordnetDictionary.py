import json

'''
:global: none
:local: file, wordNetFile, wordDictionary, wordThesaurus, hindiWordNet
:return: none
Used to form the hindiWordNet json file, from the 'wordNetData.txt'
'''
file = open("NLPData/wordNetData.txt", "r", encoding = "utf-8")
wordNetFile = file.read()
file.close()
wordNetFile = wordNetFile.split("\n")
del wordNetFile[-1]
wordDictionary = []
for sent in wordNetFile:
    wordDictionary.append(sent.split(" "))
wordThesaurus = []
for i in range(len(wordDictionary)):
    wordThesaurus.append(wordDictionary[i][3].split(":"))
hindiWordNet = {}
for word in wordThesaurus:
    for synonym in range(len(word)):
        hindiWordNet[word[synonym]] = word[0]
file = open("NLPData/hindiWordNet.json", "w", encoding = "utf-8")
json.dump(hindiWordNet, file1, ensure_ascii = False)
file.close()
