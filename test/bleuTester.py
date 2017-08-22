import json
import sys
from collections import OrderedDict
import math
from operator import itemgetter
import nltk
from nltk import sent_tokenize, word_tokenize, bigrams
from nltk.corpus import indian
from nltk.tag import tnt
from collections import Counter
import copy
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def readData(fileName):
    '''
    :params: fileName
    :global: stringLines
    :local: dataFileOpen
    :return: none
    :Used to read data from the file
    '''
    global stringLines
    dataFileOpen = open(fileName, 'r', encoding = "utf-8")
    stringLines = dataFileOpen.read()
    dataFileOpen.close()

def loadFeatureVector():
    '''
    :global:featureVector,tfIdfWeight,sentPosWeight,bigramWeight,unkWordWeight,cueWordWeight,topicWeight, properNounWeight
    :local: file, featureVector, currentFeature, sentPosWeight, tfIdfWeight, bigramWeight, unkWordWeight, cueWordWeight, topicWeight, properNounWeight
    :return: none
    :Used to load the features from json file
    '''
    global featureVector,tfIdfWeight,sentPosWeight,bigramWeight,unkWordWeight,cueWordWeight,topicWeight, properNounWeight
    file = open('NLPData/featureWeightVector.json', 'r', encoding = "utf-8")
    featureVector = json.load(file)
    file.close()
    for items in featureVector:
        currentFeature = items[1]
        if(currentFeature = = "Tf-Idf"):
            tfIdfWeight = items[0]
        elif(currentFeature = = "Sentence Position Feature"):
            sentPosWeight = items[0]
        elif(currentFeature = = "Bigram Feature"):
            bigramWeight = items[0]
        elif(currentFeature = = "Unknown Word Feature"):
            unkWordWeight = items[0]
        elif(currentFeature = = "Cue Word Feature"):
            cueWordWeight = items[0]
        elif(currentFeature = = "Topic Feature"):
            topicWeight = items[0]
        elif (currentFeature = = "ProperNoun Feature"):
            properNounWeight = items[0]

def formDataDict():
    '''
    :global:stringLines, titleData, wordList, sentenceList, originalSentenceList, bigramCountList
    :local: originalSentenceList, sentenceList2 
    :return: none
    :function calls: removeStopWords(), generateBigrams()
    :Used to parse the data, remove punctuations for feature extraction.
    '''
    global stringLines, titleData, wordList, sentenceList, originalSentenceList, bigramCountList, sentenceList2
    stringLines = stringLines.replace(".", "")
    stringLines = stringLines.replace("!", " . ")
    stringLines = stringLines.replace("\"", "")
    stringLines = stringLines.replace("।", " . ")
    stringLines = stringLines.replace("?", " . ")
    stringLines = stringLines.replace("\'", "")
    stringLines = stringLines.replace("\n", "")
    stringLines = stringLines.replace("\ufeff", "")
    stringLines = stringLines.replace(",", "")
    stringLines = stringLines.replace("’", "")
    originalFile = copy.deepcopy(stringLines)
    originalSentenceList = [sentenceList.strip() for sentenceList in originalFile.split(' . ')]
    del originalSentenceList[-1]
    removeStopWord()
    generateBigrams()
    sentenceList = [sentenceList.strip() for sentenceList in stringLines.split(' . ')]
    titleData = sentenceList[0]
    del sentenceList[0]
    del sentenceList[-1]
    sentenceList2 = copy.deepcopy(sentenceList)
    bigramCountList = []
    for sentence in (range(len(sentenceList))):
        sentenceList[sentence] = word_tokenize(sentenceList[sentence])
        bigramCountList.append(countBigrams(sentenceList[sentence]))
    wordList = list(set(word_tokenize(stringLines)))

def generateBigrams():
    '''
    :global: bigrams, bigramsDict, bigramsWordsList
    :local: wordTokenizedList, bigrams, cfd
    :return: none
    :Used to calculate the bigrams, as one of the feature for summarization
    '''
    global bigrams, bigramsDict, bigramsWordsList
    bigramsDict = {}
    bigramsWordsList = []
    wordTokenizedList = word_tokenize(stringLines)
    bigrams = list(nltk.bigrams(wordTokenizedList))
    cfd = nltk.ConditionalFreqDist(bigrams)
    for inneritems in cfd.items():
        for items in inneritems[1].items():
            if (items[1] > 2 and (inneritems[0] ! = '.' and items[0] ! = '.')):
                if items[1] not in bigramsDict:
                    bigramsDict[items[1]] = []
                    bigramsDict[items[1]].append([inneritems[0], items[0]])
                else:
                    bigramsDict[items[1]].append([inneritems[0], items[0]])
                bigramsWordsList.append([inneritems[0],items[0]])

def removeStopWord():
    '''
    :global: stringLines
    :local: stopWordsFile, stopWordsFileRead
    :return: none
    :Used to remove the stopwords from 'stopwords.txt'
    '''
    global stringLines
    stopWordsFile = open("NLPData/stopwords.txt", 'r', encoding = "utf-8")
    stopWordsFileRead = stopWordsFile.readlines()
    for words in stopWordsFileRead:
        words = words.strip()
        stringLines = stringLines.replace(words, "")

def getSuffixes():
    '''
    :global: suffixes
    :local: file
    :return: none
    :Used to get the suffixes from 'stemmer.json'
    '''
    global suffixes
    file = open('NLPData/stemmer.json', 'r', encoding = "utf-8")
    suffixes = json.load(file)
    file.close()

def generateStemWords(word):
    '''
    :params: word
    :global:suffixes
    :local:
    :return: word
    :Used to transform the words to its root form
    '''
    global suffixes
    for key in suffixes:
        if len(word) > int(key) + 1:
            for suf in suffixes[key]:
                if word.endswith(suf):
                    return word[:-int(key)]
        return word

def partsOfSpeechTagger(localWordList, checkString):
    '''
    :params: localWordList, checkString
    :global: partsOfSpeechTaggedData, partsOfSpeechTaggedTitleData
    :local: train_data
    :return: none
    :Used to use pos tagger to tag the words, and remove the unnecessary tagged words
    '''
    global partsOfSpeechTaggedData, partsOfSpeechTaggedTitleData
    train_data = indian.tagged_sents('hindi.pos')
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    if checkString = = "textData":
        partsOfSpeechTaggedData = (tnt_pos_tagger.tag(localWordList))
    elif checkString = = "titleData":
        partsOfSpeechTaggedTitleData = (tnt_pos_tagger.tag(localWordList))

def loadWordNet():
    '''
    :global: word_dict
    :local: file
    :return: none
    :Used to load the hindi wordnet from hindiWordNet.json
    '''
    global word_dict
    file = open('NLPData/hindiWordNet.json', 'r', encoding = "utf-8")
    word_dict = json.load(file)
    file.close()

def removeStopWords(localpartsOfSpeechTaggedData, flag):
    '''
    :params: localpartsOfSpeechTaggedData, flag
    :global:removedTaggedData, removedTaggedTitleData
    :local: none
    :return: none
    :Used to removed the stop words based on the tags, tagged using the pos tagger
    '''
    global removedTaggedData, removedTaggedTitleData
    removedTaggedData = []
    removedTaggedTitleData = []
    if flag:
        for words in localpartsOfSpeechTaggedData:
            if not (words[1] = = 'VAUX' or words[1] = = 'SYM' or words[1] = = 'VFM' or words[1] = = 'CC' or words[1] = = 'PRP' or
                            words[1] = = 'PUNC' or words[1] = = 'QF' or words[1] = = 'RB' or words[1] = = 'QW' or words[
                1] = = 'RP' or words[1] = = 'PREP'):
                removedTaggedData.append(words[0])
    else:
        for words in localpartsOfSpeechTaggedData:
            if (words[1] = = 'NN' or words[1] = = 'NNP' or words[1] = = 'Unk'):
                removedTaggedTitleData.append(words[0])

def stemmingForData(sentenceList):
    '''
    :params: sentenceList
    :global: none
    :local:strinTemp
    :return:none
    :Used to stem the data(words) into its stem form
    '''
    for sentence in range(len(sentenceList)):
        stringTemp = []

        for words in sentenceList[sentence]:
            if words in removedTaggedData:
                if words in word_dict:
                    temp_word = word_dict[words]
                else:
                    temp_word = generateStemWords(words)
                    if temp_word in word_dict:
                        temp_word = word_dict[temp_word]
                stringTemp.append(temp_word)
        sentenceList[sentence] = stringTemp

def stemmingForTitle():
    '''
    :global: titleList
    :local: stringTemp
    :return: none
    :Used to stem the data(words) in the title
    '''
    global titleList
    titleList = titleData.split(" ")
    stringTemp = []
    for words in removedTaggedTitleData:
        if words in removedTaggedData:
            if words in word_dict:
                temp_word = word_dict[words]
            else:
                temp_word = generateStemWords(words)
                if temp_word in word_dict:
                    temp_word = word_dict[temp_word]
            stringTemp.append(temp_word)
        titleList = stringTemp

def properNounFeature(localPartsOfSpeechTaggedData):
    '''
    :params:localPartsOfSpeechTaggedData
    :global:properNounList, unknownWordList
    :local: none
    :return: none
    :Used to identify and weight the sentences based on the proper noun feature using pos tagger
    '''
    global properNounList, unknownWordList
    properNounList = []
    unknownWordList = []
    for items in localPartsOfSpeechTaggedData:
        if (items[1] = = "NNP"):
            properNounList.append(items[0])
        if (items[1] = = "Unk"):
            unknownWordList.append(items[0])

def generateCueWordList(titleList):
    '''
    :params: titleList
    :global: CueWordList
    :local: none
    :return: none
    :Used to generate the cue words(synonyms) for the title data
    '''
    global cueWordList
    for items in titleList:
        if items in word_dict:
            cueWordList.append(word_dict[items])

def calculateIdf():
    '''
    :global: idf
    :local: allWords
    :return: none
    :Used to calculate the tf-idf score for the sentences
    '''
    global idf
    allWords = []
    for sentence in range(len(sentenceList)):
        allWords.extend(list(set(sentenceList[sentence])))
    idf = Counter(allWords)
    for items in idf:
        idf[items] = math.log(len(sentenceList) / idf[items])

def countBigrams(sentences):
    '''
    :params: sentences
    :global: none
    :local: sentenceBigrams
    :return: count
    :Used to count the bigrams in the sentences and weighting the sentences accordingly
    '''
    sentenceBigrams = list(nltk.bigrams(sentences))
    count = 0
    for items in sentenceBigrams:
        if list(items) in bigramsWordsList:
            count += 1
    return count

def calculateFeatures():
    '''
    :global: featureProbablity
    :local: i, j, countTopicFeature, countCueFeature, countUnknownWordFeature, tfNumerator
    :return: none
    : Used to store the calculated features in the dictionary 'featureProbablity'
    '''
    global featureProbablity
    featureProbablity = {}
    for i in range(1,len(originalSentenceList)):
        featureProbablity[originalSentenceList[i]] = {}
    i = 1
    j = len(originalSentenceList)-1
    tfIdf = [0] * len(sentenceList)
    for sentences in sentenceList:
        countTopicFeature = 0
        countCueFeature = 0
        countProperWordFeature = 0
        countUnknownWordFeature = 0
        for words in sentences:
            if words in properNounList:
                countProperWordFeature += 1
            elif words in unknownWordList:
                countUnknownWordFeature += 1
            if words in titleList:
                countTopicFeature += 1
                countCueFeature += 1
            if words in cueWordList:
                countCueFeature += 1
        if (len(sentences) = = 0):
            sentences.append(" ")
        featureProbablity[originalSentenceList[i]]["topicFeature"] = countTopicFeature / len(sentences)
        featureProbablity[originalSentenceList[i]]["properWordFeature"] = countProperWordFeature / len(sentences)
        featureProbablity[originalSentenceList[i]]["unknownWordFeature"] = countUnknownWordFeature / len(sentences)
        featureProbablity[originalSentenceList[i]]["cueWordFeature"] = countCueFeature / len(sentences)
        featureProbablity[originalSentenceList[i]]["bigrams"] = bigramCountList[i-1] / len(sentences)

        tfNumerator = {}
        tfNumerator = Counter(sentences)
        for words in tfNumerator:
            tfNumerator[words] = (tfNumerator[words] / len(sentences) * idf[words])
            tfIdf[i - 1] += tfNumerator[words]
        featureProbablity[originalSentenceList[i]]["tfIdf"] = tfIdf[i - 1] / len(sentences)

        if (i < = j):
            featureProbablity[originalSentenceList[i]]["sentencePositionFeature"] = (j - (len(sentenceList) / 2)) / ((len(sentenceList) / 2))
            featureProbablity[originalSentenceList[j]]["sentencePositionFeature"] = (j - (len(sentenceList) / 2)) / ((len(sentenceList) / 2))
        i += 1
        j -= 1

def sentenceRank():
    '''
    :global: rankSentences
    :local: i
    :return: none
    :Used to rank the sentences, based on the weights obtained from various algorithms
    '''
    global rankSentences
    rankSentences = {}
    i = 1
    for sentences in sentenceList:
        sentenceWeight = 0
        sentenceWeight += featureProbablity[originalSentenceList[i]]["topicFeature"] * topicWeight
        sentenceWeight += featureProbablity[originalSentenceList[i]]["properWordFeature"] *  properNounWeight
        sentenceWeight += featureProbablity[originalSentenceList[i]]["unknownWordFeature"] * unkWordWeight
        sentenceWeight += featureProbablity[originalSentenceList[i]]["cueWordFeature"] * cueWordWeight
        sentenceWeight += featureProbablity[originalSentenceList[i]]["bigrams"] * bigramWeight
        sentenceWeight += featureProbablity[originalSentenceList[i]]["tfIdf"] * tfIdfWeight
        sentenceWeight += featureProbablity[originalSentenceList[i]]["sentencePositionFeature"] * sentPosWeight
        rankSentences[i] = sentenceWeight
        i += 1

loadWordNet()
getSuffixes()
loadFeatureVector()
readData("NLPData/updated_articles/"+str(201)+".txt") #File to be tested
formDataDict()
partsOfSpeechTagger(wordList, "textData")
removeStopWords(partsOfSpeechTaggedData, True) 
partsOfSpeechTagger(titleData.split(" "), "titleData")
properNounFeature(partsOfSpeechTaggedData)
removeStopWords(partsOfSpeechTaggedTitleData, False)
stemmingForData(sentenceList)
stemmingForTitle()
generateCueWordList(titleList)
calculateIdf()
calculateFeatures()
sentenceRank()
sorted_x = OrderedDict(sorted(rankSentences.items(), key = itemgetter(1)))
answer = []
for key in sorted_x:
    answer.insert(0,key)
summary = []
for i in range(math.ceil(len(answer)*0.6)):
    summary.append(answer[i])
summary = sorted(summary)
summaryText = []
for i in summary:
   summaryText.append(sentenceList2[i-1])
print(summaryText)

def graphSimilarity(summaryText):
    '''
    :params: summaryText
    :global:none
    :local: bow_matrix, normalized, similarity_graph, sil, remove_sent, keep, drop
    :return: drop
    : Used to calculate the graphSimilarity and remove sentences reducing redundancy
    '''
    bow_matrix = CountVectorizer().fit_transform(summaryText)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
    similarity_graph = normalized * normalized.T
    sil = similarity_graph.toarray()
    remove_sent = []
    keep = []
    drop = []
    for i in range(len(sil)):
        for j in range(len(sil[i])):
            if (sil[i][j] > 0.5 and i ! = j and j>i):
                remove_sent.append([i + 1, j + 1])
                if i+1 in keep and j+1 not in keep:
                    drop.append(j+1)
                elif j+1 in keep and i+1 not in keep:
                    drop.append(i+1)
                elif i+1 in drop and j+1 not in drop:
                    keep.append(j+1)
                elif j+1 in drop and i+1 not in drop:
                    keep.append(i+1)
                elif j+1 in drop and i+1 in drop:
                    continue
                elif j+1 in keep and i+1 in keep:
                    continue
                else:
                    keep.append(i+1)
                    drop.append(j+1)
    keep = list(set(keep))
    drop = list(set(drop))
    return(drop)

remove_sent = graphSimilarity(summaryText)
print(remove_sent)
fileOp = open('summaryOutput.txt', 'w', encoding = "utf-8")
for i in summary:
    if i not in drop:
        fileOp.write(originalSentenceList[i]+". ")
fileOp.close()
print(len(answer))
fileOpen = open('summaryOutput.txt','r', encoding = "utf-8")
fileopen = open('NlPData/summary/201_sum.txt', 'r', encoding = "utf-8")
BLEUscore = nltk.translate.bleu_score.sentence_bleu([fileopen.read()], fileOpen.read())
print (BLEUscore)