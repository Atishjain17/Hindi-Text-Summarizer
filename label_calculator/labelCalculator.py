import json
import sys

def assignClasses():
    '''
    :global: allArticlesWithFeatures
    :local: file, summaryStringLines
    :return none
    :Used to assign classes 1 or 0 depending upon the presence of the sentence in the summary.
    '''
    global allArticlesWithFeatures
    file = open('NLPData/allArticlesWithFeatures.json', 'r', encoding = 'utf-8')
    allArticlesWithFeatures = json.load(file)
    for i in range(801):
        file = open("NLPData/summary/" + str(i) + "_sum.txt", "r", encoding = "utf-8")
        summaryStringLines = file.read()
        file.close()
        summaryStringLines = summaryStringLines.replace(".", "")
        summaryStringLines = summaryStringLines.replace("?", " . ")
        summaryStringLines = summaryStringLines.replace("!", " . ")
        summaryStringLines = summaryStringLines.replace("\"", "")
        summaryStringLines = summaryStringLines.replace("।", " . ")
        summaryStringLines = summaryStringLines.replace("?", " . ")
        summaryStringLines = summaryStringLines.replace("\'", "")
        summaryStringLines = summaryStringLines.replace("\n", "")
        summaryStringLines = summaryStringLines.replace("\ufeff", "")
        summaryStringLines = summaryStringLines.replace(",", "")
        summaryStringLines = summaryStringLines.replace("’", "")
        summaryLines = [sentenceList.strip() for sentenceList in summaryStringLines.split(' . ')]
        del summaryLines[-1]
        print("\nFile Name - " + str(i))
        print("Length of Summary File - " + str(len(summaryLines)))
        print("Sentences in Summary File\n")
        for keys in summaryLines:
            print(keys)
        print("\n\n")
        count = 0
        print("Sentences in Text File")
        for keys in allArticlesWithFeatures[i].keys():
            print (keys)
        print ("\n\n")
        for sent in summaryLines:
            if sent in allArticlesWithFeatures[i]:
                allArticlesWithFeatures[i][sent]["class"] = 1
                count += 1
            else:
                print("Line Not matched - ")
                print("\t" + sent)
                sys.exit()
        print("Number of lines in Summary file that matched - " + str(count))

def saveFinalDataset():
    '''
    :global: allArticlesWithFeatures
    :local: file, articleCount
    :return: none
    :Used to the final dataset after the labels are assigned
    '''
    global allArticlesWithFeatures
    file = open('NLPData/finalDataset.csv', 'w+', encoding = 'utf-8')
    file.write("Article Number, Topic Feature, ProperNoun Feature, Unknown Word Feature, Cue Word Feature,  Bigram Feature, Tf-Idf, Sentence Position Feature, Class\n")
    articleCount = -1
    for articles in allArticlesWithFeatures:
        articleCount + =  1
        for sentenceKey in articles:
            file.write(str(articleCount) + ", " + str(articles[sentenceKey]["topicFeature"]) + ", " + str(articles[sentenceKey]["properWordFeature"]) + ", " + 
                str(articles[sentenceKey]["unknownWordFeature"]) + ", " + str(articles[sentenceKey]["cueWordFeature"]) + ", " + str(articles[sentenceKey]["bigrams"]) +
                ", " + str(articles[sentenceKey]["tfIdf"]) + ", " + str(articles[sentenceKey]["sentencePositionFeature"]) + ", " + str(articles[sentenceKey]["class"]) + "\n")
    file.close()

allArticlesWithFeatures = list()
assignClasses()
saveFinalDataset()