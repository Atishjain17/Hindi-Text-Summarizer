from bs4 import BeautifulSoup
from urllib.request import urlopen
import re

'''
:global: none
:local: file, fileOpen, tempFile, url, html, bsObj, title, text, ans
:return: none 
:Used for scrapping the articles from the links extracted using linkScraper.py
'''
file = open("NLPData/articleLinksUpdated.txt", "r", encoding = "utf-8")
fileOpen = file.readlines()
file.close()
for i in range(1216,len(fileOpen)):
    tempFile = open("NLPData/articles/" + str(i) + ".txt","w+", encoding = "utf-8")
    url = fileOpen[i]
    html = urlopen(url)
    bsObj = BeautifulSoup(html.read(), "lxml")
    title = bsObj.article.header.h1.string
    tempFile.write(title)
    tempFile.write("\n")
    text = bsObj.find('div', attrs = {"class":"entry-content cf"}).findAll('p')
    ans = str(text)
    ans = re.sub('[a-z<>/]',"",ans)
    ans = ans.replace("[","")
    ans = ans.replace("]","")
    tempFile.write(ans)
    tempFile.write("\n")
    tempFile.close()
