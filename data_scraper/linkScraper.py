from bs4 import BeautifulSoup
from urllib.request import urlopen
from requests import HTTPError

'''
:global: pages
'''
global pages
pages = set()
def getdegree(url):
    '''
    :global: none
    :local: html, bsObj, tags
    :return: none
    : Used to add the pages following the Url from url
    '''
    try:
        html = urlopen(url)
    except HTTPError as e:
        return None
    bsObj = BeautifulSoup(html.read(), "lxml")
    for link in bsObj.find_all('article'):
        tags = link.find_all('a')
        for links in tags:
            pages.add(links.get('href'))

for i in range(451):
    url = "http://www.sampadkiya.com/news/editorials/category/hindi-editorials/page/" + str(i) + "/"
    degree = getdegree(url)
file = open("NLPData/articleLinks.txt","w+",encoding = "utf-8")
for items in pages:
    file.write(items)
    file.write("\n")