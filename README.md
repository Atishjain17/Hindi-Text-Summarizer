## M-HITS: Hindi Text Summarizer 

* Text Summarization is a process of condensing a given article into a shorter version preserving its content and overall semantics.    There is an abundance of data available on the internet which provides a lot of information and creates a situation called ‘infobesity’.
Manual summarization of articles has also become a cumbersome task. The goal of TextSummarization is to give a comprehensive and concise gist of a document, saving time and effort.
* A summary can be employed using an abstractive way that attempts to develop an understanding of the main concepts in an article and then express those concepts in natural language or, in an extractive way by selecting key text segments from the article based on statistical analysis of individual or mixed features.
* We propose an extractive text summarizer that extracts main text fragments based on statistical analysis of features (cue word, bigrams, etc.). Although, Hindi text summarization has been done before, our model incorporates supervised learning technique along with graph based similarity search technique to alleviate the summarization problem. Our motivation in this paper is to evaluate the performance of our learning model on this task by extracting deep statistical characteristics of the content whereas the extant work uses classical machine learning algorithms like Support Vector Machine or Neural Networks. This enables summarization without the requirement of deep understanding of the article.

**Installation Guidelines**
 
 * [Python Official Website](https://www.python.org/)
 * [Anaconda](https://www.continuum.io/downloads)
 
 **Data**
 * We have scraped the news articles using [beautiful soup](https://pypi.python.org/pypi/beautifulsoup4) library in python from this 	[website](http://www.sampadkiya.com/) 

| Features | Values |
| --- | --- |
| Number Of Articles|      4,316   |
|   Sentence Count  |    1,52,270  |
|     Word Count    |    30,88,571 |

* We have generated the summary of the above Hindi articles using this [tool](https://bigdatasummarizer.com/summarizer/online/advanced.jsp?ui.lang=en)

| Features | Values |
| --- | --- |
| Number Of Articles |     4,316   |
|   Sentence Count   |    63,915   |
|    Word Count      |   17,18,785 |


**Features**

* Topic Feature
* Sentence Position Feature
* Proper Noun Feature
* Cue Word Feature
* Bigram Feature
* Unknown Word Feature
* TF-IDF Feature

**Testing Algorithms**

* Gradient Boost
* Random Forest
* AdaBoost
* SVM
* K-Nearest Neighbor
* Extremely Randomized Trees
* Logistic Regression

**Testing Summaries**

* Bleu Score
* Manual Testing


**Team**  
* [Akshat Mukesh Shah](https://github.com/akshat1710/)
* [Atish Mukesh Jain](https://github.com/atishjain17/)
* [Harsh Rakesh Shah](https://github.com/harshshah1306/)   
* [Sahil Murad Modak](https://github.com/sahilmodak1/)
