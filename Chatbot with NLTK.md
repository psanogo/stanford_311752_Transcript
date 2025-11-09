## $ \color{blue}{\text {Welcome to AI: Critical Principles & Strategy!} } $

#### $\color{purple}{\text{This course is a subset of a}}$ [3 credit graduate level course on Artificial Intelligence Strategy at Rutgers University](https://bloustein.rutgers.edu/graduate/public-informatics/mpi/)

Connect to Faculty: [@ Jim Samuel](https://twitter.com/jimsamuel/)  ----  https://twitter.com/jimsamuel/

---

[Please see the copyright statement below at the end of the notebook.](#ethics)

## $ \color{purple}{\text {Building a Simple Chatbot from Scratch in Python (using NLTK)} } $




We have created a rule-based chatbot using the NLTK library in Python. This chatbot is very basic and has limited cognitive abilities, but it is still a good starting point for learning about natural language processing and chatbots.



## NLP

NLP is a way for computers to analyze, understand, and derive meaning from human language in a smart and useful way. By utilizing NLP, developers can organize and structure knowledge to perform tasks such as automatic summarization, translation, named entity recognition, relationship extraction, sentiment analysis, speech recognition, and topic segmentation.

## Import necessary libraries


```python
import random
import string # to process standard python strings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
```

## Downloading and installing NLTK
NLTK (Natural Language Toolkit) is a library in Python that provides tools for natural language processing (NLP). It is a powerful library that can be used for tasks such as tokenization, stemming, and lemmatization. NLTK also includes a wide range of text processing libraries, such as those for creating frequency distributions, concordances, and collocations. Additionally, NLTK has a large corpus of sample texts, which can be used for training and testing language models. Overall, NLTK is a comprehensive library that can be used for a wide range of NLP tasks and is a popular choice among researchers and developers working in the field of NLP.

For platform-specific instructions, read [here](https://www.nltk.org/install.html)




```python
!pip install nltk
```

    Requirement already satisfied: nltk in /opt/conda/lib/python3.7/site-packages (3.8.1)
    Requirement already satisfied: regex>=2021.8.3 in /opt/conda/lib/python3.7/site-packages (from nltk) (2022.10.31)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.7/site-packages (from nltk) (4.45.0)
    Requirement already satisfied: click in /opt/conda/lib/python3.7/site-packages (from nltk) (7.1.2)
    Requirement already satisfied: joblib in /opt/conda/lib/python3.7/site-packages (from nltk) (0.14.1)
    [33mWARNING: You are using pip version 21.3.1; however, version 24.0 is available.
    You should consider upgrading via the '/opt/conda/bin/python3 -m pip install --upgrade pip' command.[0m


### Installing NLTK Packages





```python
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)
# for downloading packages
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only
```




    True



## Reading in the corpus

For our example,we will be using the Wikipedia page for Artificial intelligence and this [article](https://scholars.org/contribution/call-proactive-policies-informatics-and
) as our corpus. Copy the contents from the page and place it in a text file named ‘chatbot.txt’. However, you can use any corpus of your choice.


```python
f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()
raw = raw.lower()# converts to lowercase
```


The main issue with text data is that it is all in text format (strings). However, the Machine learning algorithms need some sort of numerical feature vector in order to perform the task. So before we start with any NLP project we need to pre-process it to make it ideal for working. Basic text pre-processing includes:

* Converting the entire text into **uppercase** or **lowercase**, so that the algorithm does not treat the same words in different cases as different

* **Tokenization**: Tokenization is just the term used to describe the process of converting the normal text strings into a list of tokens i.e words that we actually want. Sentence tokenizer can be used to find the list of sentences and Word tokenizer can be used to find the list of words in strings.

_The NLTK data package includes a pre-trained Punkt tokenizer for English._

* Removing **Noise** i.e everything that isn’t in a standard number or letter.
* Removing the **Stop words**. Sometimes, some extremely common words which would appear to be of little value in helping select documents matching a user need are excluded from the vocabulary entirely. These words are called stop words
* **Stemming**: Stemming is the process of reducing inflected (or sometimes derived) words to their stem, base or root form — generally a written word form. Example if we were to stem the following words: “Stems”, “Stemming”, “Stemmed”, “and Stemtization”, the result would be a single word “stem”.
* **Lemmatization**: A slight variant of stemming is lemmatization. The major difference between these is, that, stemming can often create non-existent words, whereas lemmas are actual words. So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary, but you can look up a lemma. Examples of Lemmatization are that “run” is a base form for words like “running” or “ran” or that the word “better” and “good” are in the same lemma so they are considered the same.



## Tokenisation


```python
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words
```

## Preprocessing

We shall now define a function called LemTokens which will take as input the tokens and return normalized tokens.


```python
lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)# remove the Punctuation from the sentences 

'''
The below function "LemNormalize(text)" takes in a string, normalize it to lowercase, 
remove the punctuation using above dict, tokenize the string and lemmatize the tokenized 
string using the above defined LemTokens() function and returning the lemmatized tokens.
'''

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
```

## Keyword matching

Next, we shall define a function for a greeting by the bot i.e if a user’s input is a greeting, the bot shall return a greeting response.We use a simple keyword matching for greetings. We will utilize the same concept here.


```python
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me","How can I help you today?","Good to see you!","Nice to meet you!","What's on your mind?","I can assist you with any question you might have regrading Artificial intelligence."]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
```

## Generating Response

### Bag of Words
After the initial preprocessing phase, we need to transform text into a meaningful vector (or array) of numbers. Bag of Words (BoW) is a method for representing text data in natural language processing. It is a way of extracting features from text data, and representing the text as a numerical feature vector. The basic idea behind the BoW model is to take a piece of text and represent it as a "bag" (or unordered set) of its words, disregarding grammar and even word order but keeping track of the occurrence of each word.

In the bag of words model, a text (such as a sentence or a document) is represented as a numerical vector, where each dimension of the vector represents a specific word from the vocabulary, and the value in each dimension is the frequency count of that word in the text..

For example, if our dictionary contains the words {"the", "cat", "sat", "on", "mat", "dog", "rug"}, and we want to vectorize the text “"The cat sat on the mat."”, we would have the following vector: (1, 1, 1, 1, 1, 0, 0).


### TF-IDF Approach

**Term Frequency: is a scoring of the frequency of the word in the current document.**

```
TF = (Number of times term t appears in a document)/(Number of terms in the document)
```

**Inverse Document Frequency: is a scoring of how rare the word is across documents.**

```
IDF = 1+log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.
```
### Cosine Similarity
How similar are two words?

```
Cosine Similarity (d1, d2) =  Dot product(d1, d2) / ||d1|| * ||d2||
```
where d1,d2 are two non zero vectors.



To generate a response from our bot for input questions, the concept of document similarity will be used. We define a function response which searches the user’s utterance for one or more known keywords and returns one of several possible responses. If it doesn’t find the input matching any of the keywords, it returns a response:” I am sorry! I don’t understand you”


```python
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')#TFIdf model and removing stop words 
    tfidf = TfidfVec.fit_transform(sent_tokens)#Fit the Tfidf model on the sent tokens 
    vals = cosine_similarity(tfidf[-1], tfidf) #Finding the cosine similarity of the current model with all other sentences
    idx=vals.argsort()[0][-2]#Sorting the index of last of index of the sencond last result because the last one will be user response 
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]# to check if tfidf value is zero 
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


```

Finally, we will feed the lines that we want our bot to say while starting and ending a conversation depending upon user’s input.


```python
flag=True
print("Chatterbot: My name is Chatterbot. I will answer your queries about AI. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Chatterbot: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("Chatterbot: "+greeting(user_response))
            else:
                print("Chatterbot: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Chatterbot: Bye! take care..")
```

    Chatterbot: My name is Chatterbot. I will answer your queries about AI. If you want to exit, type Bye!


Example Questions that can be asked 


1.   Who is Father of AI?
2.   Sub sections of artificial intelligence ?
3.   Risk of AI?
4.   What is the defination of artificial intelligence ?





Refrence-> https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e

### Notes & More resources

- This notebook is collection of foundational instructions from multiple unlisted sources
- If you click on "Help" in the toolbar, there is a list of references for common Python tools, e.g. numpy, pandas.
- [IPython website](https://ipython.org/) |||||||
- [Markdown basics](https://daringfireball.net/projects/markdown/) |||||||
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/en/stable/index.html) |||||||
- [Real Python Jupyter Tutorial](https://realpython.com/jupyter-notebook-introduction/) |||||||
- [Dataquest Jupyter Notebook Tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/) |||||||
- [Stack Overflow](https://stackoverflow.com/) |||||||

<div class="alert alert-info">
<a class="anchor" id="ethics"></a>

## Ethics and Copyright Statement: All rights reserved.

We support and continue to contribute to open-source code and resources BUT the contents of this for-credit and graded course are protected for ethical reasons and course integrity. Beyond use within this course, none of the course materials developed for this course may be copied, reproduced, re-published, uploaded, posted, transmitted, or distributed in any way without written authorization from the concerned faculty /authors.

- <b> Therefore, for the benefit of future students and course integrity, PLEASE DO NOT SHARE OR DISSMEINATE </b> any of these materials outside of this class so that the learning experience of future students remains unique and valuable.
    - <b> Please do not post these materials to GitHub or to any other platform or website. </b>
    - When / If using Google Colab, pls. ensure that the file is not set up for public access (default expected setting is private).
    </div>


```python

```
