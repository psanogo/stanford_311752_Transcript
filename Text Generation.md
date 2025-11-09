## $ \color{blue}{\text {Welcome to AI: Critical Principles & Strategy!} } $

### $\color{purple}{\text{This course is a subset of a}}$ [3 credit graduate level course on Artificial Intelligence Strategy at Rutgers University](https://bloustein.rutgers.edu/graduate/public-informatics/mpi/)

Connect to Faculty: [@ Jim Samuel](https://twitter.com/jimsamuel/)  ----  https://twitter.com/jimsamuel/

---

[Please see the copyright statement below at the end of the notebook.](#ethics)

## $ \color{purple}{\text {Text Generation} } $

In this notebook, we will use Python along with libraries such as NLTK (Natural Language Toolkit) to generate text based on existing textual data. We will start by cleaning the text data and then explore methods to generate sentences from the given data.




```python
!pip install nltk
```

    Requirement already satisfied: nltk in /opt/conda/lib/python3.7/site-packages (3.8.1)
    Requirement already satisfied: regex>=2021.8.3 in /opt/conda/lib/python3.7/site-packages (from nltk) (2022.10.31)
    Requirement already satisfied: click in /opt/conda/lib/python3.7/site-packages (from nltk) (7.1.2)
    Requirement already satisfied: joblib in /opt/conda/lib/python3.7/site-packages (from nltk) (0.14.1)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.7/site-packages (from nltk) (4.45.0)
    [33mWARNING: You are using pip version 21.3.1; however, version 24.0 is available.
    You should consider upgrading via the '/opt/conda/bin/python3 -m pip install --upgrade pip' command.[0m


#### Importing Libraries

The below cell is used to download the 'punkt' tokenizer from the NLTK library. Tokenization is a crucial step in NLP that breaks down the text into smaller units (tokens) for further analysis.


```python
import requests
import nltk
nltk.download('punkt')
import pandas as pd
from numpy.random import choice
```

    [nltk_data] Downloading package punkt to /home/jovyan/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!


#### Reading input file to generate text  you can import your own file and use it.


```python
with open("11-0.txt") as f:
    txt=f.read()
```

#### Cleaning and tokenizing text
The cell is to process the text data. It converts the text to lowercase and tokenizes it using the NLTK tokenizer. The resulting tokens are stored in a Pandas DataFrame.The script then creates a new DataFrame data containing unique tokens from the raw data, essentially removing duplicates.



```python
def clean_data(text):
    text = text.lower()
    data = nltk.word_tokenize(text)
    data = pd.DataFrame(data, columns = ['tokens'])
    #data = pd.DataFrame(list(text), columns = ['tokens'])
    return data

raw_data = clean_data(txt)
data = pd.DataFrame(raw_data['tokens'].unique())
```


```python
txt[1:100]
```




    'The Project Gutenberg eBook of Alice’s Adventures in Wonderland, by Lewis Carroll\n\nThis eBook is fo'




```python
txt[1:5000]
```




    'The Project Gutenberg eBook of Alice’s Adventures in Wonderland, by Lewis Carroll\n\nThis eBook is for the use of anyone anywhere in the United States and\nmost other parts of the world at no cost and with almost no restrictions\nwhatsoever. You may copy it, give it away or re-use it under the terms\nof the Project Gutenberg License included with this eBook or online at\nwww.gutenberg.org. If you are not located in the United States, you\nwill have to check the laws of the country where you are located before\nusing this eBook.\n\nTitle: Alice’s Adventures in Wonderland\n\nAuthor: Lewis Carroll\n\nRelease Date: January, 1991 [eBook #11]\n[Most recently updated: October 12, 2020]\n\nLanguage: English\n\nCharacter set encoding: UTF-8\n\nProduced by: Arthur DiBianca and David Widger\n\n*** START OF THE PROJECT GUTENBERG EBOOK ALICE’S ADVENTURES IN WONDERLAND ***\n\n[Illustration]\n\n\n\n\nAlice’s Adventures in Wonderland\n\nby Lewis Carroll\n\nTHE MILLENNIUM FULCRUM EDITION 3.0\n\nContents\n\n CHAPTER I.     Down the Rabbit-Hole\n CHAPTER II.    The Pool of Tears\n CHAPTER III.   A Caucus-Race and a Long Tale\n CHAPTER IV.    The Rabbit Sends in a Little Bill\n CHAPTER V.     Advice from a Caterpillar\n CHAPTER VI.    Pig and Pepper\n CHAPTER VII.   A Mad Tea-Party\n CHAPTER VIII.  The Queen’s Croquet-Ground\n CHAPTER IX.    The Mock Turtle’s Story\n CHAPTER X.     The Lobster Quadrille\n CHAPTER XI.    Who Stole the Tarts?\n CHAPTER XII.   Alice’s Evidence\n\n\n\n\nCHAPTER I.\nDown the Rabbit-Hole\n\n\nAlice was beginning to get very tired of sitting by her sister on the\nbank, and of having nothing to do: once or twice she had peeped into\nthe book her sister was reading, but it had no pictures or\nconversations in it, “and what is the use of a book,” thought Alice\n“without pictures or conversations?”\n\nSo she was considering in her own mind (as well as she could, for the\nhot day made her feel very sleepy and stupid), whether the pleasure of\nmaking a daisy-chain would be worth the trouble of getting up and\npicking the daisies, when suddenly a White Rabbit with pink eyes ran\nclose by her.\n\nThere was nothing so _very_ remarkable in that; nor did Alice think it\nso _very_ much out of the way to hear the Rabbit say to itself, “Oh\ndear! Oh dear! I shall be late!” (when she thought it over afterwards,\nit occurred to her that she ought to have wondered at this, but at the\ntime it all seemed quite natural); but when the Rabbit actually _took a\nwatch out of its waistcoat-pocket_, and looked at it, and then hurried\non, Alice started to her feet, for it flashed across her mind that she\nhad never before seen a rabbit with either a waistcoat-pocket, or a\nwatch to take out of it, and burning with curiosity, she ran across the\nfield after it, and fortunately was just in time to see it pop down a\nlarge rabbit-hole under the hedge.\n\nIn another moment down went Alice after it, never once considering how\nin the world she was to get out again.\n\nThe rabbit-hole went straight on like a tunnel for some way, and then\ndipped suddenly down, so suddenly that Alice had not a moment to think\nabout stopping herself before she found herself falling down a very\ndeep well.\n\nEither the well was very deep, or she fell very slowly, for she had\nplenty of time as she went down to look about her and to wonder what\nwas going to happen next. First, she tried to look down and make out\nwhat she was coming to, but it was too dark to see anything; then she\nlooked at the sides of the well, and noticed that they were filled with\ncupboards and book-shelves; here and there she saw maps and pictures\nhung upon pegs. She took down a jar from one of the shelves as she\npassed; it was labelled “ORANGE MARMALADE”, but to her great\ndisappointment it was empty: she did not like to drop the jar for fear\nof killing somebody underneath, so managed to put it into one of the\ncupboards as she fell past it.\n\n“Well!” thought Alice to herself, “after such a fall as this, I shall\nthink nothing of tumbling down stairs! How brave they’ll all think me\nat home! Why, I wouldn’t say anything about it, even if I fell off the\ntop of the house!” (Which was very likely true.)\n\nDown, down, down. Would the fall _never_ come to an end? “I wonder how\nmany miles I’ve fallen by this time?” she said aloud. “I must be\ngetting somewhere near the centre of the earth. Let me see: that would\nbe four thousand miles down, I think—” (for, you see, Alice had learnt\nseveral things of this sort in her lessons in the schoolroom, and\nthough this was not a _very_ good opportunity for showing off her\nknowledge, as there was no one to listen to her, still it was good\npractice to say it over) “—yes, that’s about the right distance—but\nthen I wonder what Latitude or Longitude I’ve got to?” (Alice had no\nidea what Latitude was, or Longitude either, but thought they were nice\ngrand words to say.)\n\nPresently she began again. “I wonder if I shall fall right _through_\nthe earth! How funny it’ll seem to come out among the people that walk\nwith their heads downward! The Antipa'



Text Generation Functions: There are three functions defined in the notebook:
* get_probabilities: This function takes a word as input and calculates the probabilities of words that appear after it in the text.
* pick: Given a starting word, this function picks the next word based on their probabilities of occurrence after the starting word.
* make_sentence: This function generates a sentence using the pick function, starting from a given seed word.


```python
def get_probabilities(word):
    mask = raw_data['tokens'] == word
    probabilities = raw_data[mask.shift(1).fillna(False)]['tokens'].value_counts()
    
    return probabilities/probabilities.sum()

def pick(start_word):
    x = get_probabilities(start_word)
   
    return choice(x.index, p = x.values)

def make_sentence(seed):
    for num in range(0, 500):
        print(seed, end = ' ')
        seed = pick(seed)
```


```python
make_sentence('happy')
```

    happy summer day made a feather flock together. ” “ but little ! ” she had not noticed before that her hedgehog just at once crowded round to see it fitted ! ” the king , “ why , resting their shoulders , “ oh , ” “ to get ready to keep moving about for a footman seemed to the mouse did not venture to do practically anything but to death. ’ d let us , ” “ _she_ , that there was snorting like , if my tea when you ’ s the right , my tea ; yet , the conversation . do you may convert to ask the rest waited patiently . “ you ? ” continued , that was going , and came in another moment he said alice in the individual works that one for the other bit . however , in a set the house in an oyster ! ” this time she let the king ; there ’ re a very curious dream ! ” said to herself “ that ? ” alice ; and the gryphon hastily , for some mischief , ” said to the queen . alice thought alice replied in at last concert ! ” “ and the caterpillar . first thought alice . “ there was in a serpent ? _ i must have said : perhaps you from the jury eagerly : she set them . the hall was a whisper , at that ’ ” shouted in things ! ” exclaimed in this sounded best , ” “ i haven ’ t know what the right , and there _was_ no doubt that cheshire cat . i used up. ” said alice . “ just as she noticed a hoarse growl , ” alice . “ explain it further opportunities to lose _your_ opinion , turning to them . special rules in a tone , and a timid voice , i ’ re only alice . “ really you tell you executed. ” said the next . “ if she went . the maximum disclaimer of the mock turtle . “ they ’ t time to be four times since her head appeared , and down a wink with passion . “ arrum. ” said i ’ m afraid , at last , ” he ’ ” the caterpillar . one as she gave herself up and be in the unjust things— ” she felt sure to them at this young lady tells us dry enough to the youth , with me out , ” so long time to herself out to sing you ought to say. ” said the ground near the milk-jug into custody by this paragraph 1.f.3 , so long time together. ’ t like a ) distribution of _this ! ” said alice , she looked down , and two sobs , and the chimney close above a white rabbit read out of bathing machines in reply . the idea what 


```python
make_sentence('sad')
```

    sad and saw maps and picking them a large birds complained that is— ‘ let the hedgehog to turn a house , has lasted the multiplication table as she soon began picking the hatter , and peeped into the little anxiously about it ; and the only bowed and yet i _never_ get what ’ s more , look of rome , you can , and the next , very much under this piece of present of that for it , “ it ’ s a friend . “ here the wretched height to a little sharp hiss made no pleasing them , judging by u.s. laws of that perhaps as the only yesterday , dear little shrieks , leaning over me like a large eyes , which way with , ” cried out the large crowd collected at once in a failure . project gutenberg literary archive foundation 's ein or unenforceability of bread-and-butter in existence ; it a knife , and beasts and you know . “ we were me like : they were , and shoes done that it is made of that , it was busily stirring the queen was , and some day , “ it very civil of his eyes , and now and low-spirited . “ stupid things i shall tell him : “ i wonder if any longer ! ” ( she did so _very_ ugly child away under this sort of the mock turtle replied very difficult game of public support . “ consider your head— do you couldn ’ t go on , reports , please , ” said the hatter looked at the moment the dormouse slowly after a poor little birds of living would not agree to comply with this , that ’ t ! ” he wasn ’ m , there ’ here , and birds with his head ! down it ’ m not ada , so , ” the gryphon only grinned when the air off , and she had hoped ) . “ _everybody_ has agreed to look ! ” said alice , “ i ’ ll go nearer is the phrase `` right , ” “ for , sending a large plate came , so she came suddenly called out , as they all her very angrily . this agreement , “ not . “ then at the chimney , and then she said alice , kept from people here with it yet , ” this sounded an invitation from ear and alice , under a more calmly , ” cried the e—e—evening , who was : you ’ am very dull reality—the grass would break . to double themselves flat , ” but everything is narrow , “ well , my way ! let ’ re mad as such a while more tea and yet , and all came the queen till the gryphon , ” added as a bottle that it takes twenty-four hours to rise like to one , saying 


```python

```
