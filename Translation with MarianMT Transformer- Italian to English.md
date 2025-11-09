## $ \color{blue}{\text {Welcome to AI: Critical Principles & Strategy!} } $

#### $\color{purple}{\text{This course is a subset of a}}$ [3 credit graduate level course on Artificial Intelligence Strategy at Rutgers University](https://bloustein.rutgers.edu/graduate/public-informatics/mpi/)

Connect to Faculty: [@ Jim Samuel](https://twitter.com/jimsamuel/)  ----  https://twitter.com/jimsamuel/

---

[Please see the copyright statement below at the end of the notebook.](#ethics)

## $ \color{purple}{\text {Translation with MarianMT Transformer: Italian to English} } $




[General Documentatation](https://huggingface.co/docs/transformers/model_doc/marian)

[OPUS Books, Corpus/Dataset](https://huggingface.co/datasets/opus_books)

[More language models to use](https://huggingface.co/Helsinki-NLP)




```python
!pip install transformers -q
!pip install sentencepiece
!pip install sacremoses
```

    [33mWARNING: You are using pip version 21.3.1; however, version 24.0 is available.
    You should consider upgrading via the '/opt/conda/bin/python3 -m pip install --upgrade pip' command.[0m
    Requirement already satisfied: sentencepiece in /opt/conda/lib/python3.7/site-packages (0.1.95)
    [33mWARNING: You are using pip version 21.3.1; however, version 24.0 is available.
    You should consider upgrading via the '/opt/conda/bin/python3 -m pip install --upgrade pip' command.[0m
    Collecting sacremoses
      Downloading sacremoses-0.0.53.tar.gz (880 kB)
         |████████████████████████████████| 880 kB 33.2 MB/s            
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: regex in /opt/conda/lib/python3.7/site-packages (from sacremoses) (2022.10.31)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from sacremoses) (1.14.0)
    Requirement already satisfied: click in /opt/conda/lib/python3.7/site-packages (from sacremoses) (7.1.2)
    Requirement already satisfied: joblib in /opt/conda/lib/python3.7/site-packages (from sacremoses) (0.14.1)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.7/site-packages (from sacremoses) (4.45.0)
    Building wheels for collected packages: sacremoses
      Building wheel for sacremoses (setup.py) ... [?25ldone
    [?25h  Created wheel for sacremoses: filename=sacremoses-0.0.53-py3-none-any.whl size=895254 sha256=4874d03fe36a5baa44b317e1ab2f6a741ac1fc48c43b0efd47b11ee9524daaf8
      Stored in directory: /home/jovyan/.cache/pip/wheels/87/39/dd/a83eeef36d0bf98e7a4d1933a4ad2d660295a40613079bafc9
    Successfully built sacremoses
    Installing collected packages: sacremoses
    Successfully installed sacremoses-0.0.53
    [33mWARNING: You are using pip version 21.3.1; however, version 24.0 is available.
    You should consider upgrading via the '/opt/conda/bin/python3 -m pip install --upgrade pip' command.[0m



```python
import pandas as pd   # standard: import pandas as pd  
```


```python
df = pd.read_csv("italian_neg_pos_neu.csv")
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Italian-positive</th>
      <th>Italian-negative</th>
      <th>Italian-neutral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sono felice</td>
      <td>Sono arrabbiato</td>
      <td>Quel libro che ho letto mi ha lasciata indiffe...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ieri alla festa mi sono divertita molto</td>
      <td>Che tristezza!</td>
      <td>Il tuo comportamento non mi fa né caldo né freddo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>È stato bello parlare con te</td>
      <td>Questo è un grosso guaio!</td>
      <td>In questo momento mi sento piuttosto calma non...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ho trascorso una bella serata</td>
      <td>Non mi piace quello che dici</td>
      <td>A me interessa poco chi vince il campionato, l...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Oggi è una giornata serena</td>
      <td>Ho visto un film d'orrore ed ora ho paura</td>
      <td>Mi piacciono tutti i tipi di cibo, non ho pref...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sono al settimo cielo!</td>
      <td>Sono molto triste</td>
      <td>Mi ritengo una persona equa e senza pregiudizi</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ho il cuore pieno di gioia</td>
      <td>Ho trascorso una giornataccia!</td>
      <td>Non mi lascio mai influenzare da quello che pe...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sono innamorata</td>
      <td>Il buio mi fa paura</td>
      <td>Il film ha presentato l'argomento in una manie...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Hai un bellissimo sorriso</td>
      <td>La solitudine è triste</td>
      <td>In realtà non ho un'opinione su quanto è accaduto</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Che begli occhi!</td>
      <td>Che brutto!</td>
      <td>Alla fine dei conti puoi fare quello che desid...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Che bella ragazza!</td>
      <td>Mi hai delusa</td>
      <td>Possiamo andare in vacanza da qualsiesi parte,...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sei una persona fortunata</td>
      <td>Che delusione!</td>
      <td>Non ho preferenze su cosa fare stasera</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Che bello averti incontrato</td>
      <td>Ho visto un film orribbile</td>
      <td>Non ho mai favorito nessuno studente, per me s...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Mi piace molto il mio lavoro</td>
      <td>Ho letto un libro orribile e noioso</td>
      <td>Alcune persone non mostrano alcuna emozione</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sei una brava persona</td>
      <td>Sei noioso!</td>
      <td>Ha un carattere molto controllato ed è capace ...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mi piace tanto studiare</td>
      <td>Che cattivo!</td>
      <td>Sembra essere un tipo molto disinvolto, non si...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Sono soddisfatto</td>
      <td>Il cibo non era buono per niente</td>
      <td>È sempre così rilassata su tutto</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ho fatto un bel sogno</td>
      <td>Che cosa disgustosa!</td>
      <td>I due si lasciarano senza emozione</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Che dolce emozione!</td>
      <td>Odio camminare da sola di sera</td>
      <td>È molto impegnato con i suoi esperimenti e spe...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Il pranzo è stato delizioso</td>
      <td>Ho visto un orso e mi sono spaventata a morte</td>
      <td>In questo caso entrambe le parti hanno deciso ...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Il pane è buono</td>
      <td>Che persona malvagia</td>
      <td>Dopo aver visto il film abbiamo scoperto che l...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Abbracciare rende felici</td>
      <td>Mi sento male</td>
      <td>La musica american attrae sempre molti giovani...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>L'abbraccio fa bene al cuore</td>
      <td>Mi sento solo</td>
      <td>Ieri sono andata a cena in un ristorante giapp...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Avere un buon amico rende la vita più bella</td>
      <td>Detesto il mio lavoro</td>
      <td>Oggi sono andata dal dentsita</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Che bel sole!</td>
      <td>Non ti sopporto proprio!</td>
      <td>I miei amici sono venuti a farmi visita</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_pos = pd.DataFrame(df["Italian-positive"], columns=["Italian-positive"])
df_neg = pd.DataFrame(df["Italian-negative"], columns=["Italian-negative"])
df_neu = pd.DataFrame(df["Italian-neutral"], columns=["Italian-neutral"])
```


```python
df_neu.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Italian-neutral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Quel libro che ho letto mi ha lasciata indiffe...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Il tuo comportamento non mi fa né caldo né freddo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>In questo momento mi sento piuttosto calma non...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A me interessa poco chi vince il campionato, l...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mi piacciono tutti i tipi di cibo, non ho pref...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_neg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Italian-negative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sono arrabbiato</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Che tristezza!</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Questo è un grosso guaio!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Non mi piace quello che dici</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ho visto un film d'orrore ed ora ho paura</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_pos.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Italian-positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sono felice</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ieri alla festa mi sono divertita molto</td>
    </tr>
    <tr>
      <th>2</th>
      <td>È stato bello parlare con te</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ho trascorso una bella serata</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Oggi è una giornata serena</td>
    </tr>
  </tbody>
</table>
</div>




```python
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
import sentencepiece


```


```python
model_name = 'Helsinki-NLP/opus-mt-tc-big-it-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading (…)"source.spm";', max=820134.0, style=Progre…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading (…)"target.spm";', max=802852.0, style=Progre…


    



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Downloading (…)"vocab.json";', max=1.0,…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading (…)enizer_config.json";', max=337.0, style=Pr…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading (…)al_tokens_map.json";', max=65.0, style=Pro…


    



    HBox(children=(FloatProgress(value=1.0, bar_style='info', description='Downloading (…)"config.json";', max=1.0…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading (…)"pytorch_model.bin";', max=575827779.0, st…


    



    HBox(children=(FloatProgress(value=0.0, description='Downloading (…)ration_config.json";', max=301.0, style=Pr…


    



```python
text = "Il pane sul tavolo sembra delizioso"

input_ids = tokenizer.encode(text, return_tensors="pt")
outputs = model.generate(input_ids)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded) 
```

    The bread on the table looks delicious



```python
results = []
for sentence in df_pos['Italian-positive']:
  input_ids = tokenizer.encode(sentence, return_tensors="pt")
  outputs = model.generate(input_ids)
  results.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
df_pos['English_opus-mt-tc-big-it-en'] = results
```


```python
df_pos.head()
```


```python
df_pos.tail()
```

* Looks like it worked!


```python
df_pos.describe()
```


```python
# Shorten translation name
df_pos.columns = ["Italian-positive", "English_opus-mt-posi"]
```


```python
df_pos.head()
```


```python
df_pos.head()
header = ["Italian-positive", "English_opus-mt-posi"]
```


```python
df_pos.to_csv("English_opus-mt-posi.csv", columns = header)
```


```python
# Look at above and repeat for Neutral 
results = []
for sentence in df_neu['Italian-neutral']:
  input_ids = tokenizer.encode(sentence, return_tensors="pt")
  outputs = model.generate(input_ids)
  results.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
df_neu['English_opus-mt-tc-big-it-en'] = results
```


```python
# Shorten translation name
df_neu.columns = ["Italian-neutral", "English_opus-mt-neu"]
```


```python
df_neu.head()
header = ["Italian-neutral", "English_opus-mt-neu"]
```


```python
df_neu.to_csv("English_opus-mt-neu.csv", columns = header)
```


```python
# Look at above for positive and repeat for Negative
results = []
for sentence in df_neg['Italian-negative']:
  input_ids = tokenizer.encode(sentence, return_tensors="pt")
  outputs = model.generate(input_ids)
  results.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
df_neg['English_opus-mt-tc-big-it-en'] = results
```


```python
# Shorten translation name
df_neg.columns = ["Italian-negative", "English_opus-mt-neg"]
```


```python
df_neg.head()
header = ["Italian-negative", "English_opus-mt-neg"]
```


```python
df_neg.to_csv("English_opus-mt-neg.csv", columns = header)
```


```python
df_new = pd.concat([df_pos, df_neu,df_neg], axis=1)
```


```python
df_new
```


```python
header = df_new.columns
```


```python
df_new.to_csv("English_opus-mt-final.csv", columns = header)
```

* If you have run everything correctly, your data_main_1 file should have 6 tabs now:
  * 3 Original 
  * 3 Opus-MT

* See "Data-Main-1-Final-Output" (correct solution) to verify your final output. 

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
