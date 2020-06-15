import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re


url="https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
s=requests.get(url).content
df1=pd.read_csv(io.StringIO(s.decode('utf-8')))
df1['date']= pd.to_datetime(df1['date'])
df1.to_csv(os.getcwd()+"/app/static/data/us-states.csv",index_label = 'id')




url="https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-states.csv"
s=requests.get(url).content
df1=pd.read_csv(io.StringIO(s.decode('utf-8')))
df1['date']= pd.to_datetime(df1['date'])
URL = "https://inkplant.com/code/state-latitudes-longitudes"
r = requests.get(URL)
mycolumns = ['state', 'Latitude', 'Longitude']
state_coordinates = pd.DataFrame(columns=mycolumns)
soup = BeautifulSoup(r.content, 'html5lib') 
tbody = soup.find_all('tbody')
tr = tbody[0].find_all('tr')
for i in tr:
  td=i.find_all('td')
  state_coordinates.loc[len(state_coordinates)] = [td[0].getText(),td[1].getText(),td[2].getText()]
state_coordinates['Latitude'] = state_coordinates['Latitude'].astype(float)
state_coordinates['Longitude'] = state_coordinates['Longitude'].astype(float)
state_coordinates.to_csv(os.getcwd()+"/app/static/data/location.csv",index_label = 'id')






news_contents = []
list_links = []
list_titles = []
def parse_dailymail():
  url = "https://www.dailymail.co.uk/home/search.html?sel=site&searchPhrase=Coronavirus"
  r1 = requests.get(url)
  r1.status_code


  coverpage = r1.content


  soup1 = BeautifulSoup(coverpage, 'html5lib')


  coverpage_news = soup1.find_all('h3', class_='sch-res-title')
  l = len(coverpage_news)
  number_of_articles = l
  count = 1
  n=0
  while(True):
        
    
      link = "https://www.dailymail.co.uk" + coverpage_news[n].find('a')['href']
      
      if link.startswith("https://www.dailymail.co.uk/news/"):
        print(link)
        list_links.append(link)
      
      
        title = coverpage_news[n].find('a').get_text()
        list_titles.append(title)
      
      
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('p', class_='mol-para-with-font')
      
      
        list_paragraphs = []
        final_article = ""
        for p in np.arange(0, len(body)):
            paragraph = body[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)
          
      
        article = re.sub("\\xa0", "", final_article)
          
        news_contents.append(article)
      if n==(l-1):
        break
      n=n+1 


def parse_theguardian():
  url = "https://www.theguardian.com/uk"
  r1 = requests.get(url)
  r1.status_code

  # We'll save in coverpage the cover page content
  coverpage = r1.content

  # Soup creation
  soup1 = BeautifulSoup(coverpage, 'html5lib')

  # News identification
  coverpage_news = soup1.find_all('h3', class_='fc-item__title')
  l=len(coverpage_news)
  number_of_articles = l


  for n in np.arange(0, number_of_articles):
        
    # We need to ignore "live" pages since they are not articles
    if "live" in coverpage_news[n].find('a')['href']:  
        continue
    
    # Getting the link of the article
    link = coverpage_news[n].find('a')['href']
    
    
    # Getting the title
    title = coverpage_news[n].find('a').get_text()
    
    
    # Reading the content (it is divided in paragraphs)
    article = requests.get(link)
    article_content = article.content
    soup_article = BeautifulSoup(article_content, 'html5lib')
    body = soup_article.find_all('div', class_='content__article-body from-content-api js-article__body')
    if len(body)!=0 :
      x = body[0].find_all('p')
    
      # Unifying the paragraphs
      list_paragraphs = []
      for p in np.arange(0, len(x)):
          paragraph = x[p].get_text()
          list_paragraphs.append(paragraph)
          final_article = " ".join(list_paragraphs)
        
      news_contents.append(final_article)
      list_titles.append(title)
      list_links.append(link)
    else:
      number_of_articles = number_of_articles +1
def parse_themirror():
    
    # url definition
    url = "https://www.mirror.co.uk/"
    
    # Request
    r1 = requests.get(url)
    r1.status_code

    # We'll save in coverpage the cover page content
    coverpage = r1.content

    # Soup creation
    soup1 = BeautifulSoup(coverpage, 'html5lib')

    # News identification
    coverpage_news = soup1.find_all('a', class_='headline publication-font')
    l=len(coverpage_news)
    number_of_articles = l

    for n in np.arange(0, number_of_articles):

        # Getting the link of the article
        link = coverpage_news[n]['href']
        list_links.append(link)

        # Getting the title
        title = coverpage_news[n].get_text()
        list_titles.append(title)

        # Reading the content (it is divided in paragraphs)
        article = requests.get(link)
        article_content = article.content
        soup_article = BeautifulSoup(article_content, 'html5lib')
        body = soup_article.find_all('div', class_='articulo-cuerpo')
        x = soup_article.find_all('p')

        # Unifying the paragraphs
        list_paragraphs = []
        for p in np.arange(0, len(x)):
            paragraph = x[p].get_text()
            list_paragraphs.append(paragraph)
            final_article = " ".join(list_paragraphs)

        news_contents.append(final_article)

parse_dailymail()
parse_theguardian()
parse_themirror()
import os
df_news = pd.DataFrame(
    {'Article Title': list_titles,
     'Article Link': list_links,
     'Article Content': news_contents})


#Imports the libraries and read the data files
import random
import re
from nltk.stem.snowball import SnowballStemmer
import os
import gensim
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
import nltk
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



#"""### Downloading extra dependencies from NLTK"""
nltk.download('stopwords')
nltk.download('punkt')


from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 






#"""### Tokenizing the document and filtering the tokens"""
def tokenize(train_texts):
  filtered_tokens = []
  tokens = [word for sent in nltk.sent_tokenize(train_texts) for word in nltk.word_tokenize(sent)]
  for token in tokens:
    if re.search('[a-zA-Z]',token):
        if (('http' not in token) and ('@' not in token) and ('<.*?>' not in token) and token.isalnum() and (not token in stop_words)):
            filtered_tokens.append(token)
  return filtered_tokens





#"""### Tokenizing and stemming using Snowball stemmer"""
def tokenize_stem(train_texts):
  tokens = tokenize(train_texts)
  stemmer = SnowballStemmer('english')
  stemmed_tokens = [stemmer.stem(token) for token in tokens]
  return stemmed_tokens



#"""**Loading Data**"""
seed = 137
def load_data(seed):
  train_texts = []
  for index,data in df_news.iterrows():
        train_texts.append(str(data))
  random.seed(seed)
  random.shuffle(train_texts)
  return train_texts
train_texts = load_data(seed)






#"""**Create a list of tagged emails. **"""
LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
all_content = []
j=0
k=0

vocab_tokenized = []
vocab_stemmed = []
for text in train_texts:
    allwords_tokenized = tokenize(text)
    vocab_tokenized.append(allwords_tokenized)
        
    allwords_stemmed = tokenize_stem(text)
    vocab_stemmed.append(allwords_tokenized)

for text in vocab_tokenized:           
    # add tokens to list
    if len(text)>0:
        all_content.append(LabeledSentence1(text,[j]))
        j+=1
        
    k+=1

print("Number of emails processed: ", k)
print("Number of non-empty emails vectors: ", j)





#"""**Create a model using Doc2Vec and train it**"""

d2v_model = Doc2Vec(all_content, vector_size = 2000,min_count = 5,dm = 0, 
                alpha=0.0025, min_alpha=0.0001)
d2v_model.train(all_content, total_examples=d2v_model.corpus_count, epochs=50, start_alpha=0.002, end_alpha=-0.016)



#"""**Apply K-means clustering on the model**"""
#Elbow Method
'''
nc = range(1,10)
kmeans = []
score = []
kmeans = [KMeans(n_clusters = i, n_init = 100, max_iter = 500, precompute_distances = 'auto' ) for i in nc]               
score = [kmeans[i].fit(d2v_model.docvecs.doctag_syn0).score(d2v_model.docvecs.doctag_syn0) for i in range(len(kmeans))]
# Plot the elbow
plt.plot(nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
'''



K_value = 4
kmeans_model = KMeans(n_clusters = K_value, init='k-means++', n_init = 2000, max_iter = 6000, precompute_distances = 'auto')  
X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
labels=kmeans_model.labels_.tolist()
clusters = kmeans_model.fit_predict(d2v_model.docvecs.doctag_syn0)


df_clus=pd.DataFrame({'Cluster':clusters})

df_news.join(df_clus).to_csv(os.getcwd()+"/app/static/data/news_data.csv",index_label = 'id')

# #PCA
# l = kmeans_model.fit_predict(d2v_model.docvecs.vectors_docs)
# pca = PCA(n_components=2).fit(d2v_model.docvecs.vectors_docs)
# datapoint = pca.transform(d2v_model.docvecs.vectors_docs)



# #GRAPH
# #"""**Plot the clustering result**"""

# plt.figure
# label1 = ["#FFFF00", "#008000", "#0000FF", "#800080"]
# color = [label1[i] for i in labels]
# plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

# centroids = kmeans_model.cluster_centers_
# centroidpoint = pca.transform(centroids)
# plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
# plt.show()
