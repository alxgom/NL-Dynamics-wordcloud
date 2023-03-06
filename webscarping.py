# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 02:58:35 2023

@author: alexi
"""



#%%

import requests
from bs4 import BeautifulSoup
import json
from collections import Counter
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as  pd
import csv
#%%
# Get the links to the articles
urls = ["https://link.springer.com/journal/11071/volumes-and-issues/111-6",
        "https://link.springer.com/journal/11071/volumes-and-issues/111-5",
        "https://link.springer.com/journal/11071/volumes-and-issues/111-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/111-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/111-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/111-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/110-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/110-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/110-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/110-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/109-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/109-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/109-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/109-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/108-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/108-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/108-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/108-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/107-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/107-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/107-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/107-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/106-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/106-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/106-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/106-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/105-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/105-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/105-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/105-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/104-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/104-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/104-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/104-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/103-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/103-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/103-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/103-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/102-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/102-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/102-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/102-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/101-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/101-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/101-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/101-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/100-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/100-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/100-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/100-1",
        "https://link.springer.com/journal/11071/volumes-and-issues/99-4",
        "https://link.springer.com/journal/11071/volumes-and-issues/99-3",
        "https://link.springer.com/journal/11071/volumes-and-issues/99-2",
        "https://link.springer.com/journal/11071/volumes-and-issues/99-1",]
#all volumes since january 2021
all_keywords = []

# Extract the keywords for each article from both URLs
for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("a", class_="title")

    for article in soup.find_all('h3', {'class': 'c-card__title'}):
        link = article.find('a')
        if link:
            article_url = link.get('href')
        article_response = requests.get(article_url)
        article_html=article_response.text

        # find script tag containing JSON object
        pattern = re.compile(r'<script>.*?window\.dataLayer = (\[.*?\]);.*?</script>', re.DOTALL)
        match = pattern.search(article_html)

        if match:
            # extract JSON object and parse it
            data = match.group(1)
            data = json.loads(data)

            # extract keywords from data
            keywords = data[0]['Keywords'].replace(", ", ",")
            # print(article_url)
            print(keywords)

            # Append keywords to the list
            all_keywords += keywords.split(",")
#%%
# Count the frequency of each keyword
stopwords = set(stopwords.words('english'))
stopwords.update(['observer','solution','equation', 'differential','time','space','algorithm','generalized','motion','problem','dynamical',
                  'analysis','variable','value','theory','number','approach','system','do','unknown','force','estimation',
                  'hirota','image','model','diagnosis','function', 'wikipedia','edit','article','research','method','nonlinear',
                  'nonlinearity','usde','lstm','fpga','multiple']) # Add custom stopwords here
lemmatizer = WordNetLemmatizer()

tokens = []
for word in nltk.word_tokenize(" ".join(all_keywords)):
    word = word.lower()
    if word not in stopwords and re.match('^[a-zA-Z]{3,}$', word):
        lemma = lemmatizer.lemmatize(word)
        tokens.append(lemma)

word_count = Counter(tokens)
#%%
# Generate word cloud
wordcloud = WordCloud(width = 1600, height = 1000, background_color ='white', 
                      min_font_size = 14, max_words=50).generate_from_frequencies(word_count)

# Plot the word cloud
plt.figure(figsize = (16, 10), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()  

#%%

# Save keywords and their frequencies as a CSV file
df = pd.DataFrame.from_dict(word_count, orient='index', columns=['frequency'])
df.index.name = 'keyword'
df.to_csv('keywords.csv')

print(df)
