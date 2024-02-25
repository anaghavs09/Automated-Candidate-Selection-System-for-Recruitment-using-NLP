# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics import classification_report, confusion_matrix

# load the data
resume_df=pd.read_csv(r'D:\ISE\Major Project\CODE\data\resume.csv',encoding='latin-1')
resume_df.head()

# data containing resume
resume_df= resume_df[['resume_text','class']]
print(resume_df.head())

# obtain dataframe information
resume_df.info()

# Find Null/Missing Values
resume_df.isnull().sum()


#View distribution
resume_df['class'].value_counts().plot(kind='barh', figsize=(6, 6))

resume_df['class']=resume_df['class'].apply(lambda x:1 if x=='flagged' else 0)
print(resume_df.head())

resume_df['resume_text']=resume_df['resume_text'].apply(lambda x: str(x).replace('\r',''))
print(resume_df.head())

# download nltk packages
nltk.download('punkt')
# download nltk packages
nltk.download("stopwords")

# Get additional stopwords from nltk
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
stop_words.extend(['from','subject','re','edu','use','email'])

# Define function to remove stop words and remove words with 2 or less characters
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stop_words:
            result.append(token)
            
    return ' '.join(result)

# Cleaned text
resume_df['cleaned']=resume_df['resume_text'].apply(preprocess)
print(resume_df.head())

# plot the word cloud for text that is flagged
plt.figure(figsize = (20,20)) 
wc=WordCloud(max_words=2000,width=1600, height=800,stopwords=stop_words).generate(str(resume_df[resume_df['class']==1].cleaned))
plt.imshow(wc)

# plot the word cloud for text that is flagged
plt.figure(figsize = (20,20)) 
wc_not=WordCloud(max_words=2000,width=1600, height=800,stopwords=stop_words).generate(str(resume_df[resume_df['class']==0].cleaned))
plt.imshow(wc_not)

# CountVectorizer example
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()
countvectorizer=vectorizer.fit_transform(resume_df['cleaned'])

vectorizer.get_feature_names_out()

X=countvectorizer
print(X)
y=resume_df['class']

from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.3)

from sklearn.naive_bayes import MultinomialNB

NB_clf=MultinomialNB()
NB_clf.fit(X_train,y_train)

# Predicting the performance on train data
y_predict_train = NB_clf.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot = True)

# Predicting the Test set results
y_predict_test = NB_clf.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True)

# classification report
print(classification_report(y_test, y_predict_test))