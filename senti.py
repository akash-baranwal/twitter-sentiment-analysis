# -*- coding: utf-8 -*-
!pip install vadersentiment

nltk.download('all')

import nltk
from tweepy import API 
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import Blobber
tb = Blobber(analyzer=NaiveBayesAnalyzer())
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from string import punctuation 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import json
import os
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
import nltk
consumer_key ='your consumer key'
consumer_secret ='your consumer secret'
access_key ='your access key'
access_secret ='your access secret'

_stopwords=set(list(punctuation) + ['AT_USER','URL'])

def authenticate():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    try:
        api.verify_credentials()
        print('verified')
    except:
        print('not verified')
    return api

def create_csv(screen_name):
    frames=[]
    for i in screen_name:
            i=pd.read_json(f'data/{i}.json', orient='records',lines=True)
            frames.append(i)
    df = pd.concat(frames, ignore_index=True)
    return df

def clean(tweet):
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
#         tweet=tweet.encode('ascii', 'ignore').decode('ascii')
        tweet = word_tokenize(tweet) 
        return ' '.join([word for word in tweet if word not in _stopwords])

def process(df,col):
    df[col]=df[col].progress_apply(clean)
    return df

def stream_tweets(search_term,api,count=1000):
    data = [] 
    counter = 0 
    for tweet in tweepy.Cursor(api.search, q='\"{}\" -filter:retweets'.format(search_term), count=100, lang='en', tweet_mode='extended').items():
        tweet_details = {}
        tweet_details['name'] = tweet.user.screen_name
        tweet_details['tweet'] = tweet.full_text
        tweet_details['retweets'] = tweet.retweet_count
        tweet_details['followers'] = tweet.user.followers_count
        tweet_details['is_user_verified'] = tweet.user.verified
        data.append(tweet_details)
        counter += 1
        if counter == count:
            break
        else:
            pass
    with open('data/{}.json'.format(search_term), 'a+') as f:
        for i in data:
          json.dump(i,f)
    with open('data/{}.json'.format(search_term), 'r') as file :
      filedata = file.read()

      
      filedata = filedata.replace('}{', '},{')

    
    with open('data/{}.json'.format(search_term), 'w') as file:
      file.write(filedata)
        
    print(f'completed {search_term}')

def main():
    screen_name=list(input('enter all search keywords:  ').split())
    count=int(input('enter count:  '))
    api=authenticate()
    for i in screen_name:
        stream_tweets(i,api,count)
    df=create_csv(screen_name)
    processed_df=process(df,'tweet')
    return processed_df
    # return df
df=main()


def sentiment(x):
    s=analyser.polarity_scores(x)
    return max(s,key=s.get)
def classify_sentiment(df):
    df['sentiment']=df['tweet'].progress_apply(sentiment)
    return df

def txblob(x):
    return tb(x).sentiment[0]
def classify_txblob(df):
    df['txtblobsenti']=df['tweet'].progress_apply(txblob)
    return df

def draw_word_cloud(df):
    tweets=' '.join(df['tweet'])
    tweets=tweets.split()
    data_analysis = nltk.FreqDist(tweets)
    filter_words = dict([(m, n) for m, n in data_analysis.items() if len(m) > 3])
    wcloud = WordCloud().generate_from_frequencies(filter_words)
    plt.imshow(wcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def save(df):
    x=input('enter name of csv:  ')
    df.to_csv(f'{x}.csv',index=False)

draw_word_cloud(df)

