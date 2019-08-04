import praw
import json

import pandas as pd
import numpy as np

import string
import urllib.request
import httplib2
import requests
from requests.exceptions import ConnectionError, RequestException
from bs4 import BeautifulSoup, SoupStrainer
import csv
import time
import re  #https://regex101.com/

import os
import string
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from parse import *

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# load reddit credentials
with open('reddit_credentials.json') as cred_data:
    info = json.load(cred_data)
client_id = info['my_client_id']
client_secret = info['my_client_secret']
user_agent = info['my_user_agent']


# Create the api endpoint
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

# comment threads to scrape
# https://www.reddit.com/r/soccer/search/?q=women%27s%20world%20cup&restrict_sr=1

urls_wwc = [
    'https://www.reddit.com/r/soccer/comments/c8cy9w/match_thread_england_vs_united_states_fifa_womens/',
    'https://www.reddit.com/r/soccer/comments/ca7jix/match_thread_united_states_vs_netherlands_fifa/',
    'https://www.reddit.com/r/soccer/comments/c4qb68/match_thread_spain_vs_united_states_fifa_womens/',
    'https://www.reddit.com/r/soccer/comments/c8s0pb/match_thread_netherlands_vs_sweden_fifa_womens/',
    'https://www.reddit.com/r/soccer/comments/c3te38/match_thread_norway_vs_australia_fifa_womens/',
    'https://www.reddit.com/r/soccer/comments/c6o3a7/match_thread_france_vs_united_states_fifa_womens/',
    'https://www.reddit.com/r/soccer/comments/bzgjgv/match_thread_united_states_vs_thailand_womens/',
    'https://www.reddit.com/r/soccer/comments/c69g2g/match_thread_norway_vs_england_fifa_womens_world/',
    'https://www.reddit.com/r/soccer/comments/c2zywy/match_thread_sweden_vs_united_states_fifa_womens/',
    'https://www.reddit.com/r/soccer/comments/c46de8/match_thread_england_vs_cameroon_fifa_womens/',
    'https://www.reddit.com/r/soccer/comments/c49maz/match_thread_france_vs_brazil_fifa_womens_world/',
    'https://www.reddit.com/r/soccer/comments/bzv32w/match_thread_france_vs_norway_womens_world_cup/',
    'https://www.reddit.com/r/soccer/comments/c5dc0b/match_thread_netherlands_vs_japan_fifa_womens/',
    'https://www.reddit.com/r/soccer/comments/bzsuvt/match_thread_germany_vs_spain_womens_world_cup/',
    'https://www.reddit.com/r/soccer/comments/c1bcwu/match_thread_united_states_vs_chile_womens_world/',
    'https://www.reddit.com/r/soccer/comments/c4th9e/match_thread_sweden_vs_canada_fifa_womens_world/',
    'https://www.reddit.com/r/soccer/comments/c7152r/match_thread_germany_vs_sweden_fifa_womens_world/',
    'https://www.reddit.com/r/soccer/comments/c6ymxd/match_thread_italy_vs_netherlands_fifa_womens/',
    'https://www.reddit.com/r/soccer/comments/c2l2gv/match_thread_scotland_vs_argentina_fifa_womens/',
    'https://www.reddit.com/r/soccer/comments/bxycuw/match_thread_france_vs_south_korea_fifa_womens/'
]

# https://www.reddit.com/r/soccer/search/?q=match%20thread%20%22fifa%20world%20cup%22&restrict_sr=1

urls_mwc = [
    'https://www.reddit.com/r/soccer/comments/8wut2w/match_thread_russia_vs_croatia_2018_fifa_world/',
    'https://www.reddit.com/r/soccer/comments/8rrf5d/match_thread_germany_vs_mexico_2018_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8y1m08/match_thread_england_v_croatia_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8v1v42/match_thread_france_vs_argentina_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8ubph8/match_thread_serbia_vs_brazil_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8vu3in/match_thread_colombia_vs_england_2018_fifa_world/',
    'https://www.reddit.com/r/soccer/comments/8vinxo/match_thread_brazil_vs_mexico_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8wm01p/match_thread_brazil_vs_belgium_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8tqisr/match_thread_uruguay_vs_russia_2018_fifa_world/',
    'https://www.reddit.com/r/soccer/comments/8v9wt7/match_thread_spain_vs_russia_2018_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8sk7y3/match_thread_kingdom_of_spain_vs_islamic_republic/',
    'https://www.reddit.com/r/soccer/comments/8rzxvi/match_thread_belgium_vs_panama_2018_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8u00ni/match_thread_denmark_vs_france_2018_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8rk2rg/match_thread_peru_vs_denmark_2018_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8tsmff/match_thread_iran_vs_portugal_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8ujf8m/match_thread_senegal_vs_colombia_2018_fifa_world/',
    'https://www.reddit.com/r/soccer/comments/8yta8b/match_thread_belgium_vs_england_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8u9mbf/match_thread_mexico_vs_sweden_2018_fifa_world_cup/',
    'https://www.reddit.com/r/soccer/comments/8rbdjd/match_threadkingdom_of_morocco_vs_islamic/',
    'https://www.reddit.com/r/soccer/comments/8ssbcr/match_thread_france_vs_peru_2018_fifa_world_cup/'
]


def get_comments(url_list, gender, column_list, df):
    """
    takes a list of urls to scrape, the year and gender of the world cup
    returns a dataframe of comments and various descriptors for the comment
    """

    for url in url_list:
        submission = reddit.submission(url=url)
        top_level_comments = list(submission.comments)
        for comment in top_level_comments:
            try:
                new_entry = []
                new_entry.append(url)
                new_entry.append(gender)
                new_entry.append(search("match_thread_{}_vs", url)[0])
                new_entry.append(search("vs_{}_", url)[0])
                new_entry.append(comment.body)
                new_entry.append(comment.controversiality)
                new_entry.append(int(comment.score))
                new_entry.append(comment.gilded)
                new_entry.append(len(comment.replies))

                single_tweet_df = pd.DataFrame([new_entry], columns=column_list)
                df = df.append(single_tweet_df, ignore_index=True)

            except:
                pass

    return df

def get_sentiment(com):
    """takes a comment and uses TextBlob and vaderSentiment to return sentiment values for the comment

    Returns dictionary of sentiment values
    """

    analyser = SentimentIntensityAnalyzer()
    scores = analyser.polarity_scores(com)
    vneg = scores['neg']
    vneu = scores['neu']
    vpos = scores['pos']
    vcomp = scores['compound']
    subj = TextBlob(com).sentiment.subjectivity
    pol = TextBlob(com).sentiment.polarity

    # save all values in dictionary
    sentiments = {'vader_negative': vneg,
                 'vader_neutral': vneu,
                 'vader_positivity': vpos,
                 'vader_compound': vcomp,
                 'tb_polarity': pol,
                 'tb_subjectivity': subj}

    return sentiments

# scrape comments from url lists and combine into dataframe
cols = ['url','tournament', 'team_1', 'team_2', 'comment', 'controversiality', 'score',
        'gilded', 'top_level_replies']
df = pd.DataFrame(columns=cols)
df = get_comments(urls_wwc_2019, 1, cols,  df)
df = get_comments(urls_mwc_2018, 0, cols,  df)
df = get_comments(urls_wwc_2015, 1, cols,  df)
df = get_comments(urls_mwc_2014, 0, cols,  df)

#  add sentiment analysis to DataFramedf['sentiments'] = df['comment'].apply(lambda x: get_sentiment(x))
df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)


# save dataframe to csv
df.to_csv('data/reddit/reddit_comments.csv')
