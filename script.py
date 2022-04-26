## Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
pd.set_option('display.max_colwidth', 140)
analyzer = SentimentIntensityAnalyzer()

example = '''
The movie is terrible.
'''
vs = analyzer.polarity_scores(example)
print(str(vs))

review = '''
no film in recent has left me with such conflicted feelings as neil jordan's harrowing , humorous , horrifying adaptation of patrick mccabe's novel about young lad francie brady's ( eamonn owens ) descent into madness in 1960s ireland . 
on one hand , it was difficult for me to become invested in francie's story because he is such an unsavory character , unjustifyably venting his rage at his nosy but otherwise harmless neighbor mrs . nugent ( fiona shaw ) . 
on another hand , i found it difficult to laugh at some of francie's darkly comic shenanigans because he obviously is such a sick , needy child , having been raised by a drunken father ( stephen rea ) and a suicidal mother ( aisling o'sullivan ) . 
on yet another hand , i also found it difficult to completely sympathize with francie during his more emotional scenes because some of his , for lack of a better word , " bad " deeds are so incredibly shocking in their brutality and the malicious glee in which he performs them . 
however , the butcher boy's power is undeniable , and the film as a whole is unforgettable--perhaps because it is so disturbing . 
what makes it so unsettling is the francie's overall wink-wink yet matter-of-fact attitude about everything , expressed in a cheeky voiceover narration delivered by the adult francie ( rea again ) . 
think heavenly creatures played largely for laughs , and you'll sort of understand . 
anchoring the whole film is the astonishing debut performance of owens ; love francie or hate him , you cannot take your eyes off of owens . 
the butcher boy truly is a twisted , unusual film that is bound to make just about anyone uncomfortable . 
in the lobby after the screening , i overheard one man raving about how great yet disturbing it was ; i also heard one particularly offended woman say with disgust , " that movie was so unfunny ! " 
 " i didn't know what to expect . 
it's like something you chase for so long , but then you don't know how to react when you get it . 
i still don't know how to react . " 
--michael jordan , on winning his first nba championship in 1991 . . . or , 
my thoughts after meeting him on november 21 , 1997 
'''
print(review)
vs = analyzer.polarity_scores(review)
print(str(vs))

# this splits the review by newlines and removes any empty strings
sentences = []
for sentence in review.splitlines():
    if sentence:
        sentences.append(sentence)
sentences

df = pd.DataFrame(columns=['sentence','neg','neu','pos','compound'])
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    vs['sentence'] = sentence
    df = df.append(dict(vs), ignore_index=True)
df


## twitter tokens

# api key
#HS2o6UO72ni0KH5eRBoC8mpzC

#api key secrete
#F0jmChRhhiabAOuv6gwBgX2QPn7ua5MalPj6JkVbYRVU20XyHl

#bearere token
#AAAAAAAAAAAAAAAAAAAAAAGTbAEAAAAA0POK7FBynBFdCh3mcB8fF%2Fyh7a0%3DrPtreFvMy5F73pYY7JYeo9P1CPxLLUrOuBQbobbPXdNDbSdo7j


################ Tweet classifier ##############

import tweepy
# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
#To add wait time between requests
import time


# create token env variable
os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAAGTbAEAAAAANpnebjpE2m%2FQEZCWeD8KdAn%2Bv0U%3DSSjrPGc4rm5foXSOQt30IfIPRkLi9jNXAggkh32cEnsCtvcbpg'

# define auth that returns this secret token
def auth():
    return os.getenv('TOKEN')

client = tweepy.Client(bearer_token=auth())

# Get 'biden' tweets
biden_query = 'from :POTUS -is:retweet'

biden_tweets = client.search_recent_tweets(query=biden_query, tweet_fields=['context_annotations', 'created_at'], max_results=100)

# Create biden tweet array

biden_tweets_array = []
for tweet in biden_tweets.data:
    print(tweet.text)
    print("")
    biden_tweets_array.append(tweet.text)

# get 'musk' tweets 
musk_query = 'from :elonmusk -is:retweet'

musk_tweets = client.search_recent_tweets(query=musk_query, tweet_fields=['context_annotations', 'created_at'], max_results=100)  

musk_tweets_array = []
for tweet in musk_tweets.data:
    print(tweet.text)
    print("")
    musk_tweets_array.append(tweet.text)


print(musk_tweets_array)

# create dataframes with tweet text, label, and we can add sentiment analysis score as well
biden_df = pd.DataFrame(columns=['tweet','compound'])
for tweet in biden_tweets_array:
    vs = analyzer.polarity_scores(tweet)
    vs['tweet'] = tweet
    biden_df = biden_df.append(dict(vs), ignore_index=True)
    biden_df['label'] = "biden"
len(biden_df)


musk_df = pd.DataFrame(columns=['tweet','compound'])
for tweet in musk_tweets_array:
    vs = analyzer.polarity_scores(tweet)
    vs['tweet'] = tweet
    musk_df = musk_df.append(dict(vs), ignore_index=True)
    musk_df['label'] = "musk"
len(musk_df)

dfs = [biden_df, musk_df]
df = pd.concat(dfs)

# randomise rows
df = df.sample(frac=1).reset_index(drop=True)
df

## Tweet classification

# imports
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import re
import pickle

import nltk
from nltk import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.stem import SnowballStemmer 
from nltk.stem import PorterStemmer 
from nltk.corpus import wordnet

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

import pandas as pd
import joblib

# downloads
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# configuring stop word list
stop_words = None
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
nltk_stop_words = nltk.corpus.stopwords.words('english')

# helper functions


# combine a defined stop word list (or no stop word list) with any extra stop words defined
def set_stop_words(stop_word_list, extra_stop_words):
    if len(extra_stop_words) > 0:
        if stop_word_list is None:
            stop_word_list = []
        stop_words = list(stop_word_list) + extra_stop_words
    else:
        stop_words = stop_word_list
        
    return stop_words

# initiate stemming or lemmatising
def set_normaliser(normalise):
    if normalise == 'PorterStemmer':
        normaliser = PorterStemmer()
    elif normalise == 'SnowballStemmer':
        normaliser = SnowballStemmer('english')
    elif normalise == 'WordNetLemmatizer':
        normaliser = WordNetLemmatizer()
    else:
        normaliser = None
    return normaliser

# we are using a custom tokenisation process to allow different tokenisers and stemming/lemmatising ...
def tokenise(doc):
    global tokeniser, normalise, normaliser
    
    # you could obviously add more tokenisers here if you wanted ...
    if tokeniser == 'sklearn':
        tokenizer = RegexpTokenizer(r"(?u)\b\w\w+\b") # this is copied straight from sklearn source
        tokens = tokenizer.tokenize(doc)
    elif tokeniser == 'word_tokenize':
        tokens = word_tokenize(doc)
    elif tokeniser == 'wordpunct':
        tokens = wordpunct_tokenize(doc)
    elif tokeniser == 'nopunct':
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(doc)
    else:
        tokens = word_tokenize(doc)
        
    # if using a normaliser then iterate through tokens and return the normalised tokens ...
    if normalise == 'PorterStemmer':
        return [normaliser.stem(t) for t in tokens]
    elif normalise == 'SnowballStemmer':
        return [normaliser.stem(t) for t in tokens]
    elif normalise == 'WordNetLemmatizer':
        # NLTK's lemmatiser needs parts of speech, otherwise assumes everything is a noun
        pos_tokens = nltk.pos_tag(tokens)
        lemmatised_tokens = []
        for token in pos_tokens:
            # NLTK's lemmatiser needs specific values for pos tags - this rewrites them ...
            # default to noun
            tag = wordnet.NOUN
            if token[1].startswith('J'):
                tag = wordnet.ADJ
            elif token[1].startswith('V'):
                tag = wordnet.VERB
            elif token[1].startswith('R'):
                tag = wordnet.ADV
            lemmatised_tokens.append(normaliser.lemmatize(token[0],tag))
        return lemmatised_tokens
    else:
        # no normaliser so just return tokens
        return tokens

# CountVectorizer pre-processor - remove numerics.
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text

# train-test split
X = df['tweet']
Y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30, stratify=Y)

X_train
# Feature extraction

Vectorizer = CountVectorizer # valid values are CountVectorizer or TfidfVectorizer without quotes
lowercase = True # valid values are False or True (to make it lowercase)
tokeniser = 'nopunct' # should be one of the following (with quotes): 'sklearn', 'word_tokenize', wordpunct', 'nopunct' 
normalise = 'WordNetLemmatizer' # should be one of the following: None or 'PorterStemmer' or 'SnowballStemmer' or 'WordNetLemmatizer'
stop_word_list = None # should be either None or nltk_stop_words or sklearn_stop_words
extra_stop_words = [] #list of any extra stop words e.g. ['one' , 'two']
min_df = 0.05 
max_df = 0.95 
max_features = 100
ngram_range = (1, 1) 
encoding = 'utf-8'
decode_error = 'ignore' # what to do if contains characters not of the given encoding - options 'strict', 'ignore', 'replace'

# tweets not very long - small kbest
kbest = 15

# Define pipeline
stop_words = set_stop_words(stop_word_list, extra_stop_words)
normaliser = set_normaliser(normalise)

pipeline = Pipeline([
    ('vectorizer', Vectorizer(tokenizer = tokenise,
                              lowercase = lowercase,
                              min_df = min_df, 
                              max_df = max_df, 
                              max_features = max_features,
                              stop_words = stop_words, 
                              ngram_range = ngram_range,
                              encoding = encoding, 
                              preprocessor = preprocess_text,
                              decode_error = decode_error)),
    ('selector', SelectKBest(score_func = mutual_info_classif, k=kbest)),
    ('classifier', MultinomialNB()), #here is where you would specify an alternative classifier
])

print('Classifier settings')
print('===================')
print('classifier:', type(pipeline.steps[2][1]).__name__)
print('selector:', type(pipeline.steps[1][1]).__name__)
print('vectorizer:', type(pipeline.steps[0][1]).__name__)

print('lowercase:', lowercase)
print('tokeniser:', tokeniser)

print('normalise:', normalise)
print('min_df:', min_df)
print('max_df:', max_df)
print('max_features:', max_features)
if stop_word_list == nltk_stop_words:
    print('stop_word_list:', 'nltk_stop_words')
elif stop_word_list == sklearn_stop_words:
    print('stop_word_list:', 'sklearn_stop_words')
else:
    print('stop_word_list:', 'None')
print('extra_stop_words:', extra_stop_words)
print('ngram_range:', ngram_range)
print('encoding:', encoding)
print('decode_error:', decode_error)
print('kbest:', kbest)

# Fit model
pipeline.fit(X_train, y_train)
y_predicted = pipeline.predict(X_test)

# print report
print('Evaluation metrics')
print('==================')
print(metrics.classification_report(y_test, y_predicted))
cm = metrics.confusion_matrix(y_true=y_test, y_pred=y_predicted, labels=["biden", "musk"])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["biden","musk"])
disp = disp.plot(include_values=True, cmap='Blues', ax=None, xticks_rotation='vertical')
plt.show()


# Finally, save the pipeline:
#pickle.dump(pipeline, 'model.pkl')

# fit the model
#pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "model.x")


