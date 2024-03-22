# coding: utf-8
# Load dependencies for this Jupyter Notebook
import os, json, errno
import pandas as pd
import numpy as np
from sys import argv
import string
import time
# from lib.util import to_unix_tmsp, parse_twitter_datetime
from multiprocessing import Process


#imports for text feature extraction:
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re
from nltk.corpus import stopwords as stp
from textblob import TextBlob


class Tweets:

    def __init__(self, event_name, output_dir="data/tweets"):
        self.event = event_name
        self.data = {}
        self.output_dir = output_dir
        self.printable = set(string.printable)

        utc_offset = {
            "germanwings-crash": 1,
            "sydneysiege": 11,
            "ottawashooting": -4,
            "ferguson":-5,
            "charliehebdo":+1,
        }
        self.utc_offset = utc_offset[self.event]
    
    def append(self, twt, cat, thrd, is_src):
        """ Convert tweet metadata into features.

        Key to the `self.data` dictionary defined in this function define columns in
        the CSV file produced by the `export` method.

        Params:
            - twt: The new tweet to add to the table
            - cat: The category of the tweet, e.g. rumour
            - thrd: The thread id of the tweet
            - is_src : True if it's a source tweet and false if it is a reaction
        """
        twt['category'] = cat
        twt["thread"] = thrd
        twt["event"] = self.event
        twt["is_src"] = is_src

        twt_text=twt["text"]
        twt_text_filtered=str()
        for c in twt_text:
            if c in self.printable:
                twt_text_filtered+=c

        #print('twt text:',twt_text_filtered)
        #print('type of twt_text', type(twt_text_filtered))
        text_features=self.tweettext2features(twt_text_filtered)

        features = {
            # Thread metadata
            "is_rumor": lambda obj : 1 if obj['category'] == "rumours" else 0,
            
            # Conservation metadata
            "thread" : lambda obj : obj["thread"],
            "in_reply_tweet" : lambda obj : obj.get("in_reply_to_status_id"),
            "event" : lambda obj : obj.get("event"),
            "text" : lambda obj : obj.get("text"),
            "tweet_id" : lambda obj : obj.get("id"),
            "is_source_tweet" : lambda obj : 1 if twt["is_src"] else 0,
            "in_reply_user" : lambda obj : obj.get("in_reply_to_user_id"),
            "user_id" : lambda obj : obj["user"].get("id"),
            
            # Tweet metadata
            "hashtags_count": lambda obj : len(obj["entities"].get("hashtags", [])),
            "retweet_count": lambda obj : obj.get("retweet_count", 0),
            "favorite_count": lambda obj : obj.get("favorite_count"),
            "mentions_count": lambda obj : len(obj["entities"].get("user_mentions", "")),

            # User metadata
            "user.tweets_count": lambda obj: obj["user"].get("statuses_count", 0),
            "user.verified": lambda obj: 1 if obj["user"].get("verified") else 0,
            "user.followers_count": lambda obj: obj["user"].get("followers_count"),
            "user.friends_count": lambda obj: obj["user"].get("friends_count"),
        }

        for col in features:
            self.data.setdefault(col, []).append(features[col](twt))

        for col in text_features:
            self.data.setdefault(col, []).append(text_features[col])

    def tweettext2features(self, tweet_text):   
        """ Extracts some text features from the text of each tweet. The extracted features are as follows:
        hasperiod: has period
        number_punct: number of punctuation marks
        negativewordcount: the count of the defined negative word counts
        positivewordcount :the count of the defined positive word counts
        sentimentscore: sentiment score by textBlob
        Param:
            - tweet_text: text of tweet
        Return: a dict containing the mentioned text features
        """
        #punctuations
        def punctuationanalysis(tweet_text):
            punctuations= ["\"","(",")","*",",","-","_",".","~","%","^","&","!","#",'@'
               "=","\'","\\","+","/",":","[","]","«","»","،","؛","?",".","…","$",
               "|","{","}","٫",";",">","<","1","2","3","4","5","6","7","8","9","0"]
            hasperiod=sum(c =='.' for c in tweet_text)
            number_punct=sum(c in punctuations for c in tweet_text)
            return {'hasperiod':hasperiod,'number_punct':number_punct}

        def negativewordcount(tokens):
            count = 0
            negativeFeel = ['tired', 'sick', 'bord', 'uninterested', 'nervous', 'stressed',
                            'afraid', 'scared', 'frightened', 'boring','bad',
                            'distress', 'uneasy', 'angry', 'annoyed', 'pissed',"hate",
                            'sad', 'bitter', 'down', 'depressed', 'unhappy','heartbroken','jealous', 'fake', 'stupid', 'strange','absurd', 'crazy']
            for negative in negativeFeel:
                if negative in tokens:
                    count += 1
            return count

        def positivewordcount(tokens):
            count = 0
            positivewords = ['joy', ' happy', 'hope', 'kind', 'surprise'
                            , 'excite', ' interest', 'admire',"delight","yummy",
                            'confidenc', 'good', 'satisf', 'pleasant',
                            'proud', 'amus', 'amazing', 'awesome',"love","passion","great","like","wow","delicious", "true", "correct", "crazy"]
            for pos in positivewords:
                if pos in tokens:
                    count += 1
            return count

        def sentimentscore(tweet_text):
            analysis = TextBlob(tweet_text)
            return analysis.sentiment.polarity

        def tweets2tokens(tweet_text):
            tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+','', tweet_text.lower()))
            url=0
            for token in tokens:
                if token.startswith( 'http' ):
                    url=1

            return tokens,url


        # the code for def tweettext2features(tweet_text):
        features=dict()

        tokens,url=tweets2tokens(tweet_text)

        punc_dict=punctuationanalysis(tweet_text)
        features.update(punc_dict)
        features['negativewordcount']=(negativewordcount(tokens))
        features['positivewordcount']=(positivewordcount(tokens))
        features['sentimentscore']=(sentimentscore(tweet_text))
        # print("features",features)
        return features

    def export(self):
        fn = "%s/%s.csv" % (self.output_dir, self.event)
        df = pd.DataFrame(data=self.data)
        df.to_csv(fn, index=False)
        return fn
    
    def datestr_to_tmsp(self, datestr):
        """ Converts Twitter's datetime format to Unix timestamp 

        Param:
            - datestr: datetime string, e.g. Mon Dec 10 4:12:32.33 +7000 2018
        Return: Unix timestamp
        """
        return to_unix_tmsp([parse_twitter_datetime(datestr)])[0]

def pheme_to_csv(event, Parser=Tweets, output="data/tweets"):
    """ Parses json data stored in directories of the PHEME dataset into a CSV file.
    
    Params:
        - event: Name fake news event and directory name in PHEME dataset
    
    Return: None
    """
    start = time.time()
    data = Parser(event, output_dir=output)
    dataset = "raw/pheme-rnr-dataset"
    thread_number = 0         
    for category in os.listdir("%s/%s" % (dataset, event)):
        print('event:',event,'category:',category,category=='rumours')
        for thread in os.listdir("%s/%s/%s" % (dataset, event, category)):
            with open("%s/%s/%s/%s/source-tweet/%s.json" % (dataset, event, category, thread, thread)) as f:
                tweet = json.load(f)
            data.append(tweet, category, thread, True)
            thread_number += 1
            for reaction in os.listdir("%s/%s/%s/%s/reactions" % (dataset, event, category, thread)):
                with open("%s/%s/%s/%s/reactions/%s" % (dataset, event, category, thread, reaction)) as f:
                    tweet = json.load(f)
                data.append(tweet, category, thread, False)
    fn = data.export()
    print("%s was generated in %s minutes" % (fn, (time.time() - start) / 60))
    return None

if __name__ == "__main__":
    print("Running %s to parse %s" % (argv[0], argv[1]))
    if(argv[1]=="all"):
        events=[
            "germanwings-crash",
            "sydneysiege",
            "ottawashooting",
            "ferguson",
            "charliehebdo",
        ]
        dataset = "../raw/pheme-rnr-dataset"
        processes=[]
        for event in events:
            p=Process(target=pheme_to_csv,args=(event,))
            p.start()
            processes.append(p)
            #pheme_to_csv(event)
        for p in processes:
            p.join()
            
    else:
        pheme_to_csv(argv[1])
