#pulls data from Reddit using their API

import praw
import csv
import sys
import codecs
import random

cutoff = 150
proportion = 0.75

sys.stdout = codecs.getwriter("iso-8859-1")(sys.stdout, 'xmlcharrefreplace') #deal with characters outside of ASCII encoding

trainWrite = open('train_data.csv', 'wb')

reddit = praw.Reddit('scrape1')
subreddit = reddit.subreddit('AskReddit')

fields = ['Title', 'isPopular']
trainW = csv.DictWriter(trainWrite, fieldnames=fields)
trainW.writeheader()

testW = csv.DictWriter(testWrite, fieldnames=fields)
testW.writeheader()

#randomly write to train or test data
for submission in subreddit.hot(limit=500):
    if submission.score >= cutoff:
        isPop = 1
    else:
        isPop = 0
    #if random.uniform(0,1) > proportion:
        #testW.writerow({'Title': submission.title.encode(sys.stdout.encoding, errors='replace'), 'isPopular': isPop})
    #else:
        #trainW.writerow({'Title': submission.title.encode(sys.stdout.encoding, errors='replace'), 'isPopular': isPop})
    trainW.writerow({'Title': submission.title.encode(sys.stdout.encoding, errors='replace'), 'isPopular': isPop})

trainWrite.close()
testWrite.close()
