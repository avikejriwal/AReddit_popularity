#can we use the bag-of-words model to predict popularity of posts?

import numpy as np
import pandas as pd
from KaggleWord2VecUtility import KaggleWord2VecUtility

import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LR

from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.externals import joblib


train_file = 'train_data.csv'
outFile = 'predictions.csv'

train = pd.read_csv(train_file, header=0, delimiter= ',')

#parsing post titles...
clean_train = []
for post in train['Title']:
    clean_train.append(' '.join(KaggleWord2VecUtility.review_to_wordlist(post, remove_stopwords=True)))

###Vectorizing
vectorizer = TfidfVectorizer(max_features=1200, ngram_range = (1,4), sublinear_tf=True)
train_x = vectorizer.fit_transform(clean_train)

msk = np.random.rand(np.shape(train_x)[0]) < 0.8
val_x = train_x[~msk]
val_lab = train['isPopular'][~msk]
train_x = train_x[msk]
train_lab = train['isPopular'][msk]

###training
model = LR()
model.fit(train_x, train_lab)

#validating
p = model.predict_proba(val_x)[:,1]

#compute the ROC curve for the validation data

cutoffs = np.linspace(0,1,100)
sens = []
fp = []

for cutoff in cutoffs:
    #fill the confusion matrix
    cMatrix = [[0, 0], [0,0]]
    for i, pr in enumerate(p):
        if (pr>cutoff) == True: #pos predict.
            cMatrix[0][1-val_lab.values[i]] += 1
        if (pr>cutoff) == False: #neg predict
            cMatrix[1][1-val_lab.values[i]] +=1
    sens.append(float(cMatrix[0][0])/(cMatrix[0][0]+cMatrix[1][0]))
    fp.append(float(cMatrix[0][1])/(cMatrix[1][1]+cMatrix[0][1]))

area = auc(fp,sens,reorder=False)

fig = plt.figure()
plt.plot(fp, sens, color='red')
plt.plot([0, 1], [0,1], color='blue',linestyle=':')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel('FP rate')
plt.ylabel('TP rate')
title = '/r/AskReddit post popularity ROC; AUC = ' +  str(round(area,2))
plt.title(title)
fig.savefig('rocPop.png', bbox_inches='tight')

joblib.dump(model, 'popPredict.pkl')
