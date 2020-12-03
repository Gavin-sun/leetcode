# Count Vectors + RidgeClassifier

import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score



train_df = pd.read_csv(r'E:\git_code\Machine_learning\py\新闻分类\rs\train_set.csv', sep='\t', nrows=2,encoding='utf-8')


vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])

# clf = RidgeClassifier()
# clf.fit(train_test[:10000], train_df['label'].values[:10000])
#
# val_pred = clf.predict(train_test[10000:])
# print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))

# tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=3200)
# train_test = tfidf.fit_transform(train_df['text'])
#
#
#
# clf = RidgeClassifier()
# clf.fit(train_test[:10000], train_df['label'].values[:10000])
#
# val_pred = clf.predict(train_test[10000:])
# print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))