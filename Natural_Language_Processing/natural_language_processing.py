# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# quoting = 3 -> ignor quotes

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
	# - leave only a-zA-Z characters, remove everything else,
	#   in place of the removed, add space (' '):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
	# - Lower (ex: Z -> z)
    review = review.lower()
	# - Splits the string into a list of words:
    review = review.split()
	# - Prepare to take the root of the words (loved, loving -> love), stemming:
    ps = PorterStemmer()
	# - Stem and remove the words, which are part of the irrelevant words ("stopwords"):
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
	# - Join the list back into a string:
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)