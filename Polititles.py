import praw
import pandas as pd
import progressbar
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV




def model_score(model):
    return accuracy_score(model.predict(X_test), y_test)


data = pd.read_csv('data.csv')
# gathers only the titles of each post
all_titles = np.array(data['title'])

# use count vectorizer to create a dictionary of the words (and pairs of words) used.
count = CountVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')
bag = count.fit_transform(all_titles)
vocabulary = count.vocabulary_

# create the x_array of just words used (without the posts' scores)
x_array_sans_score = np.array(bag.toarray())
x_array_sans_score

# Transform bag of words
tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
x_array_sans_score = tfidf.fit_transform(count.fit_transform(all_titles)).toarray()
x_array_sans_score

# adds in a score of the post to the end of the bag of words array
x_array = []
for i, sentence in enumerate(bag.toarray()):
    score_to_append = 0
    if data['score'][i] > 10000:
        score_to_append = 10
    else:
        score_to_append = data['score'][i] * .001
    x_array.append(np.append(sentence, score_to_append))
x_array = np.array(x_array)

# create y corresponding to the subreddit
y_array = []
for post in data['subreddit']:
    if post == 'Conservative':
        y_array.append(1)
    else:
        y_array.append(0)
y_array = np.array(y_array)

X_train, X_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.33, random_state=1)

adaline = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))

ppn = Perceptron(eta0=0.1, random_state=1)

adaline.fit(X_train, y_train)

ppn.fit(X_train, y_train)

adaline_score = model_score(adaline)
ppn_score = model_score(ppn)

# Fine tune the SVM model using grid search
'''
gridsvm = SVC()
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']},
]
clf = GridSearchCV(gridsvm, param_grid)

clf.fit(X_train, y_train)
'''

from sklearn.svm import SVC

# initialize SVM model with tweaked hyperparameters (gamma = .015, C = 10)
svm = SVC(kernel='rbf', random_state=5, gamma=.015, C=10)

import pickle

loaded_svm = pickle.load(open("svm_90_c10,gamma015", 'rb'))


# model_score(loaded_svm)

def predict_inputs(input_titles, input_scores, model):
    count2 = CountVectorizer(max_features=5000, vocabulary=vocabulary, ngram_range=(1, 3), stop_words='english')
    input_bag = count2.fit_transform(input_titles)
    input_X = []

    for i, sentence in enumerate(input_bag.toarray()):
        score_to_append = 0
        if input_scores[i] > 10000:
            score_to_append = 10
        else:
            score_to_append = input_scores[i] * .001
        input_X.append(np.append(sentence, (score_to_append)))

    input_X = np.array(input_X)
    output = []
    for prediction in model.predict(input_X):
        if prediction == 1:
            output.append("Conservative")
        else:
            output.append("Bipartisan")
    return output


input_titles = ['U.S. Used Patriot Act to Gather Logs of Website Visitors. A disclosure sheds new light on a high-profile national security law as lawmakers prepare to revive a debate over it in the Biden administration.',
                'MSNBC’s Joy Reid Just Made Her Most Embarrassing Gaffe Yet',
                'President Obama officially endorses Joe Biden']
input_scores = [4000, 599, 9800]
print(predict_inputs(input_titles,input_scores,loaded_svm))