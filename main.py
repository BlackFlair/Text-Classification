import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.test import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

data = fetch_20newsgroups()

categories = data.target_names

train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)

labels = model.predict(test.data)

mat = confusion_matrix(test.target.labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')

def predict_category (s, train=train, model=model):
	pred = model.predict([s])
	return train.target_names(pred[0])


# Enter New Text Data Here To Predict

predict_category('Space station')