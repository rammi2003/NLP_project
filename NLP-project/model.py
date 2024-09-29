import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

with open("rt-polarity.pos", 'r', encoding='latin-1') as f:
    positive_reviews = f.readlines()

with open("rt-polarity.neg", 'r', encoding='latin-1') as f:
    negative_reviews = f.readlines()

positive_labels = [1] * len(positive_reviews)
negative_labels = [0] * len(negative_reviews)

reviews = positive_reviews + negative_reviews
labels = positive_labels + negative_labels

train_reviews = reviews[:4000] + reviews[5331:5331+4000]
train_labels = labels[:4000] + labels[5331:5331+4000]

val_reviews = reviews[4000:4500] + reviews[5331+4000:5331+4500]
val_labels = labels[4000:4500] + labels[5331+4000:5331+4500]

test_reviews = reviews[4500:5331] + reviews[5331+4500:]
test_labels = labels[4500:5331] + labels[5331+4500:]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_reviews)
X_val = vectorizer.transform(val_reviews)
X_test = vectorizer.transform(test_reviews)

model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=500))

model.fit(X_train, train_labels)

y_pred_test = model.predict(X_test)

accuracy = accuracy_score(test_labels, y_pred_test)
f1 = f1_score(test_labels, y_pred_test)

tn, fp, fn, tp = confusion_matrix(test_labels, y_pred_test).ravel()

print(f"Accuracy: {accuracy}")
print(f"F1-score: {f1}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
