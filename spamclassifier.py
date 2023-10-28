import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.iloc[:, :2]
df.head()

Y=df['v1']
X=df.drop('v1', axis=1)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, stop_words):
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    words = [word for word in words if word.isalpha()]  # Remove non-alphabetic characters
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize
    return " ".join(words)

inp=input("Enter sentence to check in spam or not: ")
X.loc[X.index[-1], 'v2'] = inp
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
tfidf_features = cv.fit_transform(X['v2'])
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=cv.get_feature_names_out())#converting X to tfidf

inp=tfidf_df.iloc[-1]
X=tfidf_df.iloc[:-1]
Y=Y.iloc[:-1]


Y.replace(to_replace='spam', value=1, inplace=True)
Y.replace(to_replace='ham', value=0, inplace=True)


from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,Y, random_state=104, test_size=0.25, shuffle=True)

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=35)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy2 = accuracy_score(y_test, y_pred)

inp=[inp]

if(knn_model.predict(inp)[0]==1):
    print("spam message received")
else:
    print("not spam message")

print("This model predicts with an accuracy of",(accuracy2*100),"%")