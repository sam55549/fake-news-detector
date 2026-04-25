import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix,precision_score,recall_score,f1_score
import re
import nltk
nltk.download('stopwords')

true_df = pd.read_csv('true.csv')
fake_df = pd.read_csv('fake.csv')

true_df['label']=1
fake_df['label']=0
dataset = pd.concat([true_df,fake_df], axis = 0)
dataset = dataset.sample(frac =1).reset_index(drop=True)
dataset.isnull().sum()
dataset['content'] = dataset['title'] + " " + dataset['text']
x = dataset['content']
y = dataset['label']
vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
x = vectorizer.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size = 0.25)
model = LogisticRegression(max_iter =1000)
model.fit(x_train,y_train)
y_training_pred = model.predict(x_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_train,y_training_pred)*100)
print(accuracy_score(y_test,y_pred)*100)
print(confusion_matrix(y_test,y_pred))
def predictnews(news):
  vect = vectorizer.transform([news])
  prediction = model.predict(vect)
  if prediction[0]==0:
    print("fake news")
  else:
    print("true news")
predictnews("Donald Trump just couldn't wish all Americans")
import pickle 
pickle.dump(model,open('model.pkl','wb'))
pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
print("Model and vectorizer saved successfully!")