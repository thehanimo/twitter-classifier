import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import pickle
# from xgboost import XGBClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import ConfusionMatrixDisplay
# from sklearn.model_selection import cross_val_score

#Load datasets
gossip_fn = pd.read_csv("/Users/seungpang/Fake News/flask-server/GC_fake.csv")
gossip_tn = pd.read_csv("/Users/seungpang/Fake News/flask-server/GC_real.csv")
politi_fn = pd.read_csv("/Users/seungpang/Fake News/flask-server/PF_fake.csv")
politi_tn = pd.read_csv("/Users/seungpang/Fake News/flask-server/PF_real.csv")

#concatenate fakenews
fakenews = np.concatenate((politi_fn, gossip_fn))
columns = ['id','news_url','title', 'tweet_ids']
fakenews = pd.DataFrame(fakenews, columns = columns)
fakenews = fakenews[['id','title']]
ones = np.ones(fakenews.shape[0])
fakenews = np.column_stack((fakenews, ones))
fakenews = pd.DataFrame(fakenews, columns = ['id','title','flag'])

#concatenate realnews
realnews = np.concatenate((politi_tn, gossip_tn))
realnews = pd.DataFrame(realnews, columns = columns)
realnews = realnews[['id','title']]
zeros = np.zeros(realnews.shape[0])
realnews = np.column_stack((realnews, zeros))
realnews = pd.DataFrame(realnews, columns = ['id','title','flag'])

#combine fakenews and realnews
allnews = np.concatenate((fakenews, realnews))
allnews = pd.DataFrame(allnews, columns = ['id','title','flag'])
a_list = list(range(0, 23196))
allnews['ind'] = a_list
allnews.set_index('ind', inplace=True)
allnews = shuffle(allnews)

#tf-idf vectorizer
tf_idf = TfidfVectorizer(analyzer='word', stop_words='english', strip_accents = "ascii")
X = tf_idf.fit_transform(allnews['title'])
pickle.dump(tf_idf, open('tfidf.pkl','wb'))
y = allnews['flag'].astype('int')
np.asarray(X)
np.asarray(y)

#split train and test
Xtr, Xts, ytr, yts = train_test_split(X, y, random_state = 0, test_size = 0.3, stratify=allnews.flag)

#SMOTE
X_res, y_res = SMOTE().fit_resample(Xtr, ytr)
np.asarray(X_res)
np.asarray(y_res)

#Logistic Regression - before SMOTE F1 0.55, after SMOTE 0.65
LR_Model = LogisticRegression()
LR_Model.fit(X_res, y_res)
pickle.dump(LR_Model, open('model.pkl','wb'))
yhat = LR_Model.predict(Xts)
print("F1: ", metrics.f1_score(yts, yhat))

# # Prediction
# text = ["Christmas festival likely to happen in Brooklyn"]
# model = pickle.load(open('model.pkl','rb'))
# vectorizer = pickle.load(open('tfidf.pkl','rb'))

# predictions = model.predict(vectorizer.transform(text))
# print(text)
# print("Predicted as: {}".format(predictions[0]))


# text = ["Christmas festival likely to happen in Brooklyn this year"]
# text_features = tf_idf.transform(text)
# predictions = LR_Model.predict(text_features)
# print(text)
# print("Predicted as: {}".format(predictions[0]))