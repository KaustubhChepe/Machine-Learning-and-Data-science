# import libraties
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
#--- HTML Tag Removal
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk



class Recommendation:
    
    def __init__(self):
        nltk.data.path.append('./nltk_data/')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        self.data = pickle.load(open('data.pkl','rb'))
        self.user_final_rating = pickle.load(open('user_final_rating.pkl','rb'))
        self.model = pickle.load(open('naive_bayes_model.pkl','rb'))
        self.raw_data = pd.read_csv("sample30.csv")
        self.data = pd.concat([self.raw_data[['id','name','brand','categories','manufacturer']],self.data], axis=1)
        
        
    def getTopProducts(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        tfs=pd.read_pickle('tfidf.pkl')
        temp=self.data[self.data.id.isin(items)]
        X = tfs.transform(temp['Review'].values.astype(str))
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Postive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][['id', 'brand',
                            'categories', 'manufacturer', 'name']].drop_duplicates().to_html(index=False)


    def getTopProductsNew(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        tfs=pd.read_pickle('tfidf.pkl')
        temp=self.data[self.data.id.isin(items)]
        X = tfs.transform(temp['Review'].values.astype(str))
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Postive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][['id', 'brand',
                              'categories', 'manufacturer', 'name']].drop_duplicates().to_json(orient="table")

    def getUsers(self):
        s= np.array(self.user_final_rating.index).tolist()
        #print(s)
        return ''.join(e+',' for e in s) 

 

