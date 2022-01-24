# Importing Libraries
import re
import pandas as pd
import nltk, spacy, string
import en_core_web_sm
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from os.path import exists as file_exists
class Model: 
    spacy_model = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words('english'))
    lmtzr = nltk.WordNetLemmatizer()

    #Reading data and pickel files
    tfidf_vec = pkl.load(open('tfidf.pkl','rb')) 
    model = pkl.load(open('XGB.pkl','rb')) 
    recommend_model = pkl.load(open('user_final_rating.pkl','rb'))
    df = pd.read_csv('sample30.csv')

    def __init__(self):  
        if file_exists('clean_data.csv'):
            self.df = pd.read_csv('clean_data.csv')
        else:
            self.df['Review'] = self.df['reviews_text'].apply(self.text_preprocessing)  
            self.df.to_csv('clean_data.csv')

    def text_preprocessing(self,text):
        text = text.lower()
        text = re.sub('[0-9]','',text)
        text = re.sub(r'[^a-zA-Z\s]','',text).strip()
        sent = self.spacy_model(text)
        print('spacy loaded')
        word_tokens = word_tokenize(str(sent))
        print ('Tokanization complete')
        sentence = [self.lmtzr.lemmatize(word) for word in word_tokens if not word in self.stop_words]
        print ('lammatization complete')
        #sentence = [token.lemma_ for token in word_tokens if token not in set(stopwords.words('english'))]
        return " ".join(sentence)

    #df['Review'] = df['reviews_text'].apply(text_preprocessing)
           
    #predicting the sentiment
    def model_predict(self,text):
        tfidf_vector = self.tfidf_vec.transform(text)
        print('tfidf done')
        output = self.model.predict(tfidf_vector)
        print('predict done')
        return output  
        
    #Recommend the products
    def recommend_products(self,user):
        print(user)
        product_list = pd.DataFrame(self.recommend_model.loc[user].sort_values(ascending=False)[0:20])
        print(product_list.index.tolist())
        product_frame = self.df[self.df.id.isin(product_list.index.tolist())]
        output_df = product_frame[['id','reviews_text','Review', 'name']]
        output_df = output_df.dropna()
        output_df['predicted_sentiment'] = self.model_predict(output_df['Review'])
        return output_df
        
    def top5_products(self,df):
        total_product=df.groupby(['id']).agg('count')
        rec_df = df.groupby(['id','predicted_sentiment']).agg('count')
        rec_df=rec_df.reset_index()
        merge_df = pd.merge(rec_df,total_product['reviews_text'],on='id')
        merge_df['%percentage'] = (merge_df['reviews_text_x']/merge_df['reviews_text_y'])*100
        merge_df=merge_df.sort_values(ascending=False,by='%percentage')
        #print('merge',merge_df['id'])
        output_products = pd.DataFrame(merge_df['id'][merge_df['predicted_sentiment'] ==  1][:5])
        return output_products
