# Importing Libraries
import re
import pandas as pd
import nltk, spacy, string
import en_core_web_sm
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

#nltk.download("stopwords")
stop_words = set(stopwords.words('english'))
lmtzr = nltk.WordNetLemmatizer()

#Reading data and pickel files
tfidf_vec = pkl.load(open('tfidf.pkl','rb')) 
model = pkl.load(open('XGB.pkl','rb')) 
recommend_model = pkl.load(open('user_final_rating.pkl','rb'))
df = pd.read_csv('C:/Users/KAUSTUBH PC/Downloads/Capstone SBPRS/sample30.csv')


def text_preprocessing(text):
    text = text.lower()
    text = re.sub('[0-9]','',text)
    text = re.sub(r'[^a-zA-Z\s]','',text).strip()
    sent = model(text)
    word_tokens = word_tokenize(str(sent))
    sentence = [lmtzr.lemmatize(word) for word in word_tokens if not word in stop_words]
    #sentence = [token.lemma_ for token in word_tokens if token not in set(stopwords.words('english'))]
    return " ".join(sentence)
    
    
#predicting the sentiment
def model_predict(text):
    tfidf_vector = tfidf_vec.transform(text)
    output = model.predict(tfidf_vector)
    return output  
    
#Recommend the products
def recommend_products(user):
    print(user)
    product_list = pd.DataFrame(recommend_model.loc[user].sort_values(ascending=False)[0:20])
    print(product_list.index.tolist())
    product_frame = df[df.id.isin(product_list.index.tolist())]
    output_df = product_frame[['id','reviews_text']]
    #print('output=',output_df['id'])
    output_df['Review'] = output_df['reviews_text'].apply(text_preprocessing)
    output_df['predicted_sentiment'] = model_predict(output_df['Review'])
    return output_df
    
def top5_products(df):
    total_product=df.groupby(['id']).agg('count')
    rec_df = df.groupby(['id','predicted_sentiment']).agg('count')
    rec_df=rec_df.reset_index()
    merge_df = pd.merge(rec_df,total_product['reviews_text'],on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x']/merge_df['reviews_text_y'])*100
    merge_df=merge_df.sort_values(ascending=False,by='%percentage')
    #print('merge',merge_df['id'])
    output_products = pd.DataFrame(merge_df['id'][merge_df['predicted_sentiment'] ==  1][:5])
    return output_products
