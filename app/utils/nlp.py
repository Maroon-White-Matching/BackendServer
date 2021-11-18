import re
import os
import json
import math
import nltk
import uuid
import pickle
import numpy as np
# import textdistance
import pandas as pd
from collections import Counter

nltk.data.path.append('app/lib/models/wordnet') 
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
genesis_ic = wn.ic(genesis, False, 0.0)
  
# Load data from pickle files
cEXT = pickle.load( open( "app/lib/models/cEXT.p", "rb"))
cNEU = pickle.load( open( "app/lib/models/cNEU.p", "rb"))
cAGR = pickle.load( open( "app/lib/models/cAGR.p", "rb"))
cCON = pickle.load( open( "app/lib/models/cCON.p", "rb"))
cOPN = pickle.load( open( "app/lib/models/cOPN.p", "rb"))
vectorizer_31 = pickle.load( open( "app/lib/models/vectorizer_31.p", "rb"))
vectorizer_30 = pickle.load( open( "app/lib/models/vectorizer_30.p", "rb"))

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

def apply_nlp(data):
    data = data.fillna('')
    dirty_text = data.iloc[:,[4,21,22,23,24,25]].copy()
    dirty_text = dirty_text.applymap(lambda s:preprocess(s))
    data['words']=dirty_text.sum(axis=1).astype(str)
    return data

def padding(data):
    fellow = data.loc[data['Role:'] == "Fellow"]
    coach = data.loc[data['Role:'] == "Coach"]
    num_fellow = len(fellow)
    num_coach = len(coach)

    diff = math.floor(num_fellow/num_coach)
    rem = num_fellow%num_coach
    
    coach = pd.concat([coach]*diff, ignore_index=True)
    
    if(rem>=1):
        last = coach.iloc[:rem]
        coach = coach.append([last], ignore_index=True)
    data = pd.concat([coach, fellow], ignore_index= "true")
    data['UID'] = ''
    uid = []
    for i in range(len(data['UID'])):
        x=uuid.uuid4()
        uid.append(x)
    data['UID']= pd.DataFrame(uid, columns=['UID'])
    
    return data

def df_column_uniquify(df):
    df_columns = df['Full Name (First Middle Last)']
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df['Full Name (First Middle Last)'] = new_columns
    return df

def unpickle(filename):
    open_first = open(filename, "rb")
    Dic1 = pickle.load(open_first)
    open_again = open(filename, "rb")
    Dic2 = pickle.load(open_again)
    open_first.close()
    open_again.close()
    return Dic1,Dic2

def predict_personality(text):
    sentences = re.split("(?<=[.!?]) +", text)
    text_vector_31 = vectorizer_31.transform(sentences)
    text_vector_30 = vectorizer_30.transform(sentences)
    EXT = cEXT.predict(text_vector_31)
    NEU = cNEU.predict(text_vector_30)
    AGR = cAGR.predict(text_vector_31)
    CON = cCON.predict(text_vector_31)
    OPN = cOPN.predict(text_vector_31)
    return [EXT[0], NEU[0], AGR[0], CON[0], OPN[0]]

def big5(data):
    score = []
    for row in data['words']:
        score.append(predict_personality(row))
    data[['EXT', 'NEU', 'AGR', 'CON', 'OPN']]= pd.DataFrame(score, columns=[['EXT', 'NEU', 'AGR', 'CON', 'OPN']])
    data = data.astype(str)
    return data

def KNN_dictionary(data):
    final = data
    final = final.applymap(lambda s:preprocess(s))
    # text = final['words']
    text = final['Briefly describe a leader or person you admire and why.'].map(str) + " " + final['What do you hope to gain/learn as a result of the fellow-coach relationship?'].map(str) + " " + final['A few words to describe myself are:'].map(str) + " " + final['What values (3-5) influence your leadership style the most?'].map(str) 
    word_kw = text.to_frame()
    # print(word_kw[0])
    UID_kw = final['UID'] 
    keywords_hash= dict()
    for i in range(len(word_kw)):
        keywords_hash[UID_kw[i]] = word_kw[0][i]
    data = calculate_keyword(keywords_hash, UID_kw, final)
    return data

def calculate_keyword(keywords_hash, UID_kw, final):
    x = KNN_NLC_Classifer()
    common_words = []
    for i in range(len(keywords_hash)):
        split_it = keywords_hash[UID_kw[i]].split()
        CounterVariable = Counter(split_it)
        most_occur = CounterVariable.most_common(1)
        if most_occur[0][0] not in common_words:
            common_words.append(most_occur[0][0])

    centroid = " ".join(common_words)
    print(common_words)

    origin_hash= dict()
    for i in range(len(keywords_hash)):
        origin_hash[UID_kw[i]] = 1-x.document_similarity(centroid, keywords_hash[UID_kw[i]]) #distance from all participants and origin
   
    # print("DONE!!")
    # print(origin_hash[UID_kw[0]])
    final["distance"] = ''
    for i in range(len(origin_hash)):
        final["distance"][i] = origin_hash[UID_kw[i]]
    return final

class KNN_NLC_Classifer():
    def __init__(self, k=1, distance_type = 'path'):
        self.k = k
        self.distance_type = distance_type

    # This function is used for training
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    # This function runs the K(1) nearest neighbour algorithm and
    # returns the label with closest match. 
    def predict(self, x_test):
        self.x_test = x_test
        y_predict = []

        for i in range(len(x_test)):
            max_sim = 0
            max_index = 0
            for j in range(self.x_train.shape[0]):
                temp = self.document_similarity(x_test[i], self.x_train[j])
                if temp > max_sim:
                    max_sim = temp
                    max_index = j
            y_predict.append(self.y_train[max_index])
        return y_predict
    def convert_tag(self, tag):
        """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
        tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
        try:
            return tag_dict[tag[0]]
        except KeyError:
            return None


    def doc_to_synsets(self, doc):
        tokens = word_tokenize(doc+' ')

        l = []
        tags = nltk.pos_tag([tokens[0] + ' ']) if len(tokens) == 1 else nltk.pos_tag(tokens)

        for token, tag in zip(tokens, tags):
            syntag = self.convert_tag(tag[1])
            syns = wn.synsets(token, syntag)
            if (len(syns) > 0):
                l.append(syns[0])
        return l 
    def similarity_score(self, s1, s2, distance_type = 'path'):
          s1_largest_scores = []

          for i, s1_synset in enumerate(s1, 0):
              max_score = 0
              for s2_synset in s2:
                  if distance_type == 'path':
                      score = s1_synset.path_similarity(s2_synset, simulate_root = False)
                  else:
                      score = s1_synset.wup_similarity(s2_synset)                  
                  if score != None:
                      if score > max_score:
                          max_score = score
              
              if max_score != 0:
                  s1_largest_scores.append(max_score)
          
          mean_score = np.mean(s1_largest_scores)
                 
          return mean_score 
    def document_similarity(self,doc1, doc2):
          """Finds the symmetrical similarity between doc1 and doc2"""

          synsets1 = self.doc_to_synsets(doc1)
          synsets2 = self.doc_to_synsets(doc2)
          
          return (self.similarity_score(synsets1, synsets2) + self.similarity_score(synsets2, synsets1)) / 2