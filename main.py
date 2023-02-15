#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import jaro
from difflib import SequenceMatcher as SM
import pickle

def direct_match(a,b):
    a1=a.strip()
    b1=b.strip()
    a2=a1.replace(" ", "")
    b2=b1.replace(" ", "")
    if a2==b2:
        k=1
    else:
        k=0
    return k

def remove_prefix(a) :
    prefix = ["Ms", "MS", "Mrs", "MRS", "Mr", "MR", "DR", "Dr", "Prof", "Ms.", "MS.", "Mrs.", "MRS.", "Mr.", "MR.", "DR.", "Dr.", "Prof.","Miss","Miss."]
    person_split = a.split(" ")
    if person_split[0] in prefix:
        person_split.remove(person_split[0])
        people = " ".join(person_split)
    else:
        people=a
    return people

def scaling(a,b):
    len_a=len(a.split())
    len_b=len(b.split())
    if  len_a != len_b :
        modifier = abs(len_a-len_b)
    else :
        modifier = 0
    return modifier
def char_scaling(a,b):
    len_a=len(a)
    len_b=len(b)
    if  len_a != len_b :
        modifier = abs(len_a/len_b)
    else :
        modifier = 0
    return modifier

def scaling_percent(a,b):
    len_a=len(a.split())
    len_b=len(b.split())
    if  len_a != len_b :
        modifier = abs(len_a/len_b)
    else :
        modifier = 0
    return modifier
def char_scaling_percent(a,b):
    len_a=len(a)
    len_b=len(b)
    if  len_a != len_b :
        modifier = abs(len_a-len_b)
    else :
        modifier = 0
    return modifier
def exact_match(a,b):
    if a.lower()==b.lower():
        k=1
    else:
        k=0
    return k
def sub_string_check(a,b):
    if a in b:
        k=1
    else :
        k=0
    return k

def text_to_vector(text):
    corpus =[text]
    vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char")
    vector = vectorizer.fit_transform(corpus)
    vec=vector.todense()
    return vec

def str_to_int(z):
    list_1 = list()
    for x in z:
        k=(ord(x))
        list_1.append(k)
    j=''.join(map(str, list_1))
    return j


#def lenmatch(a,b)
    

import re, math
from collections import Counter

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
    # print vec1, vec2
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / float(denominator)

def text_to_vector(text):
    return Counter(WORD.findall(text))



def get_similarity(a, b):
    a = text_to_vector(a.strip().lower())
    b = text_to_vector(b.strip().lower())

    return get_cosine(a, b)
def count_substring(string, sub_string):

    length = len(string)
    counter = 0
    for i in range(length):
        for j in range(length):
            if string[i:j+1] == sub_string:
                counter +=1
    return counter

def namematch(name_one,name_two):
    match_score=fuzz.token_sort_ratio(name_one,name_two)
    return match_score

def name_transform(df_base3):
    df_base3['pan_name']=df_base3['pan_name'].astype(str)
    df_base3['cibil_name']=df_base3['cibil_name'].astype(str)
    df_base3['n1_char_count']=df_base3.apply(lambda x: len(x['pan_name']), axis=1)
    df_base3['n1_word_count']=df_base3.apply(lambda x: len(x['pan_name'].split()), axis=1)
    df_base3['n2_char_count']=df_base3.apply(lambda x: len(x['cibil_name']), axis=1)
    df_base3['n2_word_count']=df_base3.apply(lambda x: len(x['cibil_name'].split()), axis=1)
    df_base3['pan_vs_cibil_scaling'] = df_base3.apply(lambda x: scaling(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['pan_vs_cibil_scaling2'] = df_base3.apply(lambda x: char_scaling(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['fuzzyscore']=df_base3.apply(lambda x: namematch(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['cibil_vs_pan_scaling2'] = df_base3.apply(lambda x: char_scaling(x['cibil_name'], x['pan_name']), axis=1)
    df_base3['pan_in_cibil'] = df_base3.apply(lambda x: sub_string_check(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['cibil_in_pan'] = df_base3.apply(lambda x: sub_string_check(x['cibil_name'], x['pan_name']), axis=1)
    df_base3['word_diff']=df_base3['n1_word_count']-df_base3['n2_word_count']
    df_base3['char_diff']=df_base3['n1_char_count']-df_base3['n2_char_count']
    df_base3['exact_match']=df_base3.apply(lambda x: exact_match(x['cibil_name'], x['pan_name']), axis=1)
    df_base3['n1_char_scaling_percent']=df_base3.apply(lambda x: char_scaling_percent(x['cibil_name'], x['pan_name']), axis=1)
    df_base3['n1_word_scaling_percent']=df_base3.apply(lambda x: scaling_percent(x['cibil_name'], x['pan_name']), axis=1)
    df_base3['n2_char_scaling_percent']=df_base3.apply(lambda x: char_scaling_percent(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['n2_word_scaling_percent']=df_base3.apply(lambda x: scaling_percent(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['n2_char1_count']=df_base3.apply(lambda x: len(x['cibil_name'].split()[0]), axis=1)
    df_base3['n2_char2_count']=df_base3.apply(lambda x: (len(x['cibil_name'].split()[1]) if (len(x['cibil_name'].split()) >= 2) else 0), axis=1)
    df_base3['n2_char3_count']=df_base3.apply(lambda x: (len(x['cibil_name'].split()[2]) if (len(x['cibil_name'].split()) >= 3) else 0), axis=1)
    df_base3['n2_char4_count']=df_base3.apply(lambda x: (len(x['cibil_name'].split()[3]) if (len(x['cibil_name'].split()) >= 4) else 0), axis=1)
    df_base3['n1_char1_count']=df_base3.apply(lambda x: len(x['pan_name'].split()[0]), axis=1)
    df_base3['n1_char2_count']=df_base3.apply(lambda x: (len(x['pan_name'].split()[1]) if (len(x['pan_name'].split()) >= 2) else 0), axis=1)
    df_base3['n1_char3_count']=df_base3.apply(lambda x: (len(x['pan_name'].split()[2]) if (len(x['pan_name'].split()) >= 3) else 0), axis=1)
    df_base3['n1_char4_count']=df_base3.apply(lambda x: (len(x['pan_name'].split()[3]) if (len(x['pan_name'].split()) >= 4) else 0), axis=1)
    df_base3['pan_str_to_int'] = df_base3.apply(lambda x: str_to_int(x['pan_name']), axis=1)
    df_base3['cibil_str_to_int'] = df_base3.apply(lambda x: str_to_int(x['cibil_name']), axis=1)
    df_base3['pan_vs_cibil_fuzzyscore'] = df_base3.apply(lambda x: fuzz.token_sort_ratio(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['pan_vs_cibil_fuzzyscore1'] = df_base3.apply(lambda x: fuzz.ratio(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['pan_vs_cibil_fuzzyscore2'] = df_base3.apply(lambda x: fuzz.partial_ratio(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['pan_vs_cibil_fuzzyscore3'] = df_base3.apply(lambda x: fuzz.token_set_ratio(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['pan_vs_cibil_jaroscore'] = df_base3.apply(lambda x: jaro.jaro_winkler_metric(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['pan_vs_cibil_cosinescore'] = df_base3.apply(lambda x: get_similarity(x['pan_name'], x['cibil_name']), axis=1)
    df_base3['pan_vs_cibil_smscore'] = df_base3.apply(lambda x: SM(None, x['pan_name'], x['cibil_name']).ratio(), axis=1)
    df_base3['pan_vs_cibil_directmatch'] = df_base3.apply(lambda x: direct_match( x['pan_name'], x['cibil_name']), axis=1)
    df_base3['pan_str_to_int']=df_base3['pan_str_to_int'].astype(float)
    df_base3['cibil_str_to_int']=df_base3['cibil_str_to_int'].astype(float)
    df_base3['a_b_repeat']=df_base3.apply(lambda x: count_substring( x['pan_name'], x['cibil_name']), axis=1)
    df_base3['b_a_repeat']=df_base3.apply(lambda x: count_substring( x['cibil_name'], x['pan_name']), axis=1)
    return df_base3


def get_names(a,b):
    d = {'pan_name': [a], 'cibil_name': [b]}
    q = pd.DataFrame(data=d)
    return q

def get_model_files():    
    file = open("scaler_namematch.obj",'rb')
    scaler_namematch = pickle.load(file)
    file.close()
    file = open("name_similarity_model.obj",'rb')
    name_similarity_model = pickle.load(file)
    file.close()
    return scaler_namematch,name_similarity_model

def score_predict(d):
    from sklearn.preprocessing import StandardScaler
    check=name_transform(d)
    check1=check.drop(['pan_name','cibil_name','a_b_repeat','b_a_repeat'],axis=1)
    #scaler = StandardScaler()
    scaler_name,name_similarity_model=get_model_files()

    x_traincheck = scaler_name.transform(check1)
    pred=name_similarity_model.predict(x_traincheck)
    y_hats  = pd.DataFrame(pred)
    check["Prediction"] = y_hats.reset_index()[0]
    pred_prob=name_similarity_model.predict_proba(x_traincheck)

    return check,pred_prob

def final_score(c) :    
    if c['pan_vs_cibil_directmatch'].iloc[0].astype(int) == 1 :
        matching_score = 10
    elif c['pan_vs_cibil_cosinescore'].iloc[0] <= .0 :
        matching_score = 0
    elif c['pan_vs_cibil_cosinescore'].iloc[0] < .85 :
        matching_score1 = c['pan_vs_cibil_cosinescore'].iloc[0]
        matching_score=matching_score1*10
    else :
        matching_score = c["Prediction"].iloc[0]
    return matching_score

def complex_score(matching_score,pred_probabilities):
    proba  = pd.DataFrame(pred_probabilities)
    k=max(proba.iloc[0])
    if matching_score ==10 :
        complex_score=matching_score
    elif matching_score == 0:
        complex_score=matching_score
    else :
        complex_score = matching_score + (k/10)
    return proba,complex_score,k

def decrypt(a,b):
    import rsa
    import pickle
    
    
    with open('privatekey.pem', 'rb') as p:
        private_Key = rsa.PrivateKey.load_pkcs1(p.read())
    decMessage_a = rsa.decrypt(a, private_Key).decode()
    decMessage_b = rsa.decrypt(b, private_Key).decode()
    return(decMessage_a,decMessage_b)


# In[2]:


from fastapi import FastAPI
app = FastAPI()

@app.get("/name_match_score")
def name_similarity_score(Name_one :str,Name_two:str):
    
    a1=bytes.fromhex(Name_one)
    b1=bytes.fromhex(Name_two)
    a2,b2=decrypt(a1,b1)
    from uuid import uuid4
    unique_id = str(uuid4())
    z=get_names(a2,b2)
    check,predicted_probabilities=score_predict(z)
    mscore=final_score(check)
    proba,finalscore,yha=complex_score(mscore,predicted_probabilities)
    similarity_score = finalscore*10
    output={"score" : similarity_score,"requestId":unique_id}
    return (output)

# In[ ]:




