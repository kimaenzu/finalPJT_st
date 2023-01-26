import streamlit as st

import urllib.request
import pandas as pd

import numpy as np
from numpy import dot
from numpy.linalg import norm

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')


t_data = pd.read_csv('/content/content/concat_QnA2.csv')

# t_data['embedding'] str -> numpy 형변환
a = []
for _ in range(125292):
  tmp = t_data['embedding'][_].replace('[', '').replace(']', '').replace('\n', '')
  s_to_n = np.fromstring(tmp, dtype='f', sep=' ')
  a.append(np.array(s_to_n, dtype='f'))

new_data = t_data[['Q','A']]
new_data['embedding'] = a

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def return_similar_answer(input):
    embedding = model.encode(input)
    new_data['score'] = new_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return new_data.loc[new_data['score'].idxmax()]['A']

day_log = st.text_area('일기를 입력하세요.', height=10)
st.write(return_similar_answer(day_log))
