import numpy as np
import datetime
import streamlit as st
import pandas as pd
from PIL import Image
from IPython.core.display import HTML, display
from IPython.core import display

import inspect
import os
from google.cloud import firestore

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import gluonnlp as nlp

############ 
## var
d = datetime.datetime.now()
TODAY = str(d)[:10]
calendar = [['sad','fear','happy','neutral','happy','surprise','neutral'],['angry','disgust','surprise','sad','sad','neutral','sad'],['happy','happy','happy','19','20','21','22'],['23','24','25','26','27','28','29'],['30','31','_','_','_','_','_']]
df = pd.DataFrame(calendar)
columns = ['MON','TUE','WED','THU','FRI','SAT','SUN']
df.columns = columns 

############
## kobert
## GPU 설정
device = torch.device("cuda:0")

class BERTClassifier(nn.Module):
  def __init__(self,
              bert,
              hidden_size = 768,
              num_classes=7,
              dr_rate=None,
              params=None):
    super(BERTClassifier, self).__init__()
    self.bert = bert
    self.dr_rate = dr_rate
                 
    self.classifier = nn.Linear(hidden_size , num_classes)
    if dr_rate:
      self.dropout = nn.Dropout(p=dr_rate)
    
  def gen_attention_mask(self, token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
      attention_mask[i][:v] = 1
      return attention_mask.float()

  def forward(self, token_ids, valid_length, segment_ids):
    attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
    _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
    if self.dr_rate:
      out = self.dropout(pooler)
      return self.classifier(out)

############
## bertmodel의 vocabulary
bertmodel, vocab = get_pytorch_kobert_model()

 
try:
  st.write(os.listdir())
escept:
  st.write("error : os.listdir())
try:
  model = torch.load('./model/7emotions_model.pt')
except:
  st.write("error : './model/7emotions_model.pt'")

try:
  model = torch.load('https://github.com/kimaenzu/finalPJT_st/blob/main/model/7emotions_model.pt')
  st.write("success: 'https://github.com/kimaenzu/finalPJT_st/blob/main/model/7emotions_model.pt'")
except:
  st.write("error : 'https://github.com/kimaenzu/finalPJT_st/blob/main/model/7emotions_model.pt'")

try:
  model = torch.load('https://github.com/kimaenzu/finalPJT_st/model/7emotions_model.pt')
  st.write("success: 'https://github.com/kimaenzu/finalPJT_st/model/7emotions_model.pt'")
except:
  st.write("error : 'https://github.com/kimaenzu/finalPJT_st/model/7emotions_model.pt'")


