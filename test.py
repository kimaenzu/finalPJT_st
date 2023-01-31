
#################################################################################
## import
import streamlit as st

## KoBERT
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import gluonnlp as nlp
import numpy as np
import datetime

import pandas as pd
from IPython.core.display import HTML, display
from IPython.core import display
from PIL import Image

import firebase_admin
from firebase_admin import credentials

#################################################################################
cred = credentials.Certificate("/content/content/firebase-adminsdk.json")
#firebase_admin.initialize_app(cred)

##// bucket 콜렉션
const bucket = firestore.collection("bucket");

##// bucket 콜렉션의 info 문서에 {name: 'duck';, height: 180} 데이터 추가.
##// 새로 만들거나 덮어쓰기
bucket.doc("info").set({name: 'duck', height: 180});

d = datetime.datetime.now()
TODAY = str(d)[:10]
calendar = [['sad','fear','happy','neutral','happy','surprise','neutral'],['angry','disgust','surprise','sad','sad','neutral','sad'],['happy','happy','happy','19','20','21','22'],['23','24','25','26','27','28','29'],['30','31','_','_','_','_','_']]
df = pd.DataFrame(calendar)
columns = ['MON','TUE','WED','THU','FRI','SAT','SUN']
df.columns = columns 

#################################################################################
## def 

## 일기를 작성받는 함수
def input_emotion():
  st.text("안녕하세요 00님! 오늘의 일기를 작성해주세요")
  message = ''
  message = st.text_area("일기 작성 칸") #### 작성칸 10줄 고정

  if st.button("기록", key='message'):
    result = message.title()

#    st.success(result)
    st.text(f"{TODAY} 기록이 완료됐습니다.")
    update_emo(predict(message))
    

## 일기쓴 날 스탬프 추가
def update_emo(emo):
  ## DB 업데이트로 내용이 바뀔 부분 : 임시로 DB를 전역변수로 선언하여 사용하고 있음
    global df
    df = df.replace(TODAY[-2:], emo)
    st.write(df)

## 달력에 이미지를 넣는 함수
def to_img_tag(path):
  return f'<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/{path}.png" width="15" >'
  st.write(df1)
  st.write(HTML(df1.to_html(escape=False,formatters= {'one':to_img_tag,'two':to_img_tag,'three':to_img_tag})))

## 달력 가져와서 출력하기 : 아이콘을 눌러서 일기를 볼 수 있을까?
def calendar_emo():
  ## 대충 만든 달력 ==========(달력 DB 가져오기 코드로 바뀔 것)
  # 임시 달력 DB
# emotions = {'공포' : 'fear', '놀람' : 'surprise', '분노' : 'angry', '슬픔' : 'sad', '중립' : 'neutral', '행복' : 'happy', '혐오' : 'disgust'}

#  st.dataframe(df)
  st.write(HTML(df.to_html(escape=False,formatters={'MON':to_img_tag,'TUE':to_img_tag,'WED':to_img_tag,'THU':to_img_tag,'FRI':to_img_tag,'SAT':to_img_tag,'SUN':to_img_tag})))

    ## , two=to_img_tag, three=to_img_tag 

    ##===============================================

def predict_img(emo):
  if emo == '공포가' :
    ## gif 해서 되면 gif로 (st.markdowm 사용)
    st.write(HTML('<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/fear.png">'))
  elif emo == '놀람이' :
    st.write(HTML('<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/surprise.png">'))
  elif emo == '분노가' :
    st.write(HTML('<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/angry.png">'))
  elif emo == '슬픔이' :
    st.write(HTML('<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/sad.png">'))
  elif emo == '중립이' :
    st.write(HTML('<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/neutral.png">'))
  elif emo == '행복이' :
    st.write(HTML('<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/happy.png">'))
  elif emo == '혐오가' :
    st.write(HTML('<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/disgust.png">'))

#################################################################################
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

## bertmodel의 vocabulary
bertmodel, vocab = get_pytorch_kobert_model()

## 모델 불러오기
model = torch.load('/content/drive/MyDrive/Colab Notebooks/감정분석기/models/7emotions_model.pt')

## 4. 데이터 전처리(토큰화, 정수 인코딩, 패딩)
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

## 8. 결과물 테스트
## 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

## 감정 예측
def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()
 
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("공포가")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람이")
            elif np.argmax(logits) == 2:
                test_eval.append("분노가")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 4:
                test_eval.append("중립이")
            elif np.argmax(logits) == 5:
                test_eval.append("행복이")
            elif np.argmax(logits) == 6:
                test_eval.append("혐오가")
        predict_img(test_eval[0])
        st.write(test_eval[0] + " 느껴집니다. (bert 문장으로 바뀔 예정)")

        return (test_eval[0][:-1])

#################################################################################

## Title
st.title("감정저장소")
## Header/Subheader

## 메뉴 선택
add_selectbox = st.sidebar.selectbox("무엇이 궁금하세요?",("감정기록", "과거의 감정", "감정그래프"))

## 감정기록
if add_selectbox == "감정기록":
   input_emotion()
#################################################################################


## 과거의 감정
if add_selectbox == "과거의 감정":
	## Header/Subheader
  st.header("==============================")
  st.subheader("과거의 감정")

	## 감정 스탬프
  st.text("감정 달력(1월)")
  calendar_emo()

	## Select Box
  # 가져올 DB
  dateDB1 = ["2023년 1월 1일","2023년 1월 2일","2023년 1월 3일","2023년 1월 4일","2023년 1월 5일"]
  date1 = st.selectbox("(1안) 일기를 선택하세요.", dateDB1)
  st.write(date1, " 일기를 불러왔습니다.")

  date2 = st.date_input("(2안) 일기를 선택하세요.", datetime.date(2019, 7, 6))
  st.write(date2, " 일기를 불러왔습니다.")

## 감정그래프
if add_selectbox == "감정그래프":
	## Header/Subheader
	st.header("==============================")
	st.subheader("감정그래프")

	## Text
	st.text("감정그래프 출력")

#################################################################################
