import numpy as np
import datetime
import streamlit as st
import pandas as pd
import HTML, display
import display
from PIL import Image
import inspect
import streamlit as st

import firebase_admin
from firebase_admin import credentials
############
## var
d = datetime.datetime.now()
TODAY = str(d)[:10]
calendar = [['sad','fear','happy','neutral','happy','surprise','neutral'],['angry','disgust','surprise','sad','sad','neutral','sad'],['happy','happy','happy','19','20','21','22'],['23','24','25','26','27','28','29'],['30','31','_','_','_','_','_']]
df = pd.DataFrame(calendar)
columns = ['MON','TUE','WED','THU','FRI','SAT','SUN']
df.columns = columns 

############
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
    st.text("KoBERT 감정예측 출력 보류")
#    update_emo(predict(message))
    

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
  st.write(HTML(df.to_html(escape=False,formatters={'MON':to_img_tag,'TUE':to_img_tag,'WED':to_img_tag,'THU':to_img_tag,'FRI':to_img_tag,'SAT':to_img_tag,'SUN':to_img_tag})))


## 예측감정 이미지 출력
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

############

## Title
st.title("감정저장소")
## Header/Subheader

## 메뉴 선택
add_selectbox = st.sidebar.selectbox("무엇이 궁금하세요?",("감정기록", "과거의 감정", "감정그래프"))

## 감정기록
if add_selectbox == "감정기록":
   input_emotion()

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

print(inspect.getfile(HTML))
print(inspect.getfile(firebase-admin))


## 감정그래프
if add_selectbox == "감정그래프":
	## Header/Subheader
	st.header("==============================")
	st.subheader("감정그래프")

	## Text
	st.text("감정그래프 출력")
