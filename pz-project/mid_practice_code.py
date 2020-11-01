import streamlit as st
import pandas as pd

# Get the data
df = pd.read_csv("https://github.com/Redwoods/Py/raw/master/pdm2020/my-note/py-pandas/data/diabetes.csv")

st.subheader('Data Information:')
# Show the data as a table (you can also use st.write(df))
st.dataframe(df)
# Get statistics on the data
st.write(df.describe())

# Show the data as a chart.
chart = st.line_chart(df)

## mid-term practice
## EDA of diabetes.csv
# Your code here !!
 
## Title
st.title('Streamlit Tutorial: start')
## Header/Subheader
st.header('This is header')
st.subheader('This is subheader')
## Text
st.text("Hello Streamlit! 이 글은 튜토리얼 입니다.")
 
## Markdown
st.header('Markdown')
st.markdown('# header-1')
st.markdown('## header-2')
st.markdown('###### header-6')
 
# list
st.header('List')
st.markdown('- list1')
st.markdown('- list2')
st.markdown('- list3\n'
            '   * inner list1\n'
            '   * inner list2\n'
            '          - inner_inner list\n')
 
## Latex
st.header('latex')
st.latex(r'\alpha_n, \beta, \gamma, \codts, \omega')
st.latex(r"Y = \alpha + \beta X_i")
## Latex-inline
st.markdown(r"회귀분석에서 오차는 다음과 같습니다 $e_i = y_i - \hat{y}_i$")
## Mean squared error
st.subheader("Mean squared error")
st.markdown(r"# $MSE = \frac{1}{N-1} \sum_{i=1}^{N-1} (y_i - \hat{y}_i)^2$")
 
## Message
st.info("End of first note: **text, markdown, latex**!")
 
## streamlit messages
st.success("Successful")
st.info("Information!")
st.warning("This is a warning")
st.error("This is an error!")
st.exception("NameError('Error name is not defined')")

## Load data
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris['target']
iris_df['target'] = iris_df['target'].apply(lambda x: 'setosa' if x == 0 else ('versicolor' if x == 1 else 'virginica'))
# 이까지는 데이터 프레임으로 구성한거고
## Return table/dataframe
# 여기서부터는 웹, streamlit으로 웹으로 보여줌.
# table
st.table(iris_df)  #head())

# dataframe
st.dataframe(iris_df)
 
st.write(iris_df) # write-> text로 출력 dataframe이랑 비슷함

st.header('Load diabetes data')
diabets_df = pd.read_csv('https://github.com/Redwoods/Py/raw/master/pdm2020/my-note/py-pandas/data/diabetes.csv')
st.dataframe(diabets_df)

## Show image
## PIL : Python Image Library
from PIL import Image
img = Image.open("files/example_cat.jpeg")
st.image(img, width=400, caption="Image example: Cat")
 
## Show videos
vid_file = open("files/example_vid_cat.mp4", "rb").read()
st.video(vid_file, start_time=2)
 
## Play audio file.
audio_file = open("files/loop_w_bass.mp3", "rb").read()
st.audio(audio_file, format="audio/mp3", start_time=10)

st.header("Load local image from PC")
## Load images
# Types of images
imgTypes = ["png", "jpg"]
 
st.info("Upload source image on Streamlit")
st.set_option('deprecation.showfileUploaderEncoding', False)  # 옵션 활성화(이걸 해야지 올려짐)
 
source_img_buf = st.file_uploader("Upload source image", type=imgTypes, key='src')
 
if source_img_buf is not None:
    source_img = Image.open(source_img_buf)
    #### Show image
    st.image(source_img)

## Widget
## Checkbox
if st.checkbox("Show/Hide"):
    st.text("체크박스가 선택되었습니다.")
    st.success("체크박스 선택 완료!");
 
st.markdown("* * *")  # 구분선
 
## Radio button
status = st.radio("Select status.", ("Active", "Inactive"))
if status == "Active":
    st.success("활성화 되었습니다.")
else:
    st.warning("비활성화 되었습니다.")
 
 
st.markdown("* * *")
 
 
# Select Box (ex)
occupation = st.selectbox("직군을 선택하세요.",
                          ["Backend Developer",
                           "Frontend Developer",
                           "ML Engineer",
                           "Data Engineer",
                           "Database Administrator",
                           "Data Scientist",
                           "Data Analyst",
                           "Security Engineer"])
st.write("당신의 직군은 ", occupation, " 입니다.")
 
 
st.markdown("* * *")
 
 
## MultiSelect
location = st.multiselect("선호하는 유투브 채널을 선택하세요.",
                          ("운동", "IT기기", "브이로그",
                           "먹방", "반려동물", "맛집 리뷰"))
st.write(len(location), "가지를 선택했습니다.")
 
 
st.markdown("* * *")
 
 
## Buttons
if st.button("About"):
    st.text("Streamlit을 이용한 튜토리얼입니다.")
 
 
st.markdown("* * *")
 
 
# Text Input
first_name = st.text_input("이름을 입력하세요.", "Type Here ...")
if st.button("Submit", key='first_name'):
    result = first_name.title()
    st.success(result)
 
 
# Text Area
message = st.text_area("메세지를 입력하세요.", "Type Here ...")
if st.button("Submit", key='message'):
    result = message.title()
    st.success(result)
 
 
st.markdown("* * *")
 
 
## Date Input
import datetime
today = st.date_input("날짜를 선택하세요.", datetime.datetime.now())
the_time = st.time_input("시간을 입력하세요.", datetime.time())
 
 
st.markdown("* * *")
 
 
# Display Raw Code - one line
st.subheader("Display one-line code")
st.code("import numpy as np")
 
# Display Raw Code - snippet
st.subheader("Display code snippet")
with st.echo(): # 두 줄 이상의 코드를 작성
    # 여기서부터 아래의 코드를 출력합니다.
    import pandas as pd
    df = pd.DataFrame()
    df.head()
 
 
 
## Display JSON
st.subheader("Display JSON")
st.json({'name' : '민수', 'gender':'male', 'Age': 29})
 
 
st.markdown("* * *")
 
 
## Sidebars
st.sidebar.header("사이드바 메뉴")
st.sidebar.selectbox("메뉴를 선택하세요.",
                    ["데이터",
                     "EDA",
                     "코드"])
st.sidebar.date_input("날짜를 선택하세요.", datetime.datetime.now(), key='new') # 위에서 선언해주었기 때문에 key 값을 다르게 만들어줘야 함.
 
# streamlit charts
# Plotting
# load iris.csv
# https://github.com/Redwoods/Py/raw/master/pdm2020/my-note/py-pandas/data/iris.csv
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
iris_df = pd.read_csv("https://github.com/Redwoods/Py/raw/master/pdm2020/my-note/py-pandas/data/iris.csv")
st.subheader('dataframe of iris data')
st.dataframe(iris_df)

st.subheader("Matplotlib/Pandas로 차트 그리기")
iris_df[iris_df['variety']=='Virginica']['petal.length'].hist()
st.pyplot() # 그래프 객체를 pyplot에 넣어줘야하는데 116번을 추가해줘야함.

import matplotlib.pyplot as plt
import numpy as np
# import io
from PIL import Image

# st.set_option('deprecation.showfileUploaderEncoding', False)

#### Load images
# Types of images
imgTypes = ["png", "jpg"]

st.info("Upload source image on Streamlit")
source_img_buf = st.file_uploader("Upload source image", type=imgTypes, key='src')

if source_img_buf is not None:
    source_img = Image.open(source_img_buf)
    st.success("Source image uploaded!")

st.info("Upload style image on Streamlit")
style_img_buf = st.file_uploader("Upload style image", type=imgTypes, key='style')
if style_img_buf is not None:
    style_img = Image.open(style_img_buf)
    st.success("Style image uploaded!")

st.info("Check here to see source and style images:")
if st.checkbox("Show source and style images", key='raw'):
  fig = plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.imshow(source_img)
  plt.title('Source Image')
  plt.subplot(1, 2, 2)
  plt.imshow(style_img)
  plt.title('Style Image')
  st.pyplot(fig)
  st.success("Completed!")

# New way to show images in columns (New features of streamlit)
st.info("Check here to show images:source, style in cloumns")
if st.checkbox("Show source and style images", key='column'):
    col1, col2 = st.beta_columns(2)
    col1.header("Source image")
    col1.image(source_img, use_column_width=True)
    col2.header("Style image")
    col2.image(style_img, use_column_width=True)
    st.success("Column images, Completed!")