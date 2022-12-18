from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np
import streamlit as st
import requests
import json

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('keras_Model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r', encoding='UTF8').readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image1 = Image.open('img1.jpg').convert('RGB')
image2 = Image.open('img2.jpg').convert('RGB')
image3 = Image.open('img3.jpg').convert('RGB')
image4 = Image.open('img4.jpg').convert('RGB')    

city = "Seoul" #도시
apiKey = "b96a8e79f31875b718a1ee64e6bf2cc8"
lang = 'kr' #언어
units = 'metric' #섭씨 온도로 변
api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={apiKey}&lang={lang}&units={units}"

apiWeather = requests.get(api)
apiWeather = json.loads(apiWeather.text)
weather = apiWeather['weather'][0]['main'] #날씨 정보
temperature = apiWeather['main']['temp'] #온도 정보


st.write("현재 날씨 : "+weather)
st.write("현재 온도 : ",temperature)


#테스트할 옷 차림 선택    
option = st.selectbox('테스트할 옷 차림을 골라주세요',
                         ('반팔', '후드티', '코트', '롱패딩'))

#테스트 데이터에 선택한 옷 차림을 입력하고 사진 출력  
if option == "반팔":
    image = image1
    st.image(image1)
elif option == "후드티":
    image = image2
    st.image(image2)
elif option == "코트":
    image = image3
    st.image(image3)
elif option == "롱패딩":
    image = image4
    st.image(image4)
    
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array
# run the inference
prediction = model.predict(data)
index = np.argmax(prediction) #클래스 순서
class_name = class_names[index] #클래스 이름
confidence_score = prediction[0][index] #정확도

#현재 온도 23도 이상
if temperature >= 23 :
    if index == 0 :
        st.write("현재 날씨에 맞는 옷을 입으셨습니다!")
    elif index > 0 :
        st.write("현재 날씨에 비해 옷이 너무 덥습니다!")

#현재 온도 16 ~ 22
elif temperature < 22 and temperature >= 16 :
    if index < 1 :
        st.write("현재 날씨에 비해 옷이 너무 춥습니다!")
    elif index == 1 :
        st.write("현재 날씨에 맞는 옷을 입으셨습니다!")
    if index >  1 :
        st.write("현재 날씨에 비해 옷이 너무 덥습니다!")

#현재 온도 6 ~ 16
elif temperature < 16 and temperature >= 6 :
    if index < 2 :
        st.write("현재 날씨에 비해 옷이 너무 춥습니다!")
    elif index == 2 :
        st.write("현재 날씨에 맞는 옷을 입으셨습니다!")
    if index >  2 :
        st.write("현재 날씨에 비해 옷이 너무 덥습니다!")
        
#현재 온도 6 이하
elif temperature < 6 : 
    if index < 3 :
        st.write("현재 날씨에 비해 옷이 너무 춥습니다!")
    elif index == 3 :
        st.write("현재 날씨에 맞는 옷을 입으셨습니다!")
        
st.write('정확도 :')
bar = st.progress(0)
bar.progress(int((confidence_score)*100))

