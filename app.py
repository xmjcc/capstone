# from ast import Forgi
import numpy as np
import cv2
import streamlit as st

from keras.models import model_from_json
 

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# load model
fire_dict = {0:'fire', 1 :'normal'}

with open("model.json", "r") as file:
    model_json = file.read()
classifier = model_from_json(model_json)
classifier.load_weights("weights.h5")



# try:
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# except Exception:
#     st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x

st.title("Capstone-2 Project (Fire Detection)")
activities = ["Home", "Webcam Fire Detection", "Fire Detection On Picture"]
choice = st.sidebar.selectbox("Select Activity", activities)

firecutoff = st.sidebar.slider("Input Fire Detection Sensitivity",min_value=0.50,
        max_value=1.00,
        value=0.6,
        step=0.01,
    )




def main():
    
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Fire detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.

                 1. Real time fire detection using web cam feed.

                 2. Fire Detection On Picture.

                 Developed by Wenping Wang    
                 Email : wenping.wang@dcmail.ca  
                 LinkedIn: https://www.linkedin.com/in/benjamin-wang-07a2b723b

                 Project Instructor: Professor Macros Bittencourt

                 """)
    elif choice == "Webcam Fire Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and start fire detection")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=firevideo)

    elif choice == "Fire Detection On Picture":

        uploaded_image = st.sidebar.file_uploader("upload a picture to test, a JPG, JPEG or PNG file", type=["jpg","jpeg","png"])

        if uploaded_image:


            st.sidebar.info('Uploaded image:')
            st.sidebar.image(uploaded_image, width=240)
            image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  


            face = image
 
            img1 = cv2.resize(face, (224, 224))

            if np.sum([img1]) != 0:
               x = img_to_array(img1)
               x = np.expand_dims(x, axis=0)/255

               prediction = classifier.predict(x)
               fire_dict = {0:'fire', 1 :'normal'}
               maxindex = int(np.argmax(prediction))
               firepro = str(round(prediction[0][0],2))
               nofirepro = str(round(prediction[0][1],2))
               if maxindex == 0 and  max(prediction[0]) > firecutoff:
                   st.subheader(f"The pic is on fire")
                   st.subheader(f"The probability of fire is {firepro}")
               else:
                   st.subheader(f"The pic is normal")
                   st.subheader(f"The probability of normal is {nofirepro}")

    else:
        pass



class firevideo(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img1 = cv2.resize(img, (224,224))
        
        if np.sum([img1]) != 0:
            x = img1.astype('float') / 255.0
            x = img_to_array(x)
            x = np.expand_dims(x, axis=0)

            prediction = classifier.predict(x)
        
            # prediction = classifier.predict(roi)[0]
       
            maxindex = int(np.argmax(prediction))

            # print(maxindex)
            finalout = fire_dict[maxindex]
            output1 = str(finalout)


            r1 = round(max(prediction[0]),2)
            output2 = str(r1) 

            


            if maxindex == 0 and  max(prediction[0]) > firecutoff:
            # if maxindex == 0:    
                output1 = "fire" + " "+"at 2nd floor"   
                output2 = "probability is" +" "+ output2 
               
                cv2.putText(img, output1, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, output2, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                output1 = "normal"
                
                output2 = "probability is"+ " "+ str(round(prediction[0][1],2))
                cv2.putText(img, output1, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(img, output2, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            
        return img




if __name__ == "__main__":
    main()

 
