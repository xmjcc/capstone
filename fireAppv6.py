# from ast import Forgi
import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# load model
fire_dict = {0:'fire', 1 :'normal'}
# load json and create model
# json_file = open('emotion_model1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# classifier = model_from_json(loaded_model_json)

with open("model.json", "r") as file:
    model_json = file.read()
classifier = model_from_json(model_json)
classifier.load_weights("weights.h5")



# load weights into new model
# classifier.load_weights("emotion_model1.h5")

#load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

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

class Faceemotion(VideoTransformerBase):
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
            output2 = str(max(prediction[0]))  
        # output2 = str(max(prediction[0]))        
      
      
        # cv2.putText(img, "fire", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if np.argmax(prediction) == 0:
                cv2.putText(img, output1, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, output2, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if np.argmax(prediction) == 1:
                cv2.putText(img, output1, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(img, output2, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            
        return img

def main():
    # Face Analysis Application #
    st.title("Real Time Fire Detection Application")
    activiteis = ["Home", "Webcam Fire Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Wenping Wang    
            Email : wenping.wang@dcmail.ca  
            [LinkedIn] (https://www.linkedin.com/in/benjamin-wang-07a2b723b/)""")
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

                 2. Real time fire detection.

                 """)
    elif choice == "Webcam Fire Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and start fire detection")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Mohammad Juned Khan using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at Mohammad.juned.z.khan@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
