June, 18, 2022
<!-- # Capstone
one work for colba
ne work for visual studio code (might finally switch to colab, due to local computer cpu power)
isual studio code  the camera working but the vedio stream not smooth
google colab video working very smooth but need to figure out how to output the video for predicition
kivy installed, but running with error, to be fixed

ai part 90% done, need to figure out might change from kivy to other app for UI development if Kivy has problem. -->
model file larger than 100M, can not upload to github


July 18, 2022

Use streamlit to build firedetection APP.
Tested with browser, upload to github and record video

# Build and deploy to cloudurn

Command to build the application. PLease remeber to change the project name and application name
```
gcloud builds submit --tag gcr.io/strealitdemo/streamlit-fire  --project=strealitdemo
```

Command to deploy the application
```
gcloud run deploy --image gcr.io/capstone2022-358721/streamlit-fire --platform managed  --project=capstone2022-358721 --allow-unauthenticated