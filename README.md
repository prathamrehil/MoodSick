MoodSick - Facial emotion detection based music recommendation system
![logo](https://github.com/prathamrehil/MoodSick/blob/main/images/logo3.png)

### About

This repository demonstrates an end-to-end pipeline for real-time Facial emotion recognition application along with recommending music based on detected emotions.

Done in three steps:

1. Face Detection: from the video source using OpenCV.
2. Emotion Recognition: using a model trained by using Mediapipe library.
3. Music Recommendation: Using detected emotion to create a search query on Youtube

The model is trained for 80 epochs and runs at about 91% accuracy.
![image](https://github.com/prathamrehil/MoodSick/blob/main/images/epochs.png)

### Features

1. Landing Page
   ![image](https://github.com/prathamrehil/MoodSick/blob/main/images/homepage1.png)

Minimalistic landing page filled with Light-theme.

2. Detection of various emotions like [Sad, Angry, Happy, Neutral, Surprise]
   ![image](https://github.com/prathamrehil/MoodSick/blob/main/images/emo2.png)

3. Detection of various gestures like [Hello, Thumbsup, Nope, Rock]
   ![image](https://github.com/prathamrehil/MoodSick/blob/main/images/ges1.png)

### Dependencies

This project depends on Python and following packages which can be easily installed through `requirements.txt` file by running the following command:
`pip install -r requirements.txt`

- Python 3.9.6
- pip 22.1.1
- streamlit 1.9.1
- streamlit-webrtc 0.37.0
- opencv-python 4.5.5.64
- mediapipe 0.8.10
