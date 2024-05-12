import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import webbrowser

output_shape = (None, 9)

# Load the model and labels
model = tf.keras.models.load_model("model.h5")
labels = np.load("labels.npy")

# Initialize Mediapipe objects
holistic = mp.solutions.holistic
hands = mp.solutions.hands
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Define the EmotionDetector class
class EmotionDetector:
    def __init__(self):
        print("EmotionDetector initialized")
        self.holistic = holistic.Holistic()
        self.drawing = mp.solutions.drawing_utils

    def recv(self, frame):
        print("Received frame")
        try:
            frm = frame.to_ndarray(format="bgr24")
            frm = cv2.flip(frm, 1)  # Flipping the frame from left to right
            res = self.holistic.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

            lst = []

            # Storing Landmark data
            if res.face_landmarks:
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = labels[np.argmax(model.predict(lst))]
            detected_emotion = pred.decode('utf-8')
            
            st.session_state.detected_emotion = detected_emotion  # Store detected emotion in session state

            cv2.putText(frm, detected_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            self.drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
            self.drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
            self.drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

            return av.VideoFrame.from_ndarray(frm, format="bgr24")
        except Exception as e:
            print("Error processing frame:", e)
            return None

# Streamlit app layout
st.set_page_config(page_title="Mood Sick 🎵", page_icon="🎵", layout="wide")

st.title("📜 Mood Sick")

col1, col2, col3 = st.columns([2, 2, 7])

with col1:
    st.image("D:/Moodsick/images/logo2.svg", width=500, use_column_width=False)

with col3:
    st.markdown('<p class="intro-text" style="color: #ff5733;font-weight: bold;">📌 Mood Sick is an emotion detection-based music recommendation system.<br>📌 To get recommended songs, start by allowing the microphone and camera access.</p>', unsafe_allow_html=True)

lang = st.text_input("Enter your preferred language")
artist = st.text_input("Enter your preferred artist")

# Video stream
if lang and artist:
    webrtc_streamer(
        key="example", 
        desired_playing_state=True,
        video_processor_factory=EmotionDetector
    )

# Recommendation button
# Recommendation button
btn = st.button("Recommend music 🎵")

if btn:
    if "detected_emotion" not in st.session_state:
        st.session_state.detected_emotion = None  # Initialize the session state if it doesn't exist

    detected_emotion = st.session_state.detected_emotion

    if detected_emotion is None:
        st.warning("Please let me capture your emotion first!")
    else:
        webbrowser.open(
            f"https://www.youtube.com/results?search_query={lang}+{detected_emotion}+songs+{artist}"
        )


st.write('Made with ❤️ by Pratham & Ridhima')
