import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Set Streamlit page configuration for wide mode
st.set_page_config(page_title="Mood Sick 🎵", page_icon="🎵", layout="wide",)

# Apply custom CSS for a light-themed UI
# Update your custom_css variable
custom_css = """
body {
    background-color: #f8f9fa; /* Light theme background color */
    color: #333;
    font-family: 'Arial', sans-serif;
    margin-top: -20px;
}

.container {
    max-width: 100%; /* Adjust to 100% for full-width content */
    margin: auto;
}

h1 {
    color: #ff5733; /* Music Orange */
    font-size: 36px; /* Increase the font size for h1 */
}

h2 {
    color: #007bff; /* Blue color for h2 */
    font-size: 28px; /* Set the font size for h2 */
}

footer {
    text-align: center;
    margin-top: 20px;
    color: #777;
    font-size: 16px; /* Increase the font size for the footer */
}

#streamlit-footer {
    display: none;
}

#button-recommend {
    background-color: #4CAF50; /* Green */
    border: none;
    color: white;
    padding: 15px 30px; /* Adjust padding for button */
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 24px; /* Increase the font size for the button */
    margin: 20px 0;
    cursor: pointer;
    border-radius: 8px;
}

#button-recommend:hover {
    background-color: #45a049; /* Darker shade on hover */
}

.intro-text {
    font-size: 20px; /* Set the font size for the introductory text */
    color: #ffdea7; /* Adjust text color */
    margin-top: 20px;
}

.lang-artist-input {
    margin-bottom: 20px; /* Add margin to the language and artist input fields */
}

video {
    border-radius: 10px; /* Add border-radius to the video frame */
}

#video-container {
    background-color: #f2f2f2; /* Light gray background for the video container */
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}

.emotion-text {
    font-size: 24px;
    color: #333;
    margin-top: 20px;
}

#recommendation-container {
    margin-top: 20px;
}

#recommendation-link {
    color: #007bff; /* Blue color for the recommendation link */
    text-decoration: none;
}

#recommendation-link:hover {
    text-decoration: underline; /* Underline on hover */
}

#made-with-love {
    margin-top: 20px;
    font-size: 18px;
    color: #777;
    text-align: center;
}

"""

st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)


col1, col2, col3 = st.columns([2, 2, 7])  # Adjust the width as needed

with col1:
    st.image("D:\Moodsick\images\logo2.svg", width=500, use_column_width=False)

# with col2:
#     st.text("")  # Add an empty text element for spacing



with col3:
    st.title("📜")
    st.markdown('<p class="intro-text" style="color: #ff5733;font-weight: bold;">📌 Mood Sick is an emotion detection-based music recommendation system.<br>📌 To get recommended songs, start by allowing the microphone and camera access.</p>', unsafe_allow_html=True)

model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils

if "run" not in st.session_state:
    st.session_state["run"] = "true"
try:
    detected_emotion = np.load("detected_emotion.npy")[0]
except:
    detected_emotion = ""

if not detected_emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

class EmotionDetector:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)  # Flipping the frame from left to right
        res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

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

        pred = label[np.argmax(model.predict(lst))]

        print(pred)
        cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        np.save("detected_emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Enter your preferred language")
artist = st.text_input("Enter your preferred artist")

if lang and artist and st.session_state["run"] != "false":
    webrtc_streamer(
        key="key", desired_playing_state=True, video_processor_factory=EmotionDetector
    )

btn = st.button("Recommend music 🎵", key="button-recommend")

if btn:
    if not detected_emotion:
        st.warning("Please let me capture your emotion first!")
        st.session_state["run"] = "true"
    else:
        webbrowser.open(
            f"https://www.youtube.com/results?search_query={lang}+{detected_emotion}+songs+{artist}"
        )
        np.save("detected_emotion.npy", np.array([""]))
        st.session_state["run"] = "false"

st.write('Made with ❤️ by Pratham & Ridhima')

# Streamlit Customisation
st.markdown(""" <style>
header {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
