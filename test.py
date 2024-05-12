import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import tkinter as tk
import webbrowser
from PIL import Image, ImageTk

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def process_landmarks(landmarks, reference_landmark=None):
    if landmarks:
        lst = []
        for lm in landmarks.landmark:
            lst.append(lm.x - reference_landmark.x if reference_landmark else lm.x)
            lst.append(lm.y - reference_landmark.y if reference_landmark else lm.y)
        return lst
    else:
        return [0.0] * 42

model = tf.keras.models.load_model(r"C:\Users\prath\Downloads\MoodSick-main\MoodSick-main\model.h5")  # Load the model using TensorFlow
label = np.load(r"C:\Users\prath\Downloads\MoodSick-main\MoodSick-main\labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Create Tkinter window
window = tk.Tk()
window.title("Mood Detection")
window.geometry("800x600")

# Function to capture frame and update GUI
def update_frame():
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Process face landmarks
    face_landmarks = process_landmarks(res.face_landmarks, res.face_landmarks.landmark[1]) if res.face_landmarks else [0.0] * 42

    # Process left hand landmarks
    left_hand_landmarks = process_landmarks(res.left_hand_landmarks, res.left_hand_landmarks.landmark[8]) if res.left_hand_landmarks else [0.0] * 42

    # Process right hand landmarks
    right_hand_landmarks = process_landmarks(res.right_hand_landmarks, res.right_hand_landmarks.landmark[8]) if res.right_hand_landmarks else [0.0] * 42

    # Concatenate the feature vectors
    lst = np.array(face_landmarks + left_hand_landmarks + right_hand_landmarks)

    # Ensure lst has shape (1, 1020)
    lst = lst.reshape(1, -1) if lst.shape[0] == 126 else lst.reshape(1, 1020)

    # Ensure lst has float32 data type
    lst = lst.astype(np.float32)

    # Predict the label
    pred = label[np.argmax(model.predict(lst))]

    # Display the predicted emotion
    emotion_label.config(text="Detected Emotion: " + pred)

    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS if res.left_hand_landmarks else None)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS if res.right_hand_landmarks else None)

    # Convert frame to ImageTk format
    img = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)

    # Update label with the frame
    label_img.config(image=img)
    label_img.image = img

    # Repeat after 10 ms
    window.after(10, update_frame)

# Function to handle submit button click
def submit():
    # Get the current detected emotion
    current_emotion = emotion_label.cget("text")
    # Extract the detected emotion from the label text
    detected_emotion = current_emotion.split(":")[1].strip()
    # Prepare the URL with the detected emotion
    url = f"https://www.youtube.com/results?search_query={detected_emotion}+songs"
    # Open the URL in a web browser
    webbrowser.open(url)

# Create submit button
submit_btn = tk.Button(window, text="Submit", command=submit)
submit_btn.pack()

# Create label to display detected emotion
emotion_label = tk.Label(window, text="Detected Emotion: ")
emotion_label.pack()

# Create label to display video feed
label_img = tk.Label(window)
label_img.pack()

# Start updating frame
update_frame()

# Run the GUI
window.mainloop()
