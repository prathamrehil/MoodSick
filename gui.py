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
        # Ensure each set of landmarks has size 340 by padding with zeros if necessary
        while len(lst) < 340:
            lst.extend([0.0, 0.0])
        return lst
    else:
        # If no landmarks detected, return a list of zeros with size 1020
        return [0.0] * 1020


model = tf.keras.models.load_model("model.h5")  # Load the model using TensorFlow
label = np.load("labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Create Tkinter window
window = tk.Tk()
window.title("Mood Detection")
window.geometry("900x700")
window.configure(bg="#f0f0f0")

# Function to capture frame and update GUI
def update_frame():
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Process face landmarks
    face_landmarks = process_landmarks(res.face_landmarks, res.face_landmarks.landmark[1]) if res.face_landmarks else [0.0] * 1020

    # Process left hand landmarks
    left_hand_landmarks = process_landmarks(res.left_hand_landmarks, res.left_hand_landmarks.landmark[8]) if res.left_hand_landmarks else [0.0] * 1020

    # Process right hand landmarks
    right_hand_landmarks = process_landmarks(res.right_hand_landmarks, res.right_hand_landmarks.landmark[8]) if res.right_hand_landmarks else [0.0] * 1020

    print("Face landmarks size:", len(face_landmarks))
    print("Left hand landmarks size:", len(left_hand_landmarks))
    print("Right hand landmarks size:", len(right_hand_landmarks))

    # Concatenate the feature vectors
    # Concatenate the feature vectors
    lst = np.concatenate((face_landmarks[:340], left_hand_landmarks[:340], right_hand_landmarks[:340]))


    print("Concatenated landmarks size:", len(lst))

    # Ensure lst has shape (1, 1020)
    lst = lst.reshape(1, 1020)

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
    window.after(3, update_frame)

# Function to handle submit button click
def submit():
    # Get the current detected emotion
    current_emotion = emotion_label.cget("text") #initialize
    # Extract the detected emotion from the label text
    detected_emotion = current_emotion.split(":")[1].strip() #happy
    # Get the preferred language and artist
    lang = lang_entry.get()
    artist = artist_entry.get()
    # Prepare the URL with the detected emotion, language, and artist
    url = f"https://www.youtube.com/results?search_query={lang}+{detected_emotion}+songs+{artist}"
    # Open the URL in a web browser
    webbrowser.open(url)

# Create label to display logo
logo_img = Image.open("images/MoodSick.png")
logo_img = logo_img.resize((400, 200), Image.LANCZOS)
logo_img = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(window, image=logo_img, bg="#f0f0f0")
logo_label.pack(pady=10)
# Create a frame for the left column
left_frame = tk.Frame(window, bg="#f0f0f0")
left_frame.pack(side=tk.LEFT, padx=10)

# Create label for preferred language
lang_label = tk.Label(left_frame, text="Enter your preferred language:", bg="#f0f0f0")
lang_label.grid(row=0, column=0, pady=5)
lang_entry = tk.Entry(left_frame)
lang_entry.grid(row=1, column=0, pady=5)

# Create label for preferred artist
artist_label = tk.Label(left_frame, text="Enter your preferred artist:", bg="#f0f0f0")
artist_label.grid(row=2, column=0, pady=5)
artist_entry = tk.Entry(left_frame)
artist_entry.grid(row=3, column=0, pady=5)

# Create submit button
submit_btn = tk.Button(left_frame, text="Submit", command=submit, bg="#4CAF50", fg="white", font=("Arial", 20, "bold"))
submit_btn.grid(row=4, column=0, pady=10)

# Create a frame for the right column
right_frame = tk.Frame(window, bg="#f0f0f0")
right_frame.pack(side=tk.RIGHT, padx=10)

# Create label to display detected emotion
emotion_label = tk.Label(right_frame, text="Detected Emotion: ", bg="#f0f0f0", font=("Arial", 20, "bold"))
emotion_label.pack(pady=5)

# Create label to display video feed
label_img = tk.Label(right_frame)
label_img.pack()

# Start updating frame
update_frame()

# Run the GUI
window.mainloop()
