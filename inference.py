import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import matplotlib.pyplot as plt
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def process_landmarks(landmarks, reference_landmark=None):
    if landmarks:
        lst = []
        for lm in landmarks.landmark:
            lst.append(lm.x - reference_landmark.x if reference_landmark else lm.x)
            lst.append(lm.y - reference_landmark.y if reference_landmark else lm.y)
        return lst
    else:
        # Ensure the length matches the expected length of concatenated feature vectors
        return [0.0] * 1020  # Adjust the length to 1020


model = tf.keras.models.load_model(r"C:\Users\prath\Downloads\MoodSick-main\MoodSick-main\model.h5")  # Load the model using TensorFlow
label = np.load(r"C:\Users\prath\Downloads\MoodSick-main\MoodSick-main\labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
hol = holistic.Holistic()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# model.summary()
# output_shape = model.layers[-1].output_shape
# print("Output shape:", output_shape)

while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Process face landmarks
    face_landmarks = process_landmarks(res.face_landmarks, res.face_landmarks.landmark[1]) if res.face_landmarks else [0.0] * 42
    print("Length of face landmarks:", len(face_landmarks))

    # Process left hand landmarks
    left_hand_landmarks = process_landmarks(res.left_hand_landmarks, res.left_hand_landmarks.landmark[8]) if res.left_hand_landmarks else [0.0] * 42
    print("Length of left hand landmarks:", len(left_hand_landmarks))

    # Process right hand landmarks
    right_hand_landmarks = process_landmarks(res.right_hand_landmarks, res.right_hand_landmarks.landmark[8]) if res.right_hand_landmarks else [0.0] * 42
    print("Length of right hand landmarks:", len(right_hand_landmarks))

    # Concatenate the feature vectors
    lst = np.array(face_landmarks + left_hand_landmarks + right_hand_landmarks)

    # Calculate the length of the concatenated feature vectors
    total_length = len(lst)
    print("Total length of concatenated feature vectors:", total_length)

    # Calculate the required number of repetitions to reach the expected input shape (1020)
    repetitions = int(1020 / total_length)
    print("Number of repetitions:", repetitions)

    # Tile each feature vector accordingly
    lst = np.tile(lst, (repetitions, 1))

    # Ensure lst has shape (1, 1020)
    lst = lst.reshape(1, -1)

    # Ensure lst has float32 data type
    lst = lst.astype(np.float32)

    # Print the shape of lst
    print("Shape of lst:", lst.shape)

    # Predict the label
    pred = label[np.argmax(model.predict(lst))]

    print(pred)
    
    cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS if res.left_hand_landmarks else None)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS if res.right_hand_landmarks else None)

    cv2.imshow("output_image.jpg", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
