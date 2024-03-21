import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def load_data(directory):
    npy_files = [file_name for file_name in os.listdir(directory) if file_name.endswith(".npy") and file_name != "labels.npy"]
    if not npy_files:
        print("\nNo suitable files found in the directory.")
        return None, None
    X, y = None, None
    labels = []
    for idx, file_name in enumerate(npy_files):
        print(f"Processing file {idx+1}/{len(npy_files)}: {file_name}")
        try:
            data = np.load(os.path.join(directory, file_name))
            if X is None:
                X = data
                y = np.full((data.shape[0], 1), idx)
            else:
                X = np.concatenate((X, data))
                y = np.concatenate((y, np.full((data.shape[0], 1), idx)))
            labels.append(file_name.split('.')[0])
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
    return X, y, labels

def preprocess_data(X, y):
    # Converting Integer to binary class matrix
    y = to_categorical(y)
    # Shuffling data to avoid clustering
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def build_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    middle_layer1 = Dense(1024, activation="relu")(input_layer)
    dropout1 = Dropout(0.5)(middle_layer1)
    middle_layer2 = Dense(512, activation="relu")(dropout1)
    dropout2 = Dropout(0.5)(middle_layer2)
    middle_layer3 = Dense(256, activation="relu")(dropout2)
    dropout3 = Dropout(0.5)(middle_layer3)
    output_layer = Dense(num_classes, activation="softmax")(dropout3)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['acc'])
    return model

def train_model(model, X_train, y_train, epochs=50):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, callbacks=[early_stop], verbose=1)

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

def save_model(model, labels):
    model.save("model.h5")
    np.save("labels.npy", np.array(labels))

def main():
    directory = "./data"  # Change this to your data directory
    X, y, labels = load_data(directory)
    if X is None:
        return
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = build_model(X_train.shape[1:], len(labels))
    train_model(model, X_train, y_train, epochs=80)  # Increase the number of epochs
    evaluate_model(model, X_test, y_test)
    save_model(model, labels)

if __name__ == "__main__":
    main()
