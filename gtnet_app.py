# GT-Net Application for Gaze Vector Fusion
#from cProfile import label
import os
import pickle
#from pydoc import text
from tkinter import END, filedialog, Tk, Text
import cv2
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout
from keras.callbacks import ModelCheckpoint


global Gaze_train, Gaze_test
global X, Y, Gaze, X_train, X_test, y_train, y_test


# Initialize Tkinter root and Text widget for logging
root = Tk()
root.title("GT-Net Application Log")
text = Text(root, height=20, width=100)
text.pack()
from tkinter import Button

# Buttons will be created after all function definitions.


# ======== Modified uploadDataset() ========
def uploadDataset():
    global filename, X, Y, Gaze
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END, filename + " loaded\n\n")

    # Ensure 'model' directory exists
    if not os.path.exists('model'):
        os.makedirs('model')

    # If previously processed arrays exist, load them
    if os.path.exists('model/X.txt.npy') and os.path.exists('model/Y.txt.npy') and os.path.exists('model/Gaze.txt.npy'):
        text.insert(END, "Preprocessed data found. Loading cached .npy files...\n")
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        Gaze = np.load('model/Gaze.txt.npy')
    else:
        # Process images/videos and save
        X, Y, Gaze = [], [], []
        for root, dirs, files in os.walk(filename):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')) and 'Thumbs.db' not in file:
                    img = cv2.imread(os.path.join(root, file))
                    img = cv2.resize(img, (32, 32))
                    X.append(img.ravel())
                    gaze_vector = np.random.rand(3)
                    Gaze.append(gaze_vector)
                    label = getLabel(os.path.basename(root))
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        Gaze = np.asarray(Gaze)
        np.save('model/X.txt', X)
        np.save('model/Y.txt', Y)
        np.save('model/Gaze.txt', Gaze)
    text.insert(END, "Total images found in dataset : " + str(len(X)) + "\n")



# ======== Modified preprocess() ========
def preprocess():
    text.delete('1.0', END)
    global X, Y, Gaze

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if X.size == 0:
        text.insert(END, "No data loaded. Please upload a dataset first.\n")
        return

    original_shape = X.shape
    text.insert(END, f"Original shape of X: {original_shape}\n")

    # Normalize
    X = X.astype('float32') / 255

    # Shuffle
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Gaze = Gaze[indices]

    text.insert(END, "Dataset normalized and shuffled.\n")
    text.insert(END, f"Normalized shape of X: {X.shape}\n")
    text.insert(END, f"Gaze vector shape: {Gaze.shape}\n")
    text.insert(END, f"Labels sample: {np.unique(Y)}\n\n")

    # Show class distribution
    unique, count = np.unique(Y, return_counts=True)
    for i in range(len(unique)):
        text.insert(END, f"Class {unique[i]} → {count[i]} samples\n")
    
    plt.bar(unique, count)
    plt.xticks(unique)
    plt.xlabel('Locomotion Classes')
    plt.ylabel('Sample Count')
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.show()


# ======== Modified splitDataset() ========
def splitDataset():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, Gaze_train, Gaze_test, X, Y, Gaze
    if Y is None or len(Y) == 0:
        text.insert(END, "No labels found. Please preprocess the dataset first.\n")
        return
    Y = to_categorical(Y)
    X_combined = np.concatenate((X, Gaze), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_combined, Y, test_size=0.2)
    text.insert(END, "Dataset Train & Test Split Completed\n\n")
    text.insert(END, "Total Samples : " + str(X.shape[0]) + "\n")
    text.insert(END, "Features per Sample (Image+Gaze): " + str(X_combined.shape[1]) + "\n")
    text.insert(END, "Training Records : " + str(X_train.shape[0]) + "\n")
    text.insert(END, "Testing Records : " + str(X_test.shape[0]) + "\n")
    text.update_idletasks()

# ======== Modified runGENet() ========
def runGENet():
    global X_train, X_test, y_train, y_test, gtnet_model
    text.delete('1.0', END)
    if 'X_train' not in globals() or X_train is None or len(X_train) == 0:

        text.insert(END, "❌ Training data not ready. Make sure to select 'dataset' first.\n")

        return
    gtnet_model = Sequential()
    gtnet_model.add(InputLayer(input_shape=(X_train.shape[1],)))
    gtnet_model.add(Dense(units=512, activation='relu'))
    gtnet_model.add(Dense(units=256, activation='relu'))
    gtnet_model.add(Dropout(0.25))
    gtnet_model.add(Dense(units=y_train.shape[1], activation='softmax'))
    gtnet_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    if not os.path.exists("model/gt_weights.hdf5"):
        model_check_point = ModelCheckpoint(filepath='model/gt_weights.hdf5', verbose=1, save_best_only=True)
        hist = gtnet_model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        with open('model/gt_history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
    else:
        gtnet_model.load_weights("model/gt_weights.hdf5")

    predict = gtnet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("GT-Net", predict, y_test1)
def forecastLocomotion():
    global gtnet_model, labels

    # Ask user for a video file or webcam
    video_path = filedialog.askopenfilename(title="Select Video File (Cancel for Webcam)")
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Webcam

    if not cap.isOpened():
        text.insert(END, "Error: Cannot open video source.\n")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to 32x32 and flatten
        frame_resized = cv2.resize(frame, (32, 32))
        flattened = frame_resized.ravel().astype('float32') / 255

        # Simulate gaze vector
        gaze_vector = np.random.rand(3)

        # Combine image + gaze
        fused_input = np.concatenate((flattened, gaze_vector)).reshape(1, -1)

        # Predict
        prediction = gtnet_model.predict(fused_input)
        predicted_class = np.argmax(prediction)
        label_text = labels[predicted_class]

        # Display prediction on video frame
        cv2.putText(frame, f'Predicted: {label_text}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-Time Locomotion Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    
    # ======== Added calculateMetrics() definition ========
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import seaborn as sns
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100
    text.insert(END, algorithm + " Accuracy  :  " + str(a) + "\n")
    text.insert(END, algorithm + " Precision : " + str(p) + "\n")
    text.insert(END, algorithm + " Recall    : " + str(r) + "\n")
    text.insert(END, algorithm + " FScore    : " + str(f) + "\n\n")
    text.update_idletasks()

    conf_matrix = confusion_matrix(y_test, predict)
    plt.figure(figsize=(6, 3))
    ax = sns.heatmap(conf_matrix, annot=True, cmap="viridis", fmt="g")
    plt.title(algorithm + " Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show()
def getLabel(name):
    # Simple label encoding: assign a unique integer to each folder name
    if not hasattr(getLabel, "label_map"):
        getLabel.label_map = {}
    if name not in getLabel.label_map:
        getLabel.label_map[name] = len(getLabel.label_map)
    return getLabel.label_map[name]
#if __name__ == "__main__":
   # print("Launching GUI...")
    #root.mainloop()
from tkinter import simpledialog

def inputSource():
    global use_forecasting
    global X, Y, Gaze, X_train, X_test, y_train, y_test
    use_forecasting = False

    option = simpledialog.askstring("Input Type", "Type 'dataset' to upload training data, or 'video' for webcam/video:")
    
    if option is None:
        return
    
    if option.lower() == "dataset":
        uploadDataset()
        text.insert(END, "\n✔️ Dataset uploaded.\n")
        preprocess()
        text.insert(END, "\n✔️ Dataset preprocessed.\n")
        splitDataset()
        text.insert(END, "\n✔️ Dataset split completed.\n")
    elif option.lower() == "video":
        use_forecasting = True
        forecastLocomotion()
    else:
        text.insert(END, "Invalid input. Please type 'dataset' or 'video'.\n")



if __name__ == "__main__":
    input_btn = Button(root, text="Select Input Source", command=inputSource)
    run_btn = Button(root, text="Run GT-Net", command=runGENet)

    input_btn.pack(pady=10)
    run_btn.pack(pady=10)

    print("Launching GUI...")
    root.mainloop()

