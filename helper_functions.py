import tensorflow as tf
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_recall_fscore_support

import os
import time
from datetime import datetime

def predict_img(img_path, img_size, class_names, model):
    img = img_path
    img = tf.image.resize(cv2.imread(img), img_size)
    img = tf.expand_dims(img,0)

    print(class_names[model.predict(img).argmax()])

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();


def eval_report(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.
  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array
  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results


  
def calculate_results(data, model):
  results = []
  for images, labels in data.as_numpy_iterator():
    pred_probs = model.predict(images)
    preds = tf.argmax(pred_probs, 1)
    labels = tf.argmax(labels,1)
    results.append(eval_report(labels, preds))
  
  return pd.DataFrame(results)




IMG_DIR = "data/face_images"

def take_pictures(num_images, save_to):
    cap = cv2.VideoCapture(0)

    for i in range(num_images):
        if i == int(num_images/2):
            print("You have 5 seconds to take your headphones off, and change position")
            time.sleep(5)

        print(f"Collecting image: {i}")
        ret, frame = cap.read()
        now = datetime.now().strftime("%H:%M:%S")

        img_path = os.path.join(save_to, f"{now}.jpeg")
        cv2.imwrite(img_path, frame)
        cv2.imshow("frame", frame)
        time.sleep(1)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
