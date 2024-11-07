import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

model = tf.keras.models.load_model('mnist_model_tanmay.keras')

(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype('float32') / 255.0

y_pred = model.predict(x_test)

y_pred_labels = np.argmax(y_pred, axis=1)

overall_accuracy = np.mean(y_pred_labels == y_test)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

for digit in range(10):
    digit_indices = np.where(y_test == digit)[0]
    
    digit_true = y_test[digit_indices]
    digit_pred = y_pred_labels[digit_indices]
    
    digit_accuracy = np.mean(digit_pred == digit_true)
    print(f"Accuracy for digit {digit}: {digit_accuracy:.4f}")
