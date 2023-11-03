import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

train_ds = tf.keras.utils.image_dataset_from_directory('/content/drive/MyDrive/data', validation_split=0.7, subset="training", seed=30)
val_ds = tf.keras.utils.image_dataset_from_directory('/content/drive/MyDrive/data', validation_split=0.2, subset="validation", seed=30)
test_ds = tf.keras.utils.image_dataset_from_directory('/content/drive/MyDrive/data', validation_split=0.1, subset="validation", seed=30)

model_final_ = Sequential([
  layers.Rescaling(1/255, input_shape=(256, 256, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(3)])

model_final_.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics="accuracy")

model_final_.summary()

checkpoint_path = "/content/drive/MyDrive/data/model_final_"
checkpoint_dir = (checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
callback = tf.keras.callbacks.EarlyStopping(patience=8)
epochs = 50
hist_final_ = model_final_.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cp_callback, callback])

loss = hist_final_.history['loss']
val_loss = hist_final_.history['val_loss']
accuracy = hist_final_.history['accuracy']
val_accuracy = hist_final_.history['val_accuracy']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(loss) + 1), loss, label='Training Loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(accuracy) + 1), accuracy, label='Training Accuracy')
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.title('Accuracy')

plt.show()

class_names = train_ds.class_names
img = tf.keras.utils.load_img('/content/drive/MyDrive/data/test6.jpg', target_size=(256, 256))
plt.imshow(img)
plt.show()

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model_final_.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("Obraz pasuje do {} z {:.2f} % prawdopodobieństwa".format(class_names[np.argmax(score)], 100 * np.max(score)))

def count_test_images(test_ds, model):
    incorrect_predictions = 0
    total_images = 0

    for images, labels in test_ds:
        total_images += len(images)
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        incorrect_predictions += np.sum(predicted_labels != labels)

    return incorrect_predictions, total_images

def display_misclassified_images(test_ds, model, class_names):
    for images, labels in test_ds:
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        misclassified_indices = np.where(predicted_labels != labels)[0]

        for index in misclassified_indices:
            img = images[index].numpy().astype(np.uint8)
            plt.imshow(img)
            plt.title(f"True: {class_names[labels[index]]}, Predicted: {class_names[predicted_labels[index]]}")
            plt.show()

incorrect_predictions, total_images = count_test_images(test_ds, model_final_)
print(f"Liczba błędnych predykcji: {incorrect_predictions}")
print(f"Liczba obrazów w zbiorze testowym: {total_images}")
accuracy = 100 * (1 - (incorrect_predictions / total_images))
print(f"Accuracy: {accuracy}")

if incorrect_predictions > 0:
    display_misclassified_images(test_ds, model_final_, class_names)