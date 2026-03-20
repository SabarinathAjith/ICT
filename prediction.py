import tensorflow as tf
import numpy as np
import os

# 1. Load the saved model
model_path = 'fruit_veg_model.keras'

if not os.path.exists(model_path):
    print(f"Error: Could not find '{model_path}'. Make sure you trained and saved the model first!")
    exit()

print("Loading model...")
model = tf.keras.models.load_model(model_path)



image_path = r"C:\Users\imgor\Downloads\cucumber.jpg" 

if not os.path.exists(image_path):
    print(f"Error: Could not find the image '{image_path}'. Please check the path.")
    exit()

# 3. Preprocess the image to match what the model expects

img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
img_array = tf.keras.utils.img_to_array(img)


img_array = tf.expand_dims(img_array, 0) 

# 4. Make the prediction
print("Classifying image...")
predictions = model.predict(img_array)


score = predictions[0][0]

if score < 0.5:
    class_name = "Fruit"
    confidence = (1 - score) * 100
else:
    class_name = "Vegetable"
    confidence = score * 100

print("-" * 30)
print(f"Prediction: {class_name}")
print(f"Confidence: {confidence:.2f}%")
print("-" * 30)