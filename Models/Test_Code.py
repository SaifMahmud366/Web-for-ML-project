import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as ig

# Load the saved model
model = load_model("model.h5")

# Define the classes for reference
classes = ['Damaged', 'NO Damage']

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = ig.load_img(image_path, target_size=(128, 128))
    img_array = ig.img_to_array(img)
    img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_image(model, image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
    predicted_class = classes[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    return predicted_class, confidence

# Example usage:
image_path = "test car.jpg"
predicted_class, confidence = predict_image(model, image_path)
print("Predicted Class:", predicted_class)
print("Confidence:", confidence)