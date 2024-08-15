import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model.h5')
IMG_SIZE = 224
directory_data_test = "data/data4/test"
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)
test_ds = test_datagen.flow_from_directory(
    directory=directory_data_test,
    batch_size=32,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='categorical',
    shuffle=False
)

class_label = list(test_ds.class_indices)
def predict_fruit(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    return class_label[class_index]


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Check and create the uploads directory if it doesn't exist
    if not os.path.exists('./uploads'):
        os.makedirs('./uploads')

    file_path = f"./uploads/{file.filename}"
    file.save(file_path)
    
    result = predict_fruit(file_path)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
