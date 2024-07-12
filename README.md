# Multi-Modal-Imaging-Analysis-for-Abnormality-Detection


```markdown
# Heart Echo Image Analysis

This project involves preprocessing imaging data, extracting frames from videos, normalizing and resizing images, extracting features using a pre-trained VGG16 model, splitting the data, training a neural network, evaluating the model, and implementing attention mechanisms.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Extract Frames from Videos](#extract-frames-from-videos)
  - [Normalize and Resize Images](#normalize-and-resize-images)
  - [Feature Extraction from Images](#feature-extraction-from-images)
  - [Data Splitting](#data-splitting)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Reporting](#reporting)
  - [Attention Mechanisms](#attention-mechanisms)

## Installation

To run this project, you need to install the required Python packages. Use the following commands to set up your environment:

```sh
pip install scikit-video
pip install -U scikit-learn
pip install -U scikit-video
pip install tensorflow
pip install numpy
pip install matplotlib
```

## Usage

### Extract Frames from Videos

This step involves extracting frames from video files and saving them as images.

```python
import os
import cv2

def extract_frames_from_directory(videos_dir, output_base_dir):
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.avi')]
    
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        video_output_dir = os.path.join(output_base_dir, os.path.splitext(video_file)[0])
        os.makedirs(video_output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(video_output_dir, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        cap.release()
```

### Normalize and Resize Images

Normalize and resize images to a target size.

```python
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    return img_array

def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                img_array = preprocess_image(img_path, target_size=target_size)
                output_path = os.path.join(output_dir, os.path.relpath(img_path, input_dir))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path.replace('.jpg', '.npy'), img_array)
```

### Feature Extraction from Images

Extract features from preprocessed images using a pre-trained VGG16 model.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import os

def extract_features(image_path, model):
    img_array = np.load(image_path)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features_from_directory(input_dir, model):
    features_dict = {}
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            for file in os.listdir(dir_path):
                if file.endswith('.npy'):
                    image_path = os.path.join(dir_path, file)
                    features = extract_features(image_path, model)
                    features_dict[image_path] = features
                    print(f"Extracted features from {image_path}")
    return features_dict
```

### Data Splitting

Split the data into training, validation, and test sets.

```python
from sklearn.model_selection import train_test_split

image_paths = list(features_dict.keys())
features = list(features_dict.values())

labels = np.random.randint(0, 2, size=len(features))

X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

### Model Training

Train a neural network on the extracted features.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Flatten(input_shape=(7, 7, 512)),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(np.array(X_train), np.array(y_train), 
                    validation_data=(np.array(X_val), np.array(y_val)),
                    epochs=10, batch_size=32)
```

### Model Evaluation

Evaluate the model on the test set.

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

y_pred = model.predict(np.array(X_test))
y_pred_class = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)
auc_roc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"AUC-ROC: {auc_roc}")
```

### Reporting

Plot the training and validation accuracy and loss.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
```

### Attention Mechanisms

Implement attention mechanisms in the model.

```python
from tensorflow.keras.layers import Attention, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def add_attention_layer(inputs):
    attention = Attention()([inputs, inputs])
    return attention

combined_with_attention = add_attention_layer(combined)
x = GlobalAveragePooling2D()(combined_with_attention)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model_with_attention = Model(inputs=[echo_input, mri_input], outputs=output)
model_with_attention.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_with_attention.summary()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
