
# Waste Classification Using Deep Learning (Step-by-Step Guide)

This project uses a deep learning model (MobileNetV2) to classify waste images into **recyclable**, **hazardous**, and **organic** categories. This guide walks you through each step to understand and run the project from scratch.

---

##Step 1: Install Dependencies

Make sure Python is installed. Then install required packages:

```bash
pip install tensorflow numpy matplotlib
```

---

## Step 2: Organize Your Dataset

Create a folder named `dataset/` with three subfolders:
- `recyclable/`
- `hazardous/`
- `organic/`

Each folder should contain relevant images of that waste type.

```
dataset/
recyclable/
hazardous/
organic/
```

Also prepare a folder `test_images/` with images to test the model.

---

##Step 3: Load and Preprocess Data

Use `ImageDataGenerator` to load images from the dataset and normalize pixel values:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical',
    shuffle=True
)
```

---

Step 4: Build the Model

Use MobileNetV2 as the base model (with pre-trained ImageNet weights):

```python
import tensorflow as tf

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])
```

---

##Step 5: Compile and Train

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=5)
```

---

##Step 6: Test with an Image

```python
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_waste(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    classes = list(train_data.class_indices.keys())
    print(f"Predicted Category: {classes[np.argmax(prediction)]}")
```

---

##Step 7: Batch Prediction and Summary

```python
import os

def predict_all_from_folder(folder_path):
    category_counts = {cls: 0 for cls in train_data.class_indices.keys()}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(folder_path, filename)
            img = image.load_img(path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = list(train_data.class_indices.keys())[np.argmax(prediction)]
            category_counts[predicted_class] += 1
            print(f"{filename} ➜ {predicted_class}")

    print("\n📊 Prediction Summary:")
    for cls, count in category_counts.items():
        print(f"  {cls}: {count} image(s)")
```

---

##Step 8: Improve and Deploy

- Add data augmentation for better accuracy.
- Export model using `model.save()`.
- Deploy using TensorFlow Lite or to the web/mobile apps.