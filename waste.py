import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from collections import defaultdict

# === 1. Load dataset ===
data_dir = 'dataset'  # Folder must contain subfolders for each class (e.g., recyclable, hazardous, organic)
img_size = (224, 224)  # Required size for MobileNetV2
batch_size = 4

# Load images with rescaling
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

print(f"\n✅ Loaded {train_data.samples} training images from {data_dir}")
print(f"📁 Classes found: {train_data.class_indices}")

# === 2. Load base model ===
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

# === 3. Build full model ===
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === 4. Train model ===
print("\n🚀 Starting training...\n")
history = model.fit(train_data, epochs=5)

# === 5. Single image prediction function ===
def predict_waste(image_path):
    try:
        img = load_img(image_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        classes = list(train_data.class_indices.keys())
        predicted_class = classes[np.argmax(prediction)]

        print(f"✅ {os.path.basename(image_path)} ➜ Predicted: {predicted_class}")
        return predicted_class
    except Exception as e:
        print(f"❌ Error during prediction for {image_path}: {e}")
        return None

# === 6. Predict all images in a folder and summarize ===
def predict_all_from_folder(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_paths:
        print(f"\n❌ No images found in {folder_path}")
        return

    print(f"\n🔍 Found {len(image_paths)} image(s) in {folder_path}. Predicting...\n")

    category_counts = defaultdict(int)

    for img_path in image_paths:
        predicted_class = predict_waste(img_path)
        if predicted_class:
            category_counts[predicted_class] += 1

    print("\n📊 Prediction Summary:")
    for category in ['recyclable', 'hazardous', 'organic']:
        print(f"  🗂️ {category.title()}: {category_counts[category]} image(s)")

# === 7. Run batch predictions ===
predict_all_from_folder("test_images")  # Replace with your actual folder
