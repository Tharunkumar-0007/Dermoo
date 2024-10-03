import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil  # Using shutil to copy files instead of moving

# Specify the main folder paths containing subfolders
main_folder_paths = [
    'D:/dataset/Skin diseases dataset/IMG_CLASSES',
    'D:/dataset/Skin diseases dataset2/train',
    'D:/dataset/Skin diseases dataset2/test'
]

# Combine the datasets into one directory structure
combined_data_dir = 'D:/dataset/combined_data'  # Adjust as needed

# Create a combined directory structure
for main_folder in main_folder_paths:
    for folder_name in os.listdir(main_folder):
        src_folder = os.path.join(main_folder, folder_name)
        dst_folder = os.path.join(combined_data_dir, folder_name)

        # Create destination folder if it does not exist
        os.makedirs(dst_folder, exist_ok=True)

        # Copy images to the combined directory
        for filename in os.listdir(src_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_file = os.path.join(src_folder, filename)
                dst_file = os.path.join(dst_folder, filename)
                
                # Skip if file already exists, or copy the file
                if not os.path.exists(dst_file):
                    shutil.copy(src_file, dst_file)
                else:
                    print(f"File {dst_file} already exists, skipping...")

# 2. Set up ImageDataGenerator for real-time data augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 3. Create generators for training and validation
train_generator = datagen.flow_from_directory(
    combined_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse',
    subset='training'  # Set as training data
)

validation_generator = datagen.flow_from_directory(
    combined_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse',
    subset='validation'  # Set as validation data
)

# 4. Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer for number of classes
])

# 5. Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. Train the model using the generators
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# 7. Save the trained model
model.save('image_classifier_model.h5')
print("Model saved successfully.")

# 8. Plot training history (optional)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
