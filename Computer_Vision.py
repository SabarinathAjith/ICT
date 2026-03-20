import os
import zipfile
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models


# 1. Defining the path for the zip file which contains the dataset

zip_path = r"C:\Users\imgor\Downloads\CNN.zip"
extract_dir = "dataset" 


# 2. Extracting the zip file

if not os.path.exists(extract_dir):
    print("Extracting ZIP dataset...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.\n")
else:
    print("Dataset already extracted. Skipping extraction.\n")


# 3. Reorganizing for binary classification

fruits_list = ['banana', 'apple', 'pear', 'grapes', 'orange', 'kiwi', 'watermelon', 'pomegranate', 'pineapple', 'mango']


vegetables_list = ['cucumber', 'carrot', 'capsicum', 'onion', 'potato', 'lemon', 'tomato', 'radish', 'raddish', 'beetroot', 'cabbage', 
                   'lettuce', 'spinach', 'soybean', 'soy beans', 'cauliflower', 'bell pepper', 'chilli pepper', 'chilly', 'pepper', 'turnip', 
                   'corn', 'sweetcorn', 'sweet potato', 'sweetpotato', 'paprika', 'jalapeño', 'jalepeno', 'ginger', 'garlic', 'peas', 'eggplant']

def organize_binary_classes(base_dir):
    if not os.path.exists(base_dir):
        return
    
    fruits_dir = os.path.join(base_dir, 'Fruits')
    veg_dir = os.path.join(base_dir, 'Vegetables')
    
    os.makedirs(fruits_dir, exist_ok=True)
    os.makedirs(veg_dir, exist_ok=True)
    
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        

        if not os.path.isdir(folder_path) or folder_name in ['Fruits', 'Vegetables']:
            continue
            
        folder_lower = folder_name.lower().strip()
        
        if folder_lower in fruits_list:
            dest_dir = fruits_dir
        elif folder_lower in vegetables_list:
            dest_dir = veg_dir
        else:
            print(f"Warning: Found unmapped folder '{folder_name}'. Leaving it alone.")
            continue
            

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                new_filename = f"{folder_name}_{filename}"
                shutil.move(file_path, os.path.join(dest_dir, new_filename))
        

        try:
            os.rmdir(folder_path)
        except OSError:
            pass

train_dir = os.path.join(extract_dir, 'train')
val_dir = os.path.join(extract_dir, 'validation')
test_dir = os.path.join(extract_dir, 'test')

if os.path.exists(os.path.join(extract_dir, 'CNN', 'train')):
    train_dir = os.path.join(extract_dir, 'CNN', 'train')
    val_dir = os.path.join(extract_dir, 'CNN', 'validation')
    test_dir = os.path.join(extract_dir, 'CNN', 'test')

organize_binary_classes(train_dir)
organize_binary_classes(val_dir)
organize_binary_classes(test_dir)
print("Reorganization complete! Check folders.\n")


# 4. Building and training the model

print("Loading datasets and building model...")

img_height = 150
img_width = 150
batch_size = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = 10
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs
)

test_loss, test_acc = model.evaluate(test_dataset)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# 5. Saving the trained model to a file

model.save('fruit_veg_model.keras')