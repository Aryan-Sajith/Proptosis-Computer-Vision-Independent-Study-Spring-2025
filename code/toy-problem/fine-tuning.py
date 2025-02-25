from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os

# --- Configuration ---
dataset_path = 'code/toy-problem/labelled-faces-in-the-wild/lfw-deepfunneled/lfw-deepfunneled'
features_path = 'code/toy-problem/features/self_supervised_features.npy'

# --- Load Features from Self-Supervised Pre-Training ---
features = np.load(features_path)
# print("Loaded features with shape:", features.shape)

# --- Load Full Dataset without Shuffling to Properly Index Features ---
full_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='categorical',  
    image_size=(250, 250),
    batch_size=32,
    shuffle=False, # Do not shuffle to properly index features
    seed=42
)
all_class_names = full_ds.class_names
# print("All classes in dataset:", all_class_names)

# --- Collect labels for all images ---
all_labels = []
for images, labels in full_ds:
    all_labels.append(labels.numpy())

all_labels = np.concatenate(all_labels, axis=0)  # shape: (num_images, num_classes)
int_labels = np.argmax(all_labels, axis=1)  # convert one-hot labels to integer indices

for label in int_labels:
    print(f'Label: {label} - Class: {all_class_names[label]}')  # print the class name for each label
# print("Total images with labels:", int_labels.shape[0])

# --- Determine the 2 monst common classes ---
def get_top_two_classes(root_path):
    """Iterate through subdirectories and count .jpg files per person.
    Return a list with the two class names having the highest counts."""
    class_counts = {}
    for person in os.listdir(root_path):
        person_path = os.path.join(root_path, person)
        if os.path.isdir(person_path):
            # Count files ending in .jpg (you can adjust the extension if needed)
            count = len([f for f in os.listdir(person_path) if f.lower().endswith('.jpg')])
            class_counts[person] = count
    # Sort classes by count (descending) and pick the top two.
    top_two = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    return {cls:cnt for cls, cnt in top_two}

top_two_classes = get_top_two_classes(dataset_path)
# print("Top two classes:", top_two_classes)

# Build a mapping from class names to indices (based on ordering in full_ds)
class_to_index = {name: idx for idx, name in enumerate(all_class_names)}
# print("Class mapping:", class_to_index)
