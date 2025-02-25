from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# --- Configuration ---
dataset_path = 'code/toy-problem/labelled-faces-in-the-wild/lfw-deepfunneled/lfw-deepfunneled'
features_path = 'code/toy-problem/features/self_supervised_features.npy'

# --- Load Features from Self-Supervised Pre-Training ---
features = np.load(features_path)
print("Loaded features with shape:", features.shape)

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
print("All classes in dataset:", all_class_names)