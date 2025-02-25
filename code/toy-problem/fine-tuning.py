from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# --- Configuration ---
dataset_path = 'code/toy-problem/labelled-faces-in-the-wild/lfw-deepfunneled/lfw-deepfunneled'
features_path = 'code/toy-problem/features/self_supervised_features.npy'

# --- Load Features from Self-Supervised Pre-Training ---
features = np.load(features_path)
print("Loaded features with shape:", features.shape)
