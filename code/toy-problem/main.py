import tensorflow as tf
import matplotlib.pyplot as plt

# Define the path to your dataset
dataset_path = 'code/toy-problem/labelled-faces-in-the-wild/lfw-deepfunneled'

# Load the dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',  # Automatically infer labels from subdirectory names
    label_mode='int',   # Labels as integers
    image_size=(250, 250),  # Resize images to 250x250 pixels
    batch_size=32,      # Number of images to return in each batch
    shuffle=True,       # Shuffle the data
    seed=123            # Seed for reproducibility
)

# Retrieve a batch of images and labels
image_batch, label_batch = next(iter(dataset))

# Define the class names based on the directory structure
class_names = dataset.class_names

# Plot the first nine images with their labels
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    plt.title(class_names[label_batch[i]])
    plt.axis("off")
plt.show()
