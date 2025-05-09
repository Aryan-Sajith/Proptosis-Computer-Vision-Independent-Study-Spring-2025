import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------- Configuration ----------
# Path to the root folder containing subdirectories (one per person)
root_path = 'code/toy-problem/labelled-faces-in-the-wild/lfw-deepfunneled/lfw-deepfunneled'
# Specify the two classes you want to use (e.g., the two most common)
top_two_classes = ['George_W_Bush', 'Colin_Powell']
# Number of examples per class to use
num_per_class = 250  # adjust as needed

# ---------- Step 1: Collect File Paths and Labels ----------
def get_file_paths_and_labels(root, classes, num_samples):
    file_paths = []
    labels = []
    # For each specified class, list all JPEG files, shuffle, and select a fixed number.
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(root, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith('.jpg')]
        random.shuffle(files)
        selected = files[:num_samples]
        for f in selected:
            file_paths.append(os.path.join(cls_dir, f))
            labels.append(label)
    return file_paths, labels

file_paths, labels = get_file_paths_and_labels(root_path, top_two_classes, num_per_class)
# print("Total images selected:", len(file_paths))
# print("Label distribution:", {top_two_classes[i]: labels.count(i) for i in range(len(top_two_classes))})

# ---------- Step 2: Split into Training and Testing Sets ----------
# Use an 80/20 split and stratify by labels
train_paths, test_paths, train_labels, test_labels = train_test_split(
    file_paths, labels, test_size=0.2, random_state=42, stratify=labels
)
# print("Training samples:", len(train_paths), "Testing samples:", len(test_paths))

# ---------- Step 3: Create TensorFlow Datasets ----------
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [250, 250])
    # Preprocess according to MobileNetV2 requirements
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(buffer_size=len(train_paths), seed=42)
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# ---------- Optional: Visualize Sample Images ----------
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(min(9, images.shape[0])):
#         ax = plt.subplot(3, 3, i + 1)
#         # Undo MobileNetV2 preprocessing for visualization (scale pixel values back to [0,1])
#         plt.imshow((images[i] + 1) / 2.0)
#         plt.title(top_two_classes[labels[i]])
#         plt.axis("off")
# plt.show()

# ---------- Step 4: Build the Supervised Model ----------
# Use MobileNetV2 as the base encoder with pre-trained ImageNet weights.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(250, 250, 3),
    include_top=False,
    weights='imagenet'
)
# Freeze the base model as we're not fine-tuning it in this baseline.
base_model.trainable = False

inputs = tf.keras.Input(shape=(250, 250, 3))
# We already apply preprocessing in the dataset pipeline, so directly pass inputs to base_model.
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
# Two output units for the two classes.
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ---------- Step 5: Train the Supervised Model ----------
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10
)

# ---------- Step 6: Save the Model ----------
save_path = 'code/toy-problem/models/supervised_classifier_top_two_limited.keras'
model.save(save_path)
print("Supervised classifier saved at:", save_path)
