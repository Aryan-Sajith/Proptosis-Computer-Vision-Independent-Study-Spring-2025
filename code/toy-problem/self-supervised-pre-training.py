import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# --- Dataset Preparation ---
dataset_path = 'code/toy-problem/labelled-faces-in-the-wild/lfw-deepfunneled/lfw-deepfunneled'

# No labels utilized for self-supervised pre-training as we utilize the images directly to extract features
ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels=None,  
    image_size=(250, 250),
    batch_size=32,
    shuffle=True,
    seed=42
)

# --- Self-Supervised General Task: Rotation Prediction ---
def random_rotate(image):
    """Apply a random rotation to a single image and return the rotation label"""
    # Randomly select a rotation angle from 0, 90, 180, or 270 degrees or 0, 1, 2, 3 in terms of class labels.
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    rotated_image = tf.image.rot90(image, k)
    return rotated_image, k

# Since ds returns batches, define a mapping function to process each batch.
def batch_random_rotate(batch):
    # Apply the random_rotate function to each image in the batch.
    rotated_images, labels = tf.map_fn(
        lambda img: random_rotate(img),
        batch,
        fn_output_signature=(tf.float32, tf.int32)
    )
    return rotated_images, labels

# Normalize the images to [0, 1] and apply random rotations
ss_ds = ds.map(lambda batch: (tf.cast(batch, tf.float32) / 255.0))  
# Apply random rotations to the images via the batch_random_rotate function
ss_ds = ss_ds.map(batch_random_rotate, num_parallel_calls=tf.data.AUTOTUNE)
# Prefetch the data for improved performance.
ss_ds = ss_ds.cache().prefetch(tf.data.AUTOTUNE)

# --- Model Setup ---
# We use the MobileNetV2 model pre-trained on ImageNet without the top classification layer. 
# This layer is dropped because we will add a new classification head for the rotation prediction task.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(250, 250, 3),
    include_top=False,
    weights='imagenet'
)
# Essentially: We want to use the pre-trained weights of the base model, but we don't want to update them 
# during training. This is known as feature extraction which is a common transfer learning technique.
base_model.trainable = False

# Build the self-supervised head for rotation prediction (4 classes).
inputs = tf.keras.Input(shape=(250, 250, 3))
# Preprocess the inputs as required by MobileNetV2.
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
# Extract features using the base encoder.
x = base_model(x, training=False)
# Global average pooling to convert feature maps to a single vector.
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# Dropout for regularization - optional but can be useful to prevent overfitting.
x = tf.keras.layers.Dropout(0.2)(x)
# Classification head: 4 output units for predicting rotation angle.
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model with a suitable optimizer and loss function.
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Self-Supervised Pre-Training ---
# Train the model on the self-supervised task.
history = model.fit(ss_ds, epochs=10)

# --- Feature Extraction ---
# After self-supervised pre-training, we want to save the features from the base encoder.
# Create a feature extractor model that outputs the pooled features.
feature_extractor = tf.keras.Model(
    inputs=base_model.input,
    outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
)

# Extract features for all images in the original (normalized) dataset.
all_features = []
for batch in ds:
    batch = tf.cast(batch, tf.float32) / 255.0  # Normalize the batch
    features = feature_extractor.predict(batch)
    all_features.append(features)

all_features = np.concatenate(all_features, axis=0)
# Save the extracted features to disk for later fine-tuning.
feature_path = 'code/toy-problem/features/self_supervised_features.npy'
np.save(feature_path, all_features)

print("Feature extraction complete. Saved features shape:", all_features.shape)
