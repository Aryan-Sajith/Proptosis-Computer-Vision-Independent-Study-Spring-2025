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

# for label in int_labels:
#     print(f'Label: {label} - Class: {all_class_names[label]}')  # print the class name for each label
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
    return [cls for cls, count in top_two]

top_two_classes = get_top_two_classes(dataset_path)
# print("Top two classes:", top_two_classes)

# Build a mapping from class names to indices (based on ordering in full_ds)
class_to_index = {name: idx for idx, name in enumerate(all_class_names)}
# print("Class mapping:", class_to_index)

# --- Filter Features and Associated Labels for the Top Two Classes ---
# Create a mask that selects images from the top two classes
mask = np.isin(int_labels, [class_to_index[cls] for cls in top_two_classes])
selected_indices = np.where(mask)[0] 
# print("Number of images for the top two classes:", len(selected_indices))

# Filters the features and labels based on the selected indices
selected_features = features[selected_indices]
selected_labels = int_labels[selected_indices]

# Remap the original label indices to 0 and 1 for the top two classes
new_labels = np.array([0 if lbl == class_to_index[top_two_classes[0]] else 1 for lbl in selected_labels])
# print("Selected features shape:", selected_features.shape)
# print("New labels distribution:", np.unique(new_labels, return_counts=True))

# ---------- Step 6: Limit to Only 100 Images Total ----------
n_samples = 25  # Total number of images to use from the filtered set
if selected_features.shape[0] > n_samples:
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(selected_features.shape[0], n_samples, replace=False)
    selected_features = selected_features[sample_indices]
    new_labels = new_labels[sample_indices]

print("Limited features shape:", selected_features.shape)
print("Limited labels distribution:", np.unique(new_labels, return_counts=True))


# --- Split the data into Training and Testing Splits ---
X_train, X_test, y_train, y_test = train_test_split(
    selected_features, new_labels, test_size=0.2, random_state=42
)
# print("Training samples:", X_train.shape[0], "Validation samples:", X_test.shape[0])

# --- Build, compile and summarize classifier on top of the pre-trained features ---
# Model definition
inputs = tf.keras.Input(shape=(selected_features.shape[1],))
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
classifier_model = tf.keras.Model(inputs, outputs)

# Model compilation
classifier_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
classifier_model.summary()

# --- Fine-tuning the classifier on the top two classes ---
history = classifier_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# --- Save the fine-tuned model ---
save_path = 'code/toy-problem/models/fine_tuned_classifier_top_two.keras'
classifier_model.save(save_path)
print("Fine-tuned classifier saved at:", save_path)