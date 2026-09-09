import tensorflow as tf
import numpy as np

###############################################################
# Globals - must match your training script
###############################################################
image_size = (224, 224)
batch_size = 32

###############################################################
# Load the saved model
###############################################################
model = tf.keras.models.load_model("resnet.keras")

###############################################################
# Load the test dataset
###############################################################
test_ds = tf.keras.utils.image_dataset_from_directory(
    "/home/cs421_share/celebdataset/test",
    #"celebdataset/test",
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)
class_names = test_ds.class_names

# Normalize — same as training, no augmentationn though
normalization_layer = tf.keras.layers.Rescaling(1./255)
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

###############################################################
# Evaluate —  loss and accuracy
###############################################################
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

###############################################################
# Predict, scorekeeping per class
###############################################################
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

# Per class accuracy
print("\nPer-Class Results:")
print("-" * 30)
for i, class_name in enumerate(class_names):
    class_indices = [j for j, val in enumerate(y_true) if val == i]
    correct = sum(1 for j in class_indices if y_pred[j] == y_true[j])
    total = len(class_indices)
    print(f"{class_name}: {correct}/{total} ({100*correct/total:.2f}%)")