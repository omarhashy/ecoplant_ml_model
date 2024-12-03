from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMG_SIZE = 224  # Image dimensions used in training
BATCH_SIZE = 32  # Batch size
MODEL_PATH = "plant_village_model.keras"  # Path to the saved model
DATASET_PATH = "plantvillage_dataset"  # Path to dataset

# Initialize the data generator with a validation split of 20%
data_generator = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    validation_split=0.3,  # Reserve 30% of the data for validation
)

# Create validation data generator
validation_generator = data_generator.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),  # Resize images
    batch_size=BATCH_SIZE,
    subset="validation",  # Use only the 20% validation split
    class_mode="categorical",  # Multi-class classification
)

# Load the trained model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Print class indices (optional, to verify mapping)
class_indices = validation_generator.class_indices
for class_name, class_index in class_indices.items():
    print(f"Class {class_index}: {class_name}")

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(
    validation_generator, steps=validation_generator.samples // BATCH_SIZE
)

# Print results
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
