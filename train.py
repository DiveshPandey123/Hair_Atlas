import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from preprocess import load_and_preprocess_data, FACE_SHAPES
from sklearn.utils.class_weight import compute_class_weight
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "face_shape_model")  # Removed .keras extension
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Custom Focal Loss implementation
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_factor = tf.pow(1.0 - pt, gamma)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        loss = alpha_t * focal_factor * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed

def check_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        logger.info(f"Using GPU: {physical_devices}")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        logger.warning("No GPU detected. Training on CPU.")

check_gpu()

def build_model(num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224, 3))
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_history.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.close()

def train_model():
    logger.info("Starting train_model function")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_and_preprocess_data()
        if X_train is None or len(X_train) == 0:
            logger.error("No training data loaded. Exiting.")
            return
        logger.info(f"Data loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
        logger.info(f"Memory usage after data load: {psutil.virtual_memory().percent}%")
        if len(X_train) == 0 or len(y_train) == 0:
            logger.error("Training data is empty. Check the dataset and preprocessing.")
            return

        # Convert labels to one-hot encoded format
        y_train = to_categorical(y_train, num_classes=len(FACE_SHAPES))
        y_val = to_categorical(y_val, num_classes=len(FACE_SHAPES))
        y_test = to_categorical(y_test, num_classes=len(FACE_SHAPES))
        logger.info(f"Labels converted to one-hot: y_train shape={y_train.shape}")

    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        return

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    datagen.fit(X_train)

    class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
    class_weight_dict = dict(enumerate(class_weights))
    logger.info(f"Class weights: {class_weight_dict}")

    model = build_model(num_classes=len(FACE_SHAPES))
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    try:
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=8),
            validation_data=(X_val, y_val),
            epochs=50,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        logger.info(f"Memory usage after training: {psutil.virtual_memory().percent}%")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Accuracy: {test_accuracy}")
    logger.info(f"Test Loss: {test_loss}")

    try:
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot back to class indices
        plot_confusion_matrix(y_test_classes, y_pred_classes, label_encoder.classes_)
        
        # Save classification report to a file
        report = classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_, zero_division=1)
        with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
            f.write(report)
        logger.info("Classification report saved to classification_report.txt")
        
        plot_training_history(history)
    except Exception as e:
        logger.error(f"Error during prediction or evaluation: {e}")

    try:
        model.save(MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
        with open(LABEL_ENCODER_PATH, "wb") as f:
            pickle.dump(label_encoder, f)
        logger.info(f"Label encoder saved to {LABEL_ENCODER_PATH}")
    except Exception as e:
        logger.error(f"Error saving model or label encoder: {e}")

if __name__ == "__main__":
    train_model()