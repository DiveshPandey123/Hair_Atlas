import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import mediapipe as mp

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = "dataset"
FACE_SHAPES = ["diamond", "heart", "oblong", "oval", "round", "square"]
IMG_SIZE = (224, 224)  

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# Function to detect and crop face
def detect_and_crop_face(img, img_path):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = img.shape
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        if width > 0 and height > 0:
            face_img = img[y:y+height, x:x+width]
            if face_img.size > 0:
                debug_dir = "debug_cropped_faces"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"cropped_{os.path.basename(img_path)}.jpg"), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                logger.info(f"Face detected with MediaPipe: {face_img.shape}")
                return cv2.resize(face_img, IMG_SIZE)
    logger.warning("No face detected with MediaPipe, trying Haar Cascade")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        logger.error("Haar Cascade not loaded")
        return None  # Return None instead of resizing full image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        if w > 0 and h > 0:
            face_img = img[y:y+h, x:x+w]
            if face_img.size > 0:
                debug_dir = "debug_cropped_faces"
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"cropped_{os.path.basename(img_path)}_haar.jpg"), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                logger.info(f"Face detected with Haar Cascade: {face_img.shape}")
                return cv2.resize(face_img, IMG_SIZE)
    logger.warning(f"No face detected in {img_path}, skipping image")
    return None  # Skip image if no face detected

# Load and preprocess data
def load_and_preprocess_data():
    images = []
    labels = []

    for shape in FACE_SHAPES:
        folder_path = os.path.join(DATA_DIR, shape)
        if not os.path.exists(folder_path):
            logger.error(f"Folder {folder_path} does not exist")
            continue
        image_count = 0
        for img_name in os.listdir(folder_path):
            if image_count >= 100:  
                break
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            logger.info(f"Processing image: {img_path}, shape: {img.shape}")
            cropped_img = detect_and_crop_face(img, img_path)
            if cropped_img is not None:
                images.append(cropped_img)
                labels.append(shape)
            image_count += 1

    if not images:
        raise ValueError("No valid images loaded. Check dataset path, structure, or face detection.")

    images = np.array(images)
    labels = np.array(labels)
    images = images / 255.0

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Adjusted split to 80-10-10
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Validation set size: {X_val.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    logger.info(f"Label encoder classes: {list(label_encoder.classes_)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_and_preprocess_data()