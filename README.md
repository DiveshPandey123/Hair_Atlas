# Hair Atlas – AI Based Hairstyle Recommendation System  

Hair Atlas is an AI-powered hairstyle recommendation web application that analyzes a user's face shape and suggests suitable hairstyles.  
It uses Computer Vision + Deep Learning (TensorFlow/Keras) to detect face shape and provides personalized hairstyle suggestions with a clean web interface.

---

## 🚀 Features

### 🔍 AI-Based Face Shape Detection
- Uses a trained neural network model to classify face shapes:
  - **Oval**
  - **Round**
  - **Square**
  - **Heart**
  - **Oblong**
  - **Diamond**

### 💇 Intelligent Hairstyle Recommendations
- Each face shape maps to a curated set of hairstyle suggestions.
- Hairstyle images stored in:
static/Hairstyle_images/
static/suggestions/

### 👤 User System (Login / Signup)
- Users can:
- Create accounts  
- Upload photos  
- View hairstyle results  
- Save **favorites**  
- View **history**

### 📸 Face Preprocessing Pipeline
- OpenCV-based face detection  
- Automatic cropping  
- Preprocessing for model prediction  
- Debug images stored in:
debug_cropped_faces/



### 🖥️ Frontend (HTML/CSS/JS)
- Responsive UI  
- Multiple pages:
- Home
- Login/Signup
- Profile
- Suggestions
- Favorites
- History

---

## 📂 Project Structure

HAIR_ATLAS/
│
├── app.py # Main Flask backend
├── hairstyle_recommendations.py # Recommendation engine
├── preprocess.py # Face preprocessing logic
├── train.py # Model training script
├── test_cv2.py # Debug/testing script
├── requirements.txt # Dependencies
│
├── models/
│ └── face_shape_model/
│ ├── saved_model.pb
│ ├── keras_metadata.pb
│ ├── variables/
│ │ ├── variables.index
│ │ ├── variables.data-00000-of-00001
│ ├── face_shape_model.h5
│ ├── face_shape_model.keras
│ ├── label_encoder.pkl
│ └── training_history.png
│
├── dataset/
│ ├── diamond/
│ ├── heart/
│ ├── oblong/
│ ├── oval/
│ ├── round/
│ └── square/
│
├── debug_cropped_faces/
│ ├── cropped_temp_image.jpg
│ └── cropped_temp_image_haar.jpg
│
├── static/
│ ├── css/
│ │ ├── hairstyles.css
│ │ ├── loginpage.css
│ │ └── styles.css
│ ├── js/
│ │ ├── loginpage.js
│ │ └── script.js
│ ├── Hairstyle_images/
│ └── suggestions/
│ ├── diamond/
│ ├── heart/
│ ├── oblong/
│ ├── oval/
│ ├── round/
│ └── square/
│
└── templates/
├── favorites.html
├── history.html
├── hairstyles.html
├── index.html
├── login.html
├── logout.html
├── profile.html
└── signup.html



---

## 🧠 AI Model Details

### 📘 Model Type  
Convolutional Neural Network (CNN) trained on labeled face-shape dataset.

### 📦 Training Script  
Run:
```bash
python train.py
Outputs:

.keras model

.h5 model

TensorFlow SavedModel (saved_model.pb)

Training history graph

🛠️ Installation
1️⃣ Clone Repository

git clone https://github.com/DiveshPandey123/Hair_Atlas.git
cd Hair_Atlas
2️⃣ Create Virtual Environment (optional)

python -m venv venv
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies

pip install -r requirements.txt
4️⃣ Run the Application

python app.py
App runs at:
👉 http://localhost:5000

🎯 How It Works (Flow)
User uploads an image

preprocess.py → detects and crops face

Model predicts face shape

hairstyle_recommendations.py → fetches hairstyles

Results displayed with images

User can add to favorites or history

📸 Screenshots (Add when you have)

[ ] Upload Screen
[ ] Predicted Face Shape
[ ] Suggested Hairstyles
[ ] Favorites Page
[ ] History Page
🔮 Future Improvements
Add more diverse hairstyle datasets

Add gender-based recommendations

Upgrade to a more robust face-shape model (MobileNetV3)

Add image enhancement before prediction

Deploy using Docker or Render/Netlify

🤝 Contributing
Pull requests are welcome!
Follow clean code practices & mention changes clearly.

📜 License
MIT License (or add your preferred license)

👨‍💻 Developed by
Divesh Pandey – CSE Engineer
