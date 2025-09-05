from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from authlib.integrations.flask_client import OAuth
from werkzeug.security import check_password_hash, generate_password_hash
import os
import tensorflow as tf
import pickle
import cv2
import numpy as np
from hairstyle_recommendations import get_hairstyle_suggestions
from preprocess import detect_and_crop_face

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Divesh_Pandey2121@localhost/users_db'

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Google OAuth Setup
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='your-google-client-id',
    client_secret='your-google-client-secret',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params={
        'response_type': 'code',
        'scope': 'openid email profile',
        'redirect_uri': 'http://127.0.0.1:5000/google_authorized'
    },
    access_token_url='https://accounts.google.com/o/oauth2/token',
    client_kwargs={'scope': 'openid email profile'}
)

# Database Models
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(20))
    hair_type = db.Column(db.String(50))
    history = db.relationship('History', backref='user', lazy=True)
    favorites = db.relationship('Favorite', backref='user', lazy=True)

class History(db.Model):
    __tablename__ = 'history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    face_shape = db.Column(db.String(50))
    suggestions = db.Column(db.String(500))

class Favorite(db.Model):
    __tablename__ = 'favorite'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    hairstyle = db.Column(db.String(100))

# Initialize database with explicit order
with app.app_context():
    db.metadata.create_all(db.engine, tables=[User.__table__, History.__table__, Favorite.__table__])

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.password and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')  # Added to match the form in login.html
        if not email or not password or not name:
            flash('Name, email, and password are required', 'error')
            return render_template('login.html')
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
        else:
            try:
                hashed_password = generate_password_hash(password)
                user = User(email=email, password=hashed_password)
                db.session.add(user)
                db.session.commit()
                print(f"User {email} successfully saved to database!")
                login_user(user)
                flash('Sign up successful! Welcome to HairAtlas.', 'success')
                return redirect(url_for('home'))
            except Exception as e:
                db.session.rollback()
                print(f"Error during signup: {str(e)}")
                flash(f'Error during signup: {str(e)}', 'error')
    return render_template('login.html')

@app.route('/google_login')
def google_login():
    return google.authorize_redirect(url_for('google_authorized', _external=True))

@app.route('/google_authorized')
def google_authorized():
    token = google.authorize_access_token()
    resp = google.get('https://www.googleapis.com/oauth2/v1/userinfo')
    user_info = resp.json()
    session['google_token'] = token
    email = user_info['email']
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(email=email, password=None)
        db.session.add(user)
        db.session.commit()
    login_user(user)
    flash('Google login successful!', 'success')
    return redirect(url_for('home'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('google_token', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.age = request.form.get('age')
        current_user.gender = request.form.get('gender')
        current_user.hair_type = request.form.get('hair_type')
        db.session.commit()
        flash('Profile updated successfully', 'success')
    return render_template('profile.html')

@app.route('/history')
@login_required
def history():
    return render_template('history.html')

@app.route('/favorites', methods=['GET', 'POST'])
@login_required
def favorites():
    if request.method == 'POST':
        hairstyle = request.form.get('hairstyle')
        if not hairstyle:
            flash('No hairstyle selected.', 'error')
            return redirect(url_for('favorites'))
        favorite = Favorite(user_id=current_user.id, hairstyle=hairstyle)
        db.session.add(favorite)
        db.session.commit()
        flash('Hairstyle added to favorites', 'success')
    return render_template('favorites.html')

@app.route('/')
def home():
    return render_template('index.html')

# Load the trained model and label encoder
MODEL_PATH = "models/face_shape_model"  # Updated to match the correct file extension
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# Load model and label encoder
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        temp_path = "temp_image.jpg"
        file.save(temp_path)
        img = cv2.imread(temp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_img = detect_and_crop_face(img, temp_path)
        if processed_img is None or processed_img.size == 0:
            return jsonify({'error': 'No face detected in the image'}), 400
        processed_img = np.expand_dims(processed_img, axis=0) / 255.0
        prediction = model.predict(processed_img)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        
        # Debug: Log the predicted class and user attributes
        print(f"Predicted Face Shape: {predicted_class}")
        print(f"Predicted Class Type: {type(predicted_class)}")
        print(f"User Hair Type: {current_user.hair_type}")
        print(f"User Gender: {current_user.gender}")
        
        # Ensure predicted_class is a string
        predicted_class = str(predicted_class)
        
        suggestions = get_hairstyle_suggestions(predicted_class, current_user.hair_type, current_user.gender)
        
        # Debug: Print the suggestions to check their structure
        print(f"Suggestions Type: {type(suggestions)}")
        print(f"Suggestions Content: {suggestions}")
        
        # Check if suggestions is a list of dictionaries
        if not isinstance(suggestions, list) or not all(isinstance(s, dict) and "name" in s for s in suggestions):
            raise ValueError("Suggestions must be a list of dictionaries with 'name' keys")
        
        history_entry = History(user_id=current_user.id, face_shape=predicted_class, suggestions=', '.join([s["name"] for s in suggestions]))
        db.session.add(history_entry)
        db.session.commit()
        return jsonify({
            'face_shape': predicted_class,
            'hairstyle_suggestions': suggestions
        })
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.route('/predict_manual', methods=['POST'])
@login_required
def predict_manual():
    data = request.get_json()
    face_shape = data.get('face_shape')
    if not face_shape:
        return jsonify({'error': 'No face shape provided'}), 400
    suggestions = get_hairstyle_suggestions(face_shape, current_user.hair_type, current_user.gender)
    
    # Debug: Print the suggestions to check their structure
    print(f"Manual Suggestions Type: {type(suggestions)}")
    print(f"Manual Suggestions Content: {suggestions}")
    
    # Check if suggestions is a list of dictionaries
    if not isinstance(suggestions, list) or not all(isinstance(s, dict) and "name" in s for s in suggestions):
        raise ValueError("Suggestions must be a list of dictionaries with 'name' keys")
    
    history_entry = History(user_id=current_user.id, face_shape=face_shape, suggestions=', '.join([s["name"] for s in suggestions]))
    db.session.add(history_entry)
    db.session.commit()
    return jsonify({
        'face_shape': face_shape,
        'hairstyle_suggestions': suggestions
    })

# route for the Hairstyles page
@app.route('/hairstyles')
def hairstyles():
    return render_template('hairstyles.html')

if __name__ == '__main__':
    print("Starting Flask server...")   
    app.run(debug=True, host='0.0.0.0', port=5000)