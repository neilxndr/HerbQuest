from flask import Flask, request, render_template
from keras.models import load_model
import requests
import cv2
import numpy as np
import openai  # Import OpenAI library
from flask import Flask, render_template, redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import os
import secrets

secret_key = secrets.token_hex(16)
print(secret_key)

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
app.secret_key = secret_key

# Load the model
model = load_model("keras_Model2.h5", compile=False)


# Load the labels
class_names = open("labels2.txt", "r").readlines()

# Define OpenAI API key
openai.api_key = "your api key here"

# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
# db = SQLAlchemy(app)
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'test.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    plants = db.relationship('Plant', backref='user', lazy=True)

with app.app_context():
    db.create_all()

class Plant(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))




@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        hashed_password = generate_password_hash(password, method='sha256')

        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')



@app.route('/login_page', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# @app.route('/')
# @login_required
# def index():
#     return render_template('index.html')




# Function to get plant information from OpenAI
def get_plant_information(class_name):
    # Define your OpenAI prompt here (customize as needed)
    prompt = f"Could you provide detailed information about the plant {class_name}? Specifically, I am interested in the following aspects: Medicinal propertiesDetails of its cultivation, Regions where it is grown and abundantly available, Its scientific nameIts common name, The family it belongs to, Its traditional uses"  # You can customize the prompt
    # Call the OpenAI API to generate information
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500,
        n=1,
        stop=None,
    )

    # Extract the generated text from the response
    plant_info = response.choices[0].text

    return plant_info


# Define a function to get image search results for a plant name
def get_plant_images(plant_name):
 # Define the endpoint URL for the Custom Search API
    endpoint = "https://www.googleapis.com/customsearch/v1"
    # Define the headers for the HTTP request
    headers = {"Accept": "application/json"}
    # Define the parameters for the HTTP request
    params = {
        "key": "AIzaSyBFwo-4XjhCPMCrNDByCxmPTsQrrHqffuA",
        "cx": "f31ef25f643a94dcf",
        "q": plant_name,
        "num": 10 # You can change the number of results as per your need
    }
    # Send the HTTP request and get the response
    response = requests.get(endpoint, headers=headers, params=params)
    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        # Parse the JSON response
        results = response.json()
        # Extract the image search results
        images = results["items"]
        # Return a list of image URLs
        return [image["link"] for image in images]
    else:
        # Handle the error
        print(f"Error: {response.status_code}")
        return None


@app.route('/', methods=['GET', 'POST'])
# @login_required
def index():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1
            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Get plant information from OpenAI
            plant_info = get_plant_information(class_name[2:])

            # Get plant images from web search
            plant_images = get_plant_images(class_name[2:])

            return render_template('index1.html', class_name=class_name[2:], confidence_score=str(np.round(confidence_score * 100))[:-2], plant_info=plant_info,plant_images = plant_images)
    return render_template('index1.html', class_name=None,plant_info=None,plant_images=None)
# @login_required
# def index():
#     return render_template('index1.html', class_name=None, plant_info=None, plant_images=None)


# @login_required
# def index():
#     return render_template('index1.html')

# @app.route('/login_page',methods=['GET','POST'])
# def login_page():
#     return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
