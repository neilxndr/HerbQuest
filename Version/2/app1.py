from flask import Flask, request, render_template
from keras.models import load_model
import requests
import cv2
import numpy as np
import openai  # Import OpenAI library

app = Flask(__name__)

# Load the model
model = load_model("keras_Model1.h5", compile=False)

# Load the labels
class_names = open("labels1.txt", "r").readlines()

# Define OpenAI API key
openai.api_key = "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"



# Function to get plant information from OpenAI
def get_plant_information(class_name):
    # Define your OpenAI prompt here (customize as needed)
    prompt = f"Tell me about the medicinal properties of  {class_name} plant and give details of its cultivation, and where its grown and abundantly avaialable"  # You can customize the prompt

    # Call the OpenAI API to generate information
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
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
def upload_predict():
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

@app.route('/login_page',methods=['GET','POST'])
def login_page():
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
