import json
import base64
import requests
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# API_URL = "http://10.66.66.2:5000/predictions/image-retrieval-v1.0"
API_URL = "https://efiss.tech/ai/predictions/image-retrieval-v1.0"
# API_URL = "http://localhost:5000/predictions/image-retrieval-v1.0"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image.filename != '':
            image_data = image.read()
            encoded_image = base64.b64encode(image_data).decode('utf-8')

            payload = {
                "top_k": 10,
                "image": encoded_image,
                "debug": True
            }

            response = requests.post(API_URL, json=payload)
            print(f"Response status code: {response.status_code}")
            res = response.json()
            print(f"Response: {res}")
            if response.status_code == 200:
                data = res
                relevant_images = data['relevant']
                # print(f"Relevant images: {relevant_images}")
                # relevant_images = [f'https://storage.googleapis.com/efiss/data/product_images/{relevant_image.split('/')[-1]}.jpg' for relevant_image in relevant_images]
                for i in range(len(relevant_images)):
                    sitename = "shopee-" + relevant_images[i].split("shopee_")[1]
                    relevant_images[i] = f'https://storage.googleapis.com/efiss/data/product_images/{sitename}/{relevant_images[i].split("/")[-1]}.jpeg'
                print(f"Relevant images: {relevant_images}")
                distances = data['distances']
                print(f"Distances: {distances}")
                return render_template('index.html', images=relevant_images, distances=distances, zip=zip)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
