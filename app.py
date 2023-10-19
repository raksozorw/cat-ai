from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
from flask_cors import CORS
import boto3
import botocore
from botocore import UNSIGNED
import torchvision.models as models
import torch.nn as nn

import os

app = Flask(__name__)
CORS(app)

# Initialize S3 - currently the resource is public 
# TODO fix local credentials and make this private
s3 = boto3.client('s3', config=botocore.config.Config(signature_version=UNSIGNED))

# Process images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the model from S3
s3_bucket_name = 'cai-test-bucket'  
model_key = 'models/cat_recognition_model4.pth'  
local_model_path = '/tmp/model.pth'  

if not os.path.exists(local_model_path):
    s3.download_file(s3_bucket_name, model_key, local_model_path)





model = models.resnet50(pretrained=True) 
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model.load_state_dict(torch.load(local_model_path, map_location='cpu')) 
model.eval()

# API endpoint to accept and eval images
@app.route('/predict', methods=['POST'])
def predict_cat():
    print('request received')
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})

        image = request.files['image'].read()
        image = Image.open(io.BytesIO(image))
        image = transform(image)  # Apply the preprocessing transformation

        # Make a prediction and get class probabilities
        with torch.no_grad():
            output = model(image.unsqueeze(0))  
            probabilities = torch.softmax(output, dim=1).numpy()

        # Get the predicted class
        _, predicted = torch.max(output, 1)

        # Define class labels
        class_labels = ["jarlsberg", "kvarg"]

        # Format probs for JSON
        probabilities_dict = {class_labels[i]: float(probabilities[0, i]) for i in range(len(class_labels))}

        # Return the prediction and class probabilities as a JSON response
        print(probabilities)
        response_data = {
            'prediction': class_labels[predicted.item()],
            'probabilities': probabilities_dict
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})

def handler():
   return app

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
