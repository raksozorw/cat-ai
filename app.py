from flask import Flask, request, jsonify
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import io  # Import the 'io' module
from flask_cors import CORS
import model


app = Flask(__name__)
CORS(app)



# Define a transformation to preprocess images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# API endpoint to accept and process images
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
            output = model.model(image.unsqueeze(0))  # Unsqueeze to add a batch dimension
            probabilities = torch.softmax(output, dim=1).numpy()

        # Get the predicted class
        _, predicted = torch.max(output, 1)

        # Define class labels
        class_labels = ["jarlsberg", "kvarg"]

        # format probs for JSON
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
   import model  
   return app
if __name__ == '__main__':
    app.run(debug=True)
