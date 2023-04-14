import torch
import json
import torchvision.transforms as transforms
from PIL import Image
import requests
from flask import Flask, jsonify, request
import urllib.request

from ultralytics import YOLO
from io import BytesIO

# Define the API server
app = Flask(__name__)

# Models
modelYolo = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

# Load the pre-trained PyTorch model
# modelResnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True, force_reload=True)
# modelResnet18.eval()

# # Define the image preprocessing function
def preprocess_image_resnet18(image_url):

    ## Preprocess the image ##
    # Load the image from the URL
    image = Image.open(requests.get(image_url, stream=True).raw)
    print(image)
    # Define the image preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Preprocess the image
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    # Use the pre-trained model to detect objects in the image
    with torch.no_grad():
        output = modelResnet18(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, indices = torch.sort(output, descending=True)
        top_indices = indices[0][:5]
    
    # Get the labels for the detected objects
    labels_path = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    labels = []
    with urllib.request.urlopen(labels_path) as f:
        classes = [line.decode('utf-8').strip() for line in f.readlines()]
    for i in top_indices:
        labels.append(classes[i])
    
    return (labels[0])

def preprocess_image_yolov5s(image_url,ext):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    modelYolo(img, save_txt= True)
    return({"result": "labels saved"})
    
# Define the API endpoint for object detection
@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    # Get the image URL from the request
    image_url = request.json['image_url']
    ext = request.json['image_ext']
    
    # predictionResnet = preprocess_image_resnet18(image_url)
    # predictionResnetJSON={"name": predictionResnet}

    predictionYolo = preprocess_image_yolov5s(image_url,ext)
    
    # predictions = [predictionResnetJSON,predictionYolo]
    # print(predictions)
    # Return the detected objects as JSON
    return jsonify(predictionYolo)

if __name__ == '__main__':
    app.run(debug=True)