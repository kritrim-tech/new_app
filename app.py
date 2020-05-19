from flask import Flask, request
import torch
from PIL import Image
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

net = torch.jit.load('ResNet_18.zip')


@app.route('/')
def hello():
    return "Hello!"


@app.route("/predict", methods=['POST'])
def predict():

    # load image
    img = Image.open(request.files['file'].stream) 
    img = img.convert(mode = 'L')
    img = img.resize((224, 224))
    img = np.array(img)
    img = torch.FloatTensor(img / 255).unsqueeze(0).unsqueeze(0)

    # get predictions
    pred = net(img)
    pred_probas = torch.softmax(pred, axis=1)

    return {
        'pneumonia': pred_probas[0][0].item(),
        'normal': pred_probas[0][1].item()
    }


if __name__ == "__main__":
    app.run(debug=True)
