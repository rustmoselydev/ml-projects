from flask import Flask, request, jsonify
from fastai.vision.all import *
from PIL import Image
import __main__
from io import BytesIO
import base64

def get_category(filename):
    return filename.parent.name.replace("_", " ")

app = Flask(__name__)
__main__.get_category = get_category

model = load_learner('api/guitar-ai-model.pkl')

@app.route("/api/guitar", methods=["POST"])
def guitar_prediction():
    data = request.get_json()
    image = data["image"]
    split = image.split("base64,")[1]
    decoded = base64.b64decode(split)
    image_stream = BytesIO(decoded)
    img = Image.open(image_stream)
    img = img.convert('RGB')
    img = img.resize([800, 800])
    #img.show()
    prediction = model.predict(img)
    return jsonify({
        'result': str(prediction[0]),
        'confidence': str(prediction[2][prediction[1]].item())
        })