from flask import Flask, request, jsonify
from fastai.vision.all import *
from PIL import Image
import __main__
from io import BytesIO
import base64
import pandas as pd
from fastai.tabular.all import *

def get_category(filename):
    return filename.parent.name.replace("_", " ")

app = Flask(__name__)
__main__.get_category = get_category

guitar_model = load_learner('api/guitar-ai-model.pkl')
cars_model = load_learner('api/car-linear-regression.pkl')

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
    prediction = guitar_model.predict(img)
    return jsonify({
        'result': str(prediction[0]),
        'confidence': str(prediction[2][prediction[1]].item())
        })

@app.route("/api/cars", methods=["POST"])
def cars_prediction():
    data = request.get_json()
    data_frame = pd.DataFrame(data, index=[0])
    data_frame['km_driven'].astype(float)
    # data_frame['km_driven'] = np.log(data_frame['km_driven'])
    data_frame['year'].astype(float)
    # data_frame['year'] = np.log(data_frame['year'])
    print(data_frame)
    test_dl = cars_model.dls.test_dl(data_frame)
    # to = TabularPandas(data_frame, procs=[Categorify, FillMissing,Normalize],
    #                cat_names=['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'model'],
    #                cont_names=['km_driven', 'year'],
    #                y_names='selling_price_norm')

    # dataloader = to.dataloaders(bs=256)
    # learn = tabular_learner(dataloader, metrics=[rmse], layers=[1000, 500])
    # learn.load(Path("api/car-linear-regression.pkl"))
    prediction = cars_model.get_preds(dl=test_dl)
    
    return jsonify({
        'result': str(prediction),
        })