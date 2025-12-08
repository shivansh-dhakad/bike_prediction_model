from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

# creating backend using flask
app=Flask(__name__)
# giving CORS permission to flask app
cors=CORS(app)

# load the model
pipeline=joblib.load('model.pkl')

# routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load-data',methods=['GET'])
def loaddata():
    encoder = pipeline.named_steps['preproc'].named_transformers_['cat'].named_steps['onehot'].categories_
    # print(encoder[1])
    return jsonify({'BrandData': encoder[1].tolist()})

@app.route('/predict',methods=['POST'])
def predict():
    data=request.get_json(force=True)
    input_data=pd.DataFrame({
        "kms_driven":[int(data['kms_driven'])],
        "owner":[data['owner']],
        "power":[int(data['power'])],
        "brand":[data['brand']],
        "Original Price":[int(data['original_price'])],
        "year":[int(data['year'])]
    })
    
    preds=pipeline.predict(input_data)
    return jsonify({'prediction':float(f"{preds[0]:.2f}")})


if __name__=='__main__':
    port =int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=port,debug=True)
