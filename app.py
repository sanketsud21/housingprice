import pickle as pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 

app=Flask(__name__)
#loading model file 
regmodel=pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(list(data.values()))
    reshaped_data=np.array(list(data.values())).reshape(1,-1)
    output=regmodel.predict(reshaped_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    print(data)
    final_input=np.array(data).reshape(-1,1)
    print(final_input)
    output=regmodel.predict(final_input)
    return render_template("home.html",prediction_text="The house Price prediction is {}".format(output[0]))


if __name__=="__main__" :
    app.run(debug=True)
