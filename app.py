from flask import Flask, render_template, request
import numpy as np

import pickle
cv = pickle.load(open('models/cv.pkl','rb'))
clf = pickle.load(open('models/clf.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def predict():
    email = request.form.get('email')
    X_input = cv.transform([email]).toarray()
    y_pred = clf.predict(X_input)
    if y_pred[0]==0:
        y_pred[0] = -1
    else:
        response = 1
    return render_template('index.html',response= y_pred[0])

if __name__ == "__main__":
    app.run(debug = True)
