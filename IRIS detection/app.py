from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import subprocess

model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)


def handler(event, context):
    command = ["streamlit", "run", "iris_streamlit_app.py"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    return {
        'statusCode': 200,
        'body': stdout.decode('utf-8')
    }

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)



if __name__ == "__main__":
    app.run(host="0.0.0.0")












