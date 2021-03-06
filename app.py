import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('clf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/rules')  
def rules():
    return render_template('Rules.html')
    


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(float(x))  for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0])
    if output == 1:
        output = "High"
    elif output ==0:
        output = "Low"

    return render_template('result.html', prediction_text='Your chances for Getting Placed should be  {}'.format(output))
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)