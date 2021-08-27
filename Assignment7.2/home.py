import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle 

from tensorflow.keras.models import load_model
app = Flask(__name__)

model = load_model("Fish.h5",compile=False)
sc=pickle.load(open("scaling.pkl",'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    a = float(request.form['iweight'])
    b = float(request.form['ifirst'])
    c = float(request.form['isecond'])
    d = float(request.form['ithird'])
    e = float(request.form['iheight'])
    f = float(request.form['iwidth'])
    total = [[a,b,c,d,e,f]]
    print(total)
    yp=model.predict(sc.transform(total))
    
    species = [ "Bream","parkki", "perch", "pike", "roach", "smelt", "whitefish"]
    prediction=species[np.argmax(yp)]
    print(prediction)

    if(prediction=='Bream'):
        output = "The species is bream"
    
    elif(prediction=='parkki'):
        output = "The species is parkki"
        
    elif(prediction=='perch'):
        output = "The species is perch"
        
    elif(prediction=='pike'):
        output = "The species is pike"
        
    elif(prediction=='roach'):
        output = "The species is roach"
    
    elif(prediction=='smelt'):
        output = "The species is smelt"
    
    elif(prediction=='whitefish'):
        output = "The species is whitefish"
    else:
        output = "Species not found"


    return render_template('home.html', prediction_text='Result: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
    
    