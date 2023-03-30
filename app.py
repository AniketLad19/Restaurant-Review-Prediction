from flask import *
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)

def predictinpdata(input_df):
    vec=pickle.load(open('Vectorizer.pkl',"rb"))
    lr=pickle.load(open("Logistic.pkl","rb"))
    x=vec.transform([input_df]).toarray()
    ans=lr.predict(x)[0]
    if ans==0:
        return "Review is Negative."
    else:
        return "Review is positive."

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/reglink",methods=["POST"])
def getinputdata():
    review=request.form["Review"]        
    ans=predictinpdata(review)
    return render_template("display.html",data=ans)

    
if __name__ =='__main__':
    app.run(debug=True)