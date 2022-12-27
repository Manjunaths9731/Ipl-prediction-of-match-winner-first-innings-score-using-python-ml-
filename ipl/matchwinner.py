import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as m



@app.route('/',methods=['GET','POST'])
def submit():
  if request.method=='GET':
    return render_template('matchwinner.html')
  if request.method=='POST':
    data=pd.read_csv('cleaned.csv')
    to_predict_list = request.form.to_dict()
    to_predict_list = list(to_predict_list.values())
    to_predict_list = list(map(int, to_predict_list))
    to_predict = np.array(to_predict_list).reshape(1,-1)
    features=data.iloc[:,[2,4,5,6,7]]
    labels=data.iloc[:,10]
    features_train,features_test,labels_train,labels_test=train_test_split(features,labels,train_size=0.9,random_state=71)
    model=RandomForestClassifier(n_estimators=100,max_depth=25,random_state=129)
    model.fit(features_train,labels_train)
    h=model.predict(to_predict)
    u={1:"Mumbai Indians",2:"Kolkata Knight Riders",3:"Royal Challengers Bangalore",5:"Chennai Super Kings",
                 6:"Rajasthan Royals",7:"Delhi Daredevils",9:"Kings XI Punjab",
                 10:"Sunrisers Hyderabad"}
    return render_template('matchwinner.html',prediction=u[h[0]]+" is having Higher chance to win")

    
if __name__ == "__main__":
    app.run(debug=True)

